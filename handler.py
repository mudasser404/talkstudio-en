"""
RunPod Serverless Handler for OpenVoice V2 Voice Cloning
High quality voice cloning with better similarity
"""

import runpod
import base64
import os
import re
import time
import tempfile
import requests
from typing import List, Optional, Tuple

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Global Settings
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {}

# ==============================
# Model Loading
# ==============================
def load_openvoice():
    """Load OpenVoice V2 model"""
    if "openvoice" not in MODELS:
        logger.info("========== LOADING OpenVoice V2 ==========")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Device: {DEVICE}")

        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS as MeloTTS

        # Load ToneColorConverter
        ckpt_converter = '/app/checkpoints_v2/converter'
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=DEVICE)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        # Fix missing attributes in SynthesizerTrn model
        if hasattr(tone_color_converter, 'model'):
            if not hasattr(tone_color_converter.model, 'device'):
                tone_color_converter.model.device = torch.device(DEVICE)
            if not hasattr(tone_color_converter.model, 'version'):
                tone_color_converter.model.version = "v2"

        # Load MeloTTS for base audio generation
        melo_en = MeloTTS(language='EN', device=DEVICE)

        MODELS["openvoice"] = {
            "converter": tone_color_converter,
            "se_extractor": se_extractor,
            "melo_en": melo_en
        }

        logger.info("OpenVoice V2 loaded successfully")
        logger.info("============================================")

    return MODELS["openvoice"]


# ==============================
# Text Processing
# ==============================
def _split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, max_chars: int = 400, min_chars: int = 150) -> List[str]:
    """Split text into chunks for TTS processing"""
    sentences = _split_into_sentences(text)
    chunks, current = [], ""

    for s in sentences:
        if not current:
            current = s
            continue

        if len(current) + len(s) + 1 <= max_chars:
            current += " " + s
        else:
            if len(current) >= min_chars:
                chunks.append(current)
                current = s
            else:
                current += " " + s
                if len(current) >= max_chars:
                    chunks.append(current)
                    current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


def _clean_for_tts(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==============================
# Audio Processing
# ==============================
def remove_silence_from_audio(audio_path: str, output_path: str):
    """Remove silence from audio using pydub"""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_file(audio_path)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

        if nonsilent_ranges:
            output_audio = AudioSegment.empty()
            for start, end in nonsilent_ranges:
                output_audio += audio[start:end]
            output_audio.export(output_path, format="wav")
        else:
            # No silence detected, copy original
            import shutil
            shutil.copy(audio_path, output_path)
    except Exception as e:
        logger.warning(f"Could not remove silence: {e}")
        import shutil
        shutil.copy(audio_path, output_path)


def combine_audio_files(audio_files: List[str], output_path: str, pause_seconds: float = 0.15, sr: int = 24000):
    """Combine multiple audio files into one"""
    import soundfile as sf

    all_audio = []
    pause_samples = int(sr * pause_seconds)
    silence = np.zeros(pause_samples, dtype=np.float32)

    for i, audio_file in enumerate(audio_files):
        audio, file_sr = sf.read(audio_file)

        # Resample if needed
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        all_audio.append(audio)

        # Add pause between chunks
        if i < len(audio_files) - 1:
            all_audio.append(silence)

    # Concatenate all
    final_audio = np.concatenate(all_audio)
    sf.write(output_path, final_audio, sr)

    return output_path


# ==============================
# Voice Cloning
# ==============================
def clone_voice_openvoice(
    text: str,
    ref_audio_path: str,
    language: str = "en",
    speed: float = 1.0,
    chunk_max_chars: int = 400,
    chunk_min_chars: int = 150,
    pause_s: float = 0.15,
    remove_silence: bool = True,
    volume: float = 1.0,
) -> Tuple[bytes, float]:
    """
    Clone voice using OpenVoice V2
    Returns: (audio_bytes, processing_time)
    """
    start_time = time.time()

    model = load_openvoice()
    converter = model["converter"]
    se_extractor = model["se_extractor"]
    melo = model["melo_en"]

    # Get speaker IDs from MeloTTS
    speaker_ids = melo.hps.data.spk2id
    speaker_key = list(speaker_ids.keys())[0]  # Use first available speaker
    speaker_id = speaker_ids[speaker_key]

    logger.info(f"Using MeloTTS speaker: {speaker_key}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract target speaker embedding from reference audio
        logger.info("Extracting target speaker embedding...")
        target_se, _ = se_extractor.get_se(ref_audio_path, converter.model, vad=True)

        # Chunk text
        chunks = _chunk_text(text, max_chars=chunk_max_chars, min_chars=chunk_min_chars)
        logger.info(f"Processing {len(chunks)} chunks ({len(text)} total chars)")

        output_files = []

        for idx, chunk in enumerate(chunks):
            cleaned = _clean_for_tts(chunk)
            chunk_start = time.time()

            # Paths for this chunk
            base_audio_path = os.path.join(tmp_dir, f"base_{idx}.wav")
            converted_path = os.path.join(tmp_dir, f"converted_{idx}.wav")

            # Step 1: Generate base audio with MeloTTS
            melo.tts_to_file(cleaned, speaker_id, base_audio_path, speed=speed)

            # Step 2: Extract source speaker embedding
            source_se, _ = se_extractor.get_se(base_audio_path, converter.model, vad=True)

            # Step 3: Convert tone color to target voice
            converter.convert(
                audio_src_path=base_audio_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=converted_path,
            )

            # Step 4: Remove silence if requested
            if remove_silence:
                final_chunk_path = os.path.join(tmp_dir, f"final_{idx}.wav")
                remove_silence_from_audio(converted_path, final_chunk_path)
            else:
                final_chunk_path = converted_path

            output_files.append(final_chunk_path)

            chunk_time = time.time() - chunk_start
            logger.info(f"Chunk {idx+1}/{len(chunks)}: {len(cleaned)} chars in {chunk_time:.2f}s")

        # Combine all chunks
        logger.info("Combining audio chunks...")
        final_output = os.path.join(tmp_dir, "final_output.wav")
        combine_audio_files(output_files, final_output, pause_seconds=pause_s)

        # Apply volume if needed
        if volume != 1.0:
            import soundfile as sf
            audio, sr = sf.read(final_output)
            audio = audio * volume
            audio = np.clip(audio, -1.0, 1.0)
            sf.write(final_output, audio, sr)

        # Read final audio as bytes
        with open(final_output, 'rb') as f:
            audio_bytes = f.read()

    processing_time = time.time() - start_time
    logger.info(f"Total processing time: {processing_time:.2f}s")

    return audio_bytes, processing_time


# ==============================
# RunPod Handler
# ==============================
def handler(job):
    """
    RunPod handler for OpenVoice V2 voice cloning

    Input format:
    {
        "input": {
            "text": "Text to synthesize",
            "ref_audio_url": "URL to reference audio",
            "ref_audio_base64": "OR base64 encoded audio",
            "language": "en",
            "remove_silence": "true",
            "chunk_max_chars": 400,
            "chunk_min_chars": 150,
            "pause_s": 0.15,
            "speed": 0.95,
            "quality": "standard",
            "volume": 1
        }
    }
    """
    logger.info("### handler version: openvoice_v2_2025-12-15 ###")

    try:
        job_input = job.get("input", {})

        # Required
        text = job_input.get("text", "").strip()
        ref_audio_url = job_input.get("ref_audio_url")
        ref_audio_base64 = job_input.get("ref_audio_base64")

        # Optional with defaults
        language = job_input.get("language", "en")
        remove_silence = str(job_input.get("remove_silence", "true")).lower() == "true"
        chunk_max_chars = int(job_input.get("chunk_max_chars", 400))
        chunk_min_chars = int(job_input.get("chunk_min_chars", 150))
        pause_s = float(job_input.get("pause_s", 0.15))
        speed = float(job_input.get("speed", 1.0))
        volume = float(job_input.get("volume", 1.0))

        # Validation
        if not text:
            return {"error": "Text is required"}

        if not ref_audio_url and not ref_audio_base64:
            return {"error": "ref_audio_url or ref_audio_base64 is required"}

        logger.info(f"Text length: {len(text)} chars")
        logger.info(f"Language: {language}, Speed: {speed}, Volume: {volume}")
        logger.info(f"Chunk settings: max={chunk_max_chars}, min={chunk_min_chars}")
        logger.info(f"Remove silence: {remove_silence}, Pause: {pause_s}s")

        # Download/decode reference audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            if ref_audio_base64:
                logger.info("Using base64 reference audio")
                f.write(base64.b64decode(ref_audio_base64))
            else:
                logger.info(f"Downloading reference audio: {ref_audio_url}")
                resp = requests.get(ref_audio_url, timeout=60)
                resp.raise_for_status()
                f.write(resp.content)
            ref_path = f.name

        try:
            # Clone voice
            audio_bytes, processing_time = clone_voice_openvoice(
                text=text,
                ref_audio_path=ref_path,
                language=language,
                speed=speed,
                chunk_max_chars=chunk_max_chars,
                chunk_min_chars=chunk_min_chars,
                pause_s=pause_s,
                remove_silence=remove_silence,
                volume=volume,
            )

            # Encode as base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            logger.info(f"Audio size: {len(audio_bytes) / 1024 / 1024:.2f}MB")

            return {
                "audio": audio_b64,
                "status": "success",
                "processing_time_seconds": round(processing_time, 2),
                "chars_processed": len(text),
                "model": "openvoice_v2"
            }

        finally:
            # Cleanup
            if os.path.exists(ref_path):
                os.remove(ref_path)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


# Pre-load model on startup
logger.info("Pre-loading OpenVoice V2 model...")
try:
    load_openvoice()
    logger.info("Model pre-loaded successfully")
except Exception as e:
    logger.error(f"Failed to pre-load model: {e}")

# Start RunPod handler
runpod.serverless.start({"handler": handler})
