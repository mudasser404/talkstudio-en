"""
RunPod Serverless Handler for XTTS-v2 Voice Cloning
Optimized for speed with chunked processing
"""

import runpod
import base64
import os
import re
import time
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Accept Coqui TTS Terms of Service (bypass interactive prompt)
# Must patch BEFORE importing TTS
# ==============================
import TTS.utils.manage as tts_manage

def _patched_ask_tos(self, output_path):
    """Patched to auto-accept TOS without interactive prompt."""
    return True

tts_manage.ModelManager.ask_tos = _patched_ask_tos
logger.info("Patched Coqui TTS to auto-accept TOS")

# ==============================
# Global model
# ==============================
from TTS.api import TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tts_model: Optional[TTS] = None

# Speaker embedding cache
_speaker_embedding = None
_gpt_cond_latent = None
_cached_ref_path = None


def get_tts_model():
    """Load XTTS-v2 model once and reuse"""
    global _tts_model
    if _tts_model is None:
        logger.info("========== LOADING XTTS-v2 MODEL ==========")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Device: {DEVICE}")

        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

        logger.info("XTTS-v2 loaded successfully")
        logger.info("============================================")

    return _tts_model


def _get_speaker_embedding(tts, ref_path: str, gpt_cond_len: int = 6):
    """
    Compute speaker embedding once and cache it.
    Saves ~2-3 seconds per chunk.
    """
    global _speaker_embedding, _gpt_cond_latent, _cached_ref_path

    if _speaker_embedding is None or _gpt_cond_latent is None or _cached_ref_path != ref_path:
        logger.info(f"Computing speaker embedding (gpt_cond_len={gpt_cond_len}s)...")
        try:
            xtts_model = tts.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                audio_path=[ref_path],
                gpt_cond_len=gpt_cond_len,
                max_ref_length=10,
            )
            _speaker_embedding = speaker_embedding
            _gpt_cond_latent = gpt_cond_latent
            _cached_ref_path = ref_path
            logger.info("Speaker embedding cached successfully")
        except Exception as e:
            logger.error(f"Could not cache embedding: {e}")
            return None, None

    return _gpt_cond_latent, _speaker_embedding


# ==============================
# Text Processing
# ==============================
def _split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, max_chars: int = 245, min_chars: int = 100) -> List[str]:
    """
    Split text into chunks for TTS processing.
    IMPORTANT: XTTS v2 has a 250 character limit per chunk!
    """
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
# TTS Processing
# ==============================
def _tts_chunk_fast(
    xtts_model,
    text: str,
    language: str,
    gpt_cond_latent,
    speaker_embedding,
    speed: float = 1.0,
) -> np.ndarray:
    """Fast TTS with cached embeddings and optimized parameters"""
    out = xtts_model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        speed=speed,
        enable_text_splitting=False,
        temperature=0.65,
        top_k=30,
        top_p=0.8,
        repetition_penalty=5.0,
        length_penalty=1.0,
        do_sample=True,
    )
    return out["wav"]


def process_text(
    tts,
    ref_path: str,
    text: str,
    language: str = "en",
    speed: float = 1.0,
    gpt_cond_len: int = 6,
) -> Tuple[List[np.ndarray], int]:
    """Process text in chunks and return audio segments"""

    # Get cached embeddings
    gpt_cond_latent, speaker_embedding = _get_speaker_embedding(tts, ref_path, gpt_cond_len)

    if gpt_cond_latent is None:
        raise Exception("Failed to compute speaker embedding")

    xtts_model = tts.synthesizer.tts_model
    sr = tts.synthesizer.output_sample_rate

    # Chunk text
    chunks = _chunk_text(text, max_chars=245, min_chars=100)
    logger.info(f"Processing {len(chunks)} chunks ({len(text)} total chars)")

    all_segments = []
    start_time = time.time()

    for idx, chunk in enumerate(chunks):
        cleaned = _clean_for_tts(chunk)
        chunk_start = time.time()

        try:
            audio_np = _tts_chunk_fast(
                xtts_model, cleaned, language,
                gpt_cond_latent, speaker_embedding, speed
            )

            # Convert to int16
            if isinstance(audio_np, torch.Tensor):
                audio_np = audio_np.cpu().numpy()

            if audio_np.dtype in [np.float32, np.float64]:
                audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

            all_segments.append(audio_np)

            chunk_time = time.time() - chunk_start
            logger.info(f"Chunk {idx+1}/{len(chunks)}: {len(cleaned)} chars in {chunk_time:.2f}s")

        except Exception as e:
            logger.error(f"Error on chunk {idx+1}: {e}")
            raise

    total_time = time.time() - start_time
    logger.info(f"All {len(chunks)} chunks done in {total_time:.2f}s")

    return all_segments, sr


def combine_audio(segments: List[np.ndarray], sr: int, pause_seconds: float = 0.1) -> bytes:
    """Combine audio segments into single WAV file"""
    import struct
    import io

    # Add pauses between segments
    if pause_seconds > 0 and len(segments) > 1:
        pause_samples = int(sr * pause_seconds)
        silence = np.zeros(pause_samples, dtype=np.int16)

        with_pauses = []
        for i, seg in enumerate(segments):
            with_pauses.append(seg)
            if i < len(segments) - 1:
                with_pauses.append(silence)
        segments = with_pauses

    # Calculate total samples
    total_samples = sum(len(seg) for seg in segments)

    # Create WAV in memory
    buffer = io.BytesIO()

    # WAV header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 0))  # Placeholder
    buffer.write(b'WAVE')

    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))
    buffer.write(struct.pack('<H', 1))   # PCM
    buffer.write(struct.pack('<H', 1))   # Mono
    buffer.write(struct.pack('<I', sr))
    buffer.write(struct.pack('<I', sr * 2))
    buffer.write(struct.pack('<H', 2))
    buffer.write(struct.pack('<H', 16))

    # Data chunk
    buffer.write(b'data')
    data_size = total_samples * 2
    buffer.write(struct.pack('<I', data_size))

    # Write audio data
    for segment in segments:
        buffer.write(segment.tobytes())

    # Update file size
    file_size = buffer.tell()
    buffer.seek(4)
    buffer.write(struct.pack('<I', file_size - 8))

    return buffer.getvalue()


# ==============================
# RunPod Handler
# ==============================
def handler(job):
    """
    RunPod handler for XTTS-v2 voice cloning

    Input:
    {
        "input": {
            "text": "Text to synthesize",
            "reference_audio": "base64 encoded audio",
            "language": "en",
            "speed": 1.0,
            "gpt_cond_len": 6
        }
    }
    """
    logger.info("### handler version: xtts_v2_optimized_v1_2025-12-15 ###")

    try:
        job_input = job.get("input", {})

        text = job_input.get("text", "").strip()
        reference_audio_b64 = job_input.get("reference_audio")
        language = job_input.get("language", "en")
        speed = float(job_input.get("speed", 1.0))
        gpt_cond_len = int(job_input.get("gpt_cond_len", 6))

        if not text:
            return {"error": "Text is required"}

        if not reference_audio_b64:
            return {"error": "reference_audio is required"}

        logger.info(f"Text length: {len(text)} chars")
        logger.info(f"Language: {language}, Speed: {speed}")

        # Save reference audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(base64.b64decode(reference_audio_b64))
            ref_path = f.name

        try:
            # Get model
            tts = get_tts_model()

            # Process text
            start_time = time.time()
            segments, sr = process_text(tts, ref_path, text, language, speed, gpt_cond_len)

            # Combine audio
            audio_bytes = combine_audio(segments, sr)
            processing_time = time.time() - start_time

            # Encode as base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            logger.info(f"Total processing time: {processing_time:.2f}s")
            logger.info(f"Audio size: {len(audio_bytes) / 1024 / 1024:.2f}MB")

            return {
                "audio": audio_b64,
                "status": "success",
                "processing_time_seconds": round(processing_time, 2),
                "chars_processed": len(text),
                "sample_rate": sr
            }

        finally:
            # Cleanup
            if os.path.exists(ref_path):
                os.remove(ref_path)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}


# Pre-load model on startup
logger.info("Pre-loading XTTS-v2 model...")
try:
    get_tts_model()
    logger.info("Model pre-loaded successfully")
except Exception as e:
    logger.error(f"Failed to pre-load model: {e}")

# Start RunPod handler
runpod.serverless.start({"handler": handler})
