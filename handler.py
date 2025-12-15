"""
RunPod Serverless Handler for Speaker Embedding Based Voice Cloning
Supports: OpenVoice, XTTS-v2, and Kokoro-82M
"""

import runpod
import base64
import os
import io
import torch
import torchaudio
import tempfile
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
MODELS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_openvoice():
    """Load OpenVoice V2 model"""
    if "openvoice" not in MODELS:
        logger.info("Loading OpenVoice V2...")
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        ckpt_converter = 'checkpoints_v2/converter'

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=DEVICE)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        MODELS["openvoice"] = {
            "converter": tone_color_converter,
            "se_extractor": se_extractor
        }
        logger.info("OpenVoice V2 loaded successfully")
    return MODELS["openvoice"]


def load_xtts():
    """Load XTTS-v2 model"""
    if "xtts" not in MODELS:
        logger.info("Loading XTTS-v2...")
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        MODELS["xtts"] = tts
        logger.info("XTTS-v2 loaded successfully")
    return MODELS["xtts"]


def load_kokoro():
    """Load Kokoro-82M model"""
    if "kokoro" not in MODELS:
        logger.info("Loading Kokoro-82M...")
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code='a')  # 'a' for American English
        MODELS["kokoro"] = pipeline
        logger.info("Kokoro-82M loaded successfully")
    return MODELS["kokoro"]


def audio_to_base64(audio_path: str) -> str:
    """Convert audio file to base64 string"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def base64_to_audio(base64_string: str, output_path: str) -> str:
    """Convert base64 string to audio file"""
    audio_bytes = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return output_path


def clone_with_openvoice(text: str, reference_audio: str, language: str = "en") -> str:
    """Clone voice using OpenVoice V2"""
    model = load_openvoice()
    converter = model["converter"]
    se_extractor = model["se_extractor"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract speaker embedding from reference audio
        target_se, _ = se_extractor.get_se(reference_audio, converter.model, vad=False)

        # Generate base TTS (using MeloTTS as base)
        from melo.api import TTS as MeloTTS
        melo = MeloTTS(language=language, device=DEVICE)
        speaker_ids = melo.hps.data.spk2id

        base_audio_path = os.path.join(tmp_dir, "base.wav")
        output_path = os.path.join(tmp_dir, "output.wav")

        # Generate base audio
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            break

        melo.tts_to_file(text, speaker_id, base_audio_path, speed=1.0)

        # Get source speaker embedding
        source_se, _ = se_extractor.get_se(base_audio_path, converter.model, vad=False)

        # Convert tone color
        converter.convert(
            audio_src_path=base_audio_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
        )

        return audio_to_base64(output_path)


def clone_with_xtts(text: str, reference_audio: str, language: str = "en") -> str:
    """Clone voice using XTTS-v2"""
    tts = load_xtts()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "output.wav")

        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_audio,
            language=language
        )

        return audio_to_base64(output_path)


def clone_with_kokoro(text: str, voice: str = "af_heart") -> str:
    """Generate speech using Kokoro-82M (fast, no cloning but high quality)"""
    pipeline = load_kokoro()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "output.wav")

        # Generate audio
        generator = pipeline(text, voice=voice)

        # Collect all audio chunks
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)

        # Concatenate and save
        import numpy as np
        full_audio = np.concatenate(all_audio)

        import soundfile as sf
        sf.write(output_path, full_audio, 24000)

        return audio_to_base64(output_path)


def handler(job):
    """
    RunPod handler function

    Input:
    {
        "input": {
            "model": "openvoice" | "xtts" | "kokoro",
            "text": "Text to synthesize",
            "reference_audio": "base64 encoded audio" (optional for kokoro),
            "language": "en" | "hi" | "zh" | etc.,
            "voice": "af_heart" (only for kokoro)
        }
    }

    Output:
    {
        "audio": "base64 encoded wav audio",
        "model_used": "model name"
    }
    """
    try:
        job_input = job["input"]

        model_name = job_input.get("model", "xtts")
        text = job_input.get("text", "")
        language = job_input.get("language", "en")
        reference_audio_b64 = job_input.get("reference_audio", None)
        voice = job_input.get("voice", "af_heart")

        if not text:
            return {"error": "Text is required"}

        logger.info(f"Processing request - Model: {model_name}, Text: {text[:50]}...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save reference audio if provided
            reference_audio_path = None
            if reference_audio_b64:
                reference_audio_path = os.path.join(tmp_dir, "reference.wav")
                base64_to_audio(reference_audio_b64, reference_audio_path)

            # Process based on model
            if model_name == "openvoice":
                if not reference_audio_path:
                    return {"error": "OpenVoice requires reference_audio"}
                audio_b64 = clone_with_openvoice(text, reference_audio_path, language)

            elif model_name == "xtts":
                if not reference_audio_path:
                    return {"error": "XTTS requires reference_audio"}
                audio_b64 = clone_with_xtts(text, reference_audio_path, language)

            elif model_name == "kokoro":
                audio_b64 = clone_with_kokoro(text, voice)

            else:
                return {"error": f"Unknown model: {model_name}. Supported: openvoice, xtts, kokoro"}

        return {
            "audio": audio_b64,
            "model_used": model_name,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}


# For local testing
if __name__ == "__main__":
    # Test locally
    test_job = {
        "input": {
            "model": "kokoro",
            "text": "Hello, this is a test of the voice cloning system.",
            "voice": "af_heart"
        }
    }
    result = handler(test_job)
    print(result)


runpod.serverless.start({"handler": handler})
