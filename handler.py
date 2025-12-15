"""
RunPod Serverless Handler for Chatterbox TTS
Zero-shot voice cloning with high quality
"""

import runpod
import base64
import os
import time
import tempfile
import requests
import logging
import io

import torch
import torchaudio as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Global Settings
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None

# ==============================
# Model Loading
# ==============================
def load_model():
    """Load Chatterbox Turbo model"""
    global MODEL
    if MODEL is None:
        logger.info("========== LOADING Chatterbox Turbo ==========")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Device: {DEVICE}")

        from chatterbox.tts_turbo import ChatterboxTurboTTS
        MODEL = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

        logger.info("Chatterbox Turbo loaded successfully")
        logger.info("============================================")

    return MODEL


# ==============================
# RunPod Handler
# ==============================
def handler(job):
    """
    RunPod handler for Chatterbox TTS

    Input format:
    {
        "input": {
            "text": "Text to synthesize",
            "ref_audio_url": "URL to reference audio (optional)",
            "ref_audio_base64": "OR base64 encoded audio (optional)",
            "exaggeration": 0.5,
            "cfg_weight": 0.5
        }
    }
    """
    logger.info("### handler version: chatterbox_turbo_2025-12-16 ###")

    try:
        job_input = job.get("input", {})

        # Required
        text = job_input.get("text", "").strip()

        # Optional
        ref_audio_url = job_input.get("ref_audio_url")
        ref_audio_base64 = job_input.get("ref_audio_base64")
        exaggeration = float(job_input.get("exaggeration", 0.5))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))

        # Validation
        if not text:
            return {"error": "Text is required"}

        logger.info(f"Text length: {len(text)} chars")
        logger.info(f"Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}")

        # Load model
        model = load_model()

        start_time = time.time()
        ref_path = None

        try:
            # Download/decode reference audio if provided
            if ref_audio_url or ref_audio_base64:
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

            # Generate audio
            logger.info("Generating audio...")
            if ref_path:
                wav = model.generate(
                    text,
                    audio_prompt_path=ref_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            else:
                wav = model.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )

            # Save to buffer
            buffer = io.BytesIO()
            ta.save(buffer, wav, model.sr, format="wav")
            audio_bytes = buffer.getvalue()

            processing_time = time.time() - start_time
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Audio size: {len(audio_bytes) / 1024:.2f}KB")

            # Encode as base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "audio": audio_b64,
                "status": "success",
                "processing_time_seconds": round(processing_time, 2),
                "chars_processed": len(text),
                "model": "chatterbox_turbo"
            }

        finally:
            # Cleanup
            if ref_path and os.path.exists(ref_path):
                os.remove(ref_path)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


# Pre-load model on startup
logger.info("Pre-loading Chatterbox Turbo model...")
try:
    load_model()
    logger.info("Model pre-loaded successfully")
except Exception as e:
    logger.error(f"Failed to pre-load model: {e}")

# Start RunPod handler
runpod.serverless.start({"handler": handler})
