"""
RunPod Serverless Handler for F5-TTS
"""

import runpod
import torch
import torchaudio
import numpy as np
import tempfile
import base64
from pathlib import Path
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT


# Global variables for model caching
model = None
vocoder = None
device = None


def initialize_models():
    """Initialize models once during cold start"""
    global model, vocoder, device

    if model is None:
        print("Loading F5-TTS model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4
        )

        model = DiT(**model_cfg).to(device)

        # Load checkpoint
        ckpt_path = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
        print(f"Loading checkpoint from {ckpt_path}")
        model = load_model(model, ckpt_path, device)

        # Load vocoder
        print("Loading vocoder...")
        vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

        print("Models loaded successfully!")


def generate_speech(job):
    """
    Generate speech from text using F5-TTS

    Expected input format:
    {
        "input": {
            "text": "Text to synthesize",
            "ref_audio": "base64_encoded_audio",  # Reference audio (optional)
            "ref_text": "Reference text",  # Reference text (optional)
            "model_type": "F5-TTS",  # F5-TTS or E2-TTS
            "remove_silence": true,
            "speed": 1.0
        }
    }
    """
    try:
        initialize_models()

        job_input = job["input"]

        # Extract parameters
        gen_text = job_input.get("text", "")
        ref_audio_b64 = job_input.get("ref_audio", None)
        ref_text = job_input.get("ref_text", "")
        remove_silence = job_input.get("remove_silence", True)
        speed = job_input.get("speed", 1.0)

        if not gen_text:
            return {"error": "No text provided"}

        # Handle reference audio
        ref_audio_path = None
        if ref_audio_b64:
            # Decode base64 audio
            audio_data = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                ref_audio_path = temp_audio.name

        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(
            ref_audio_path, ref_text, device=device
        )

        # Generate audio
        print(f"Generating speech for: {gen_text[:50]}...")

        generated_audio, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="vocos",
            speed=speed,
            device=device,
        )

        # Remove silence if requested
        if remove_silence:
            generated_audio = remove_silence_for_generated_wav(generated_audio)

        # Convert to base64
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            torchaudio.save(
                temp_output.name,
                torch.tensor(generated_audio).unsqueeze(0),
                final_sample_rate,
            )

            with open(temp_output.name, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "audio": audio_b64,
            "sample_rate": final_sample_rate,
            "text": gen_text
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}


# Start the serverless function
runpod.serverless.start({"handler": generate_speech})
