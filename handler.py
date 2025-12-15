import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile

# Global model variable
model = None

def load_model():
    """Load the Chatterbox TTS model."""
    global model
    if model is None:
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
        print(f"Chatterbox model loaded on {device}")
    return model

def handler(job):
    """
    RunPod serverless handler for Chatterbox TTS.

    Input:
        - text: Text to synthesize (required)
        - audio_prompt: Base64 encoded audio file for voice cloning (optional)
        - exaggeration: Exaggeration factor 0.0-1.0 (default: 0.5)
        - cfg_weight: CFG weight for generation (default: 0.5)

    Output:
        - audio: Base64 encoded WAV audio
        - sample_rate: Sample rate of the output audio
    """
    job_input = job["input"]

    # Get input parameters
    text = job_input.get("text")
    if not text:
        return {"error": "Text is required"}

    audio_prompt_b64 = job_input.get("audio_prompt")
    exaggeration = job_input.get("exaggeration", 0.5)
    cfg_weight = job_input.get("cfg_weight", 0.5)

    try:
        # Load model
        tts_model = load_model()

        # Handle audio prompt for voice cloning
        audio_prompt_path = None
        temp_file = None

        if audio_prompt_b64:
            # Decode base64 audio and save to temp file
            audio_bytes = base64.b64decode(audio_prompt_b64)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_bytes)
            temp_file.close()
            audio_prompt_path = temp_file.name

        # Generate speech
        wav = tts_model.generate(
            text=text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )

        # Clean up temp file
        if temp_file:
            os.unlink(temp_file.name)

        # Convert to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "audio": audio_b64,
            "sample_rate": tts_model.sr
        }

    except Exception as e:
        # Clean up temp file on error
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return {"error": str(e)}

# Pre-load model on cold start
print("Loading Chatterbox TTS model...")
load_model()
print("Model loaded successfully!")

runpod.serverless.start({"handler": handler})
