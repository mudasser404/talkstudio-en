"""
RunPod Serverless Handler for F5-TTS
Production-ready version with comprehensive error handling
"""

# Patch weights_only issue before importing F5-TTS
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Apply patch
try:
    from patch_weights_only import patch_utils_infer

    patch_utils_infer()
except Exception as e:
    print(f"Warning: Could not apply patch: {e}")

import runpod
import torch
import tempfile
import base64
import requests
import subprocess
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT


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
        print(f"Using device: {device}")

        # Model configuration
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

        # Download and cache checkpoint
        ckpt_url = "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.safetensors"
        print("Downloading checkpoint from HuggingFace...")

        try:
            cached_file = cached_path(ckpt_url)

            # Rename to preserve .safetensors extension if needed
            if not str(cached_file).endswith(".safetensors"):
                new_path = str(cached_file) + ".safetensors"
                if not os.path.exists(new_path):
                    os.symlink(cached_file, new_path)
                ckpt_path = new_path
            else:
                ckpt_path = str(cached_file)

            print(f"Checkpoint cached at: {ckpt_path}")

            model = load_model(
                model_cls=DiT,
                model_cfg=model_cfg,
                ckpt_path=ckpt_path,
                mel_spec_type="vocos",
                vocab_file="",
                use_ema=True,
                device=device,
            )

            # Load vocoder
            print("Loading vocoder...")
            vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

            print("Models loaded successfully!")

        except Exception as e:
            print(f"Failed to load models: {str(e)}")
            raise


def generate_speech(job):
    """
    Generate speech from text using F5-TTS

    Expected input format:
    {
        "input": {
            "text": "Text to synthesize",
            "ref_audio_base64": "base64_encoded_audio",  # Optional
            "ref_audio_url": "https://example.com/audio.mp3",  # Optional
            "ref_text": "Reference text",  # Optional
            "remove_silence": false,
            "speed": 1.0
        }
    }
    """
    temp_files = []  # Track temp files for cleanup

    try:
        # Initialize models
        initialize_models()

        job_input = job["input"]

        # Extract parameters
        gen_text = job_input.get("text", "")
        ref_audio_b64 = job_input.get("ref_audio_base64", None)
        ref_audio_url = job_input.get("ref_audio_url", None)
        ref_text = job_input.get("ref_text", "")
        remove_silence_flag = job_input.get("remove_silence", False)
        speed = job_input.get("speed", 1.0)

        # Validate input
        if not gen_text:
            return {"error": "No text provided"}

        if not ref_audio_b64 and not ref_audio_url:
            return {"error": "Either ref_audio_base64 or ref_audio_url must be provided"}

        # Handle reference audio
        ref_audio_path = None

        if ref_audio_b64:
            # Decode base64 audio
            print("Decoding base64 audio...")
            try:
                audio_data = base64.b64decode(ref_audio_b64)
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio_data)
                temp_audio.close()
                ref_audio_path = temp_audio.name
                temp_files.append(ref_audio_path)
            except Exception as e:
                return {"error": f"Failed to decode base64 audio: {str(e)}"}

        elif ref_audio_url:
            # Download audio from URL
            print(f"Downloading audio from {ref_audio_url}...")
            try:
                response = requests.get(ref_audio_url, timeout=30)
                response.raise_for_status()

                # Determine file extension
                url_ext = os.path.splitext(ref_audio_url)[1].lower()

                # Save to temp file
                temp_download = tempfile.NamedTemporaryFile(
                    delete=False, suffix=url_ext or ".wav"
                )
                temp_download.write(response.content)
                temp_download.close()
                temp_files.append(temp_download.name)

                # Convert to WAV if needed
                if url_ext not in [".wav", ".wave"]:
                    print(f"Converting {url_ext} to WAV using ffmpeg...")

                    wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    wav_path = wav_temp.name
                    wav_temp.close()
                    temp_files.append(wav_path)

                    result = subprocess.run(
                        [
                            "ffmpeg",
                            "-i",
                            temp_download.name,
                            "-ar",
                            "24000",
                            "-ac",
                            "1",
                            "-y",
                            wav_path,
                            "-loglevel",
                            "error",
                        ],
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode != 0:
                        return {"error": f"FFmpeg conversion failed: {result.stderr}"}

                    ref_audio_path = wav_path
                    print(f"Conversion successful: {wav_path}")
                else:
                    ref_audio_path = temp_download.name

            except requests.RequestException as e:
                return {"error": f"Failed to download audio: {str(e)}"}
            except Exception as e:
                return {"error": f"Audio processing failed: {str(e)}"}

        # Preprocess reference audio
        print("Preprocessing reference audio...")
        try:
            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        except Exception as e:
            return {"error": f"Failed to preprocess audio: {str(e)}"}

        # Generate audio
        print(f"Generating speech for: {gen_text[:50]}...")
        try:
            generated_audio, final_sample_rate, _ = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                model_obj=model,
                vocoder=vocoder,
                mel_spec_type="vocos",
                speed=speed,
                device=device,
            )
        except Exception as e:
            return {"error": f"Speech generation failed: {str(e)}"}

        # Remove silence if requested
        if remove_silence_flag:
            print("Removing silence...")
            try:
                generated_audio = remove_silence_for_generated_wav(generated_audio)
            except Exception as e:
                print(f"Warning: Failed to remove silence: {str(e)}")

        # Convert to base64
        print("Encoding output audio...")
        try:
            # Save using scipy instead of torchaudio to avoid torchcodec
            import numpy as np
            from scipy.io import wavfile

            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            output_path = output_temp.name
            output_temp.close()
            temp_files.append(output_path)

            # Convert to int16
            audio_int16 = (generated_audio * 32767).astype(np.int16)
            wavfile.write(output_path, final_sample_rate, audio_int16)

            with open(output_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            print("✓ Audio generated successfully")

            return {"audio": audio_b64, "sample_rate": final_sample_rate, "text": gen_text}

        except Exception as e:
            return {"error": f"Failed to encode output audio: {str(e)}"}

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Unexpected error: {error_details}")
        return {"error": f"Unexpected error: {str(e)}"}

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_file}: {str(e)}")


# Start the serverless function
runpod.serverless.start({"handler": generate_speech})
