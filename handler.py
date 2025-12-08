import base64
import os
import tempfile
from typing import Any, Dict, Optional

import requests
import runpod
from f5_tts.api import F5TTS


# -------------------------
# Global model (reused across jobs)
# -------------------------
_f5tts_model: Optional[F5TTS] = None


def get_model() -> F5TTS:
    """
    Lazily initialize F5TTS v1 Base and keep it in memory for warm workers.
    This uses the official F5TTS API class and loads:
      - config:  configs/F5TTS_v1_Base.yaml
      - ckpt:    hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors
    """
    global _f5tts_model
    if _f5tts_model is None:
        _f5tts_model = F5TTS(
            model="F5TTS_v1_Base",
            hf_cache_dir="/root/.cache/huggingface/hub",
        )
    return _f5tts_model


# -------------------------
# Helpers
# -------------------------
def _save_b64_to_temp(b64_str: str, suffix: str = ".wav") -> str:
    audio_bytes = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path


def _download_to_temp(url: str, suffix: str = ".wav") -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def _get_ref_audio_path(inp: Dict[str, Any]) -> str:
    """Return path to a local temp wav file from either base64 or URL."""
    ref_b64 = inp.get("ref_audio_base64")
    ref_url = inp.get("ref_audio_url")

    if ref_b64:
        return _save_b64_to_temp(ref_b64, suffix=".wav")
    if ref_url:
        return _download_to_temp(ref_url, suffix=".wav")

    raise ValueError("Provide either 'ref_audio_base64' or 'ref_audio_url' in input.")


# -------------------------
# Main RunPod job handler
# -------------------------
def generate_speech(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected RunPod request format:

    {
      "input": {
        "text": "Hello, this is F5TTS v1 base running on RunPod.",
        "ref_audio_url": "https://...wav",      # or "ref_audio_base64"
        "ref_audio_base64": "...",              # base64-encoded wav (optional if url is used)
        "ref_text": "",                         # optional, transcript of reference audio
        "remove_silence": true,                 # optional, default: false
        "speed": 1.0                            # optional, default: 1.0
      }
    }

    Response:

    {
      "audio_base64": "<base64 wav>",
      "sample_rate": 24000,
      "ref_text_used": "transcribed or provided text"
    }
    """
    inp: Dict[str, Any] = job.get("input") or {}

    text = inp.get("text")
    if not text:
        return {"error": "Missing 'text' in input."}

    try:
        ref_path = _get_ref_audio_path(inp)
    except Exception as e:
        return {"error": f"Failed to load reference audio: {e}"}

    ref_text: str = inp.get("ref_text") or ""

    remove_silence: bool = bool(inp.get("remove_silence", False))
    speed: float = float(inp.get("speed", 1.0))

    api = get_model()

    # If no ref_text is provided, use the built-in ASR (Whisper) to transcribe
    if not ref_text:
        try:
            segments = api.transcribe(ref_path)  # returns list of { "text": ... }
            ref_text = " ".join(seg.get("text", "") for seg in segments).strip()
        except Exception as e:
            # If transcription fails, still try inference with empty ref_text
            ref_text = ""
            print(f"[WARN] Transcription failed: {e}")

    # Prepare output wav path
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        # Run F5-TTS inference
        wav, sr, _ = api.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=out_path,  # also writes wav to file
            file_spec=None,
        )

        # Read generated wav & base64-encode it
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": audio_b64,
            "sample_rate": sr,
            "ref_text_used": ref_text,
        }

    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    finally:
        # Clean up temp files
        try:
            if os.path.exists(ref_path):
                os.remove(ref_path)
        except Exception:
            pass

        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass


# -------------------------
# RunPod serverless entry
# -------------------------
runpod.serverless.start({"handler": generate_speech})
