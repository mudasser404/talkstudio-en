import base64
import os
import tempfile

import requests
import runpod
from f5_tts.api import F5TTS

# Global model instance (kept between warm invocations)
f5tts = None


def get_model():
    """
    Lazily initialize F5TTS_v1_Base and keep it in memory.
    F5TTS will:
      - load configs/F5TTS_v1_Base.yaml
      - download the v1 base checkpoint from Hugging Face if not cached
    """
    global f5tts
    if f5tts is None:
        f5tts = F5TTS(
            model="F5TTS_v1_Base",   # default anyway, but explicit
            hf_cache_dir="/root/.cache/huggingface/hub",
        )
    return f5tts


def _save_b64_to_temp(b64_str, suffix=".wav"):
    audio_bytes = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path


def _download_to_temp(url, suffix=".wav"):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def generate_speech(job: dict):
    """
    RunPod will call this with a job like:

    {
      "input": {
        "text": "Text to synthesize",
        "ref_audio_base64": "...",      # OR
        "ref_audio_url": "https://...",
        "ref_text": "transcript of ref audio (optional)",
        "remove_silence": false,
        "speed": 1.0
      }
    }
    """
    inp = job.get("input") or {}

    text = inp.get("text")
    ref_b64 = inp.get("ref_audio_base64")
    ref_url = inp.get("ref_audio_url")
    ref_text = inp.get("ref_text", "")

    if not text:
        return {"error": "Missing 'text' in input."}

    if not (ref_b64 or ref_url):
        return {"error": "Provide either 'ref_audio_base64' or 'ref_audio_url'."}

    remove_silence = bool(inp.get("remove_silence", False))
    speed = float(inp.get("speed", 1.0))

    ref_path = None
    out_path = None

    try:
        # 1) Reference audio → temp file
        if ref_b64:
            ref_path = _save_b64_to_temp(ref_b64, suffix=".wav")
        else:
            ref_path = _download_to_temp(ref_url, suffix=".wav")

        # 2) Output wav path
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        # 3) F5TTS inference
        api = get_model()
        wav, sr, _ = api.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=out_path,  # save wav to file as well
            file_spec=None,
        )

        # 4) Read generated wav and return as base64
        with open(out_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": audio_b64,
            "sample_rate": sr,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        if out_path and os.path.exists(out_path):
            os.remove(out_path)


# RunPod serverless entrypoint
runpod.serverless.start({"handler": generate_speech})
