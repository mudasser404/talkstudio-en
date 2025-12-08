import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import runpod


# -------------------------
# Patch F5-TTS to avoid torchaudio/torchcodec for audio loading
# -------------------------
def patch_utils_infer() -> None:
    """
    Patch src/f5_tts/infer/utils_infer.py so that:

      audio, sr = torchaudio.load(ref_audio)

    is replaced with a SciPy-based loader, avoiding TorchCodec entirely.
    """
    # handler.py will be copied into /workspace/F5-TTS
    utils_file = Path(__file__).parent / "src" / "f5_tts" / "infer" / "utils_infer.py"

    if not utils_file.exists():
        print(f"[patch_utils_infer] Warning: {utils_file} not found, skipping patch.")
        return

    text = utils_file.read_text()

    old_snippet = (
        "def infer_process(\n"
        "    ref_audio,\n"
        "    ref_text,\n"
        "    gen_text,\n"
        "    model_obj,\n"
        "    vocoder,\n"
        "    mel_spec_type=mel_spec_type,\n"
        "    show_info=print,\n"
        "    progress=tqdm,\n"
        "    target_rms=target_rms,\n"
        "    cross_fade_duration=cross_fade_duration,\n"
        "    nfe_step=nfe_step,\n"
        "    cfg_strength=cfg_strength,\n"
        "    sway_sampling_coef=sway_sampling_coef,\n"
        "    speed=speed,\n"
        "    fix_duration=fix_duration,\n"
        "    device=device,\n"
        "):\n"
        "    # Split the input text into batches\n"
        "    audio, sr = torchaudio.load(ref_audio)\n"
    )

    new_snippet = (
        "def infer_process(\n"
        "    ref_audio,\n"
        "    ref_text,\n"
        "    gen_text,\n"
        "    model_obj,\n"
        "    vocoder,\n"
        "    mel_spec_type=mel_spec_type,\n"
        "    show_info=print,\n"
        "    progress=tqdm,\n"
        "    target_rms=target_rms,\n"
        "    cross_fade_duration=cross_fade_duration,\n"
        "    nfe_step=nfe_step,\n"
        "    cfg_strength=cfg_strength,\n"
        "    sway_sampling_coef=sway_sampling_coef,\n"
        "    speed=speed,\n"
        "    fix_duration=fix_duration,\n"
        "    device=device,\n"
        "):\n"
        "    # Split the input text into batches\n"
        "    # Load audio with SciPy instead of torchaudio.load to avoid TorchCodec backend\n"
        "    from scipy.io import wavfile\n"
        "    import numpy as np\n"
        "    import torch\n"
        "\n"
        "    sr, audio_np = wavfile.read(ref_audio)\n"
        "\n"
        "    # Convert to float32 in [-1, 1]\n"
        "    if np.issubdtype(audio_np.dtype, np.integer):\n"
        "        max_val = np.iinfo(audio_np.dtype).max\n"
        "        audio_np = audio_np.astype(np.float32) / max_val\n"
        "    else:\n"
        "        audio_np = audio_np.astype(np.float32)\n"
        "\n"
        "    if audio_np.ndim == 1:\n"
        "        audio = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)\n"
        "    else:\n"
        "        # (T, C) -> (C, T)\n"
        "        audio = torch.from_numpy(audio_np.T)\n"
    )

    if "audio, sr = torchaudio.load(ref_audio)" not in text:
        print("[patch_utils_infer] No torchaudio.load() found in utils_infer.py (already patched?).")
        return

    text = text.replace(old_snippet, new_snippet)
    utils_file.write_text(text)
    print("[patch_utils_infer] ✓ Patched utils_infer.py to use SciPy audio loading.")


# Apply patch BEFORE importing F5TTS
patch_utils_infer()

from f5_tts.api import F5TTS  # noqa: E402  (import after patch)


# -------------------------
# Global model (reused across jobs)
# -------------------------
_f5tts_model: Optional[F5TTS] = None


def get_model() -> F5TTS:
    """
    Lazily initialize F5TTS v1 Base and keep it in memory for warm workers.
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
      "ref_text_used": "some text"
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

    # IMPORTANT:
    # We DO NOT call api.transcribe() here to avoid hitting Whisper + torchcodec.
    # Instead, we always provide a non-empty ref_text so F5-TTS never tries to
    # do its own ASR inside preprocess_ref_audio_text().
    ref_text = inp.get("ref_text")
    if not ref_text or not str(ref_text).strip():
        ref_text = "This is the reference audio."

    remove_silence: bool = bool(inp.get("remove_silence", False))
    speed: float = float(inp.get("speed", 1.0))

    api = get_model()

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
