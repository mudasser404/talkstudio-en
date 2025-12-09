import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import runpod


# ==============================
# Patch F5-TTS to avoid torchaudio/torchcodec
# ==============================
def patch_utils_infer() -> None:
    """
    Patch src/f5_tts/infer/utils_infer.py so that:

        audio, sr = torchaudio.load(ref_audio)

    is replaced with a SciPy-based loader, avoiding TorchCodec completely.
    """
    utils_file = Path(__file__).parent / "src" / "f5_tts" / "infer" / "utils_infer.py"

    if not utils_file.exists():
        print(f"[patch_utils_infer] Warning: {utils_file} not found, skipping patch.")
        return

    text = utils_file.read_text()

    if "audio, sr = torchaudio.load(ref_audio)" not in text:
        print("[patch_utils_infer] No torchaudio.load() found (maybe already patched).")
        return

    old_line = "    audio, sr = torchaudio.load(ref_audio)"

    new_block = (
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

    text = text.replace(old_line, new_block)
    utils_file.write_text(text)
    print("[patch_utils_infer] ✓ Patched utils_infer.py to use SciPy audio loading.")


# Apply patch BEFORE importing F5TTS
patch_utils_infer()

from f5_tts.api import F5TTS  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


# ==============================
# Global models (reused across jobs)
# ==============================
_f5tts_model: Optional[F5TTS] = None
_asr_model: Optional[WhisperModel] = None


def get_f5tts_model() -> F5TTS:
    """
    Lazily initialize F5TTS v1 Base and keep it in memory for warm workers.
    """
    global _f5tts_model
    if _f5tts_model is None:
        _f5tts_model = F5TTS(
            model="F5TTS_v1_Base",
            hf_cache_dir="/root/.cache/huggingface/hub",
        )
        print("[get_f5tts_model] Loaded F5TTS_v1_Base")
    return _f5tts_model


def get_asr_model() -> WhisperModel:
    """
    Lazily initialize a faster-whisper model for transcribing reference audio.
    Using a medium-sized English model for decent quality.
    """
    global _asr_model
    if _asr_model is None:
        # Use GPU if available, otherwise CPU
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        _asr_model = WhisperModel(
            "medium.en",  # you can change to "small.en" if you want faster but slightly lower quality
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        print(f"[get_asr_model] Loaded faster-whisper 'medium.en' on {device}")
    return _asr_model


# ==============================
# Helper functions
# ==============================
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


def _transcribe_ref_audio(ref_path: str, language: Optional[str] = None) -> str:
    """
    Use faster-whisper to get the transcript of the reference audio.
    This becomes ref_text for F5TTS.
    """
    model = get_asr_model()
    segments, info = model.transcribe(
        ref_path,
        beam_size=5,
        language=language,  # None = auto-detect; use "en" to force English
    )
    text_parts = [seg.text.strip() for seg in segments if seg.text]
    transcript = " ".join(text_parts).strip()
    print(f"[ASR] Detected language={info.language}, text='{transcript[:80]}...'")
    return transcript


# ==============================
# Main RunPod job handler
# ==============================
def generate_speech(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected RunPod request format:

    {
      "input": {
        "text": "Hello, this is F5TTS v1 base running on RunPod.",
        "ref_audio_url": "https://...wav",      # or "ref_audio_base64"
        "ref_audio_base64": "...",              # base64-encoded wav (optional if url is used)
        "ref_text": "",                         # optional, if empty we auto-transcribe with Whisper
        "language": "en",                       # optional hint for ASR
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

    # 1) Get reference audio
    try:
        ref_path = _get_ref_audio_path(inp)
    except Exception as e:
        return {"error": f"Failed to load reference audio: {e}"}

    # 2) Get or generate ref_text
    ref_text = (inp.get("ref_text") or "").strip()
    language_hint = inp.get("language")  # e.g. "en"; or None for auto-detect

    if not ref_text:
        try:
            ref_text = _transcribe_ref_audio(ref_path, language=language_hint)
        except Exception as e:
            print(f"[WARN] ASR failed: {e}")
            ref_text = ""

    if not ref_text:
        # If ASR fails and nothing was provided, we still give a dummy text,
        # but this should be rare now.
        ref_text = "This is the reference audio."

    remove_silence: bool = bool(inp.get("remove_silence", False))
    speed: float = float(inp.get("speed", 1.0))

    api = get_f5tts_model()

    # 3) Prepare output wav path
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        # 4) Run F5-TTS inference
        wav, sr, _ = api.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=out_path,  # also writes wav to file
            file_spec=None,
        )

        # 5) Read generated wav & base64-encode it
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


# ==============================
# RunPod serverless entry
# ==============================
runpod.serverless.start({"handler": generate_speech})
