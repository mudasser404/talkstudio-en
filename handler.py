import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import runpod
import torch
from scipy.io import wavfile

# ==============================
# Patch F5-TTS to avoid torchaudio/torchcodec
# ==============================
def patch_utils_infer() -> None:
    """
    Replace torchaudio.load() with SciPy in utils_infer.py to avoid TorchCodec backend errors.
    """
    utils_file = Path(__file__).parent / "src" / "f5_tts" / "infer" / "utils_infer.py"

    if not utils_file.exists():
        print(f"[patch_utils_infer] Warning: {utils_file} not found, skipping patch.")
        return

    text = utils_file.read_text()

    if "audio, sr = torchaudio.load(ref_audio)" not in text:
        print("[patch_utils_infer] Already patched or no torchaudio.load() present.")
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
        "    if np.issubdtype(audio_np.dtype, np.integer):\n"
        "        max_val = np.iinfo(audio_np.dtype).max\n"
        "        audio_np = audio_np.astype(np.float32) / max_val\n"
        "    else:\n"
        "        audio_np = audio_np.astype(np.float32)\n"
        "\n"
        "    if audio_np.ndim == 1:\n"
        "        audio = torch.from_numpy(audio_np).unsqueeze(0)\n"
        "    else:\n"
        "        audio = torch.from_numpy(audio_np.T)\n"
    )

    text = text.replace(old_line, new_block)
    utils_file.write_text(text)
    print("[patch_utils_infer] ✓ utils_infer patched successfully.")


# Apply patch BEFORE importing F5TTS
patch_utils_infer()

from f5_tts.api import F5TTS
from faster_whisper import WhisperModel


# ==============================
# Global models
# ==============================
_f5tts_model: Optional[F5TTS] = None
_asr_model: Optional[WhisperModel] = None


def get_f5tts_model() -> F5TTS:
    """
    Load F5TTS_v1_Base once and reuse for all requests.
    """
    global _f5tts_model
    if _f5tts_model is None:
        print("========== INIT F5TTS MODEL ==========")
        print("[F5TTS] torch.cuda.is_available():", torch.cuda.is_available())
        print("[F5TTS] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

        _f5tts_model = F5TTS(
            model="F5TTS_v1_Base",
            hf_cache_dir="/root/.cache/huggingface/hub",
        )

        print("[get_f5tts_model] Loaded F5TTS_v1_Base")
        print("======== END INIT F5TTS MODEL ========")

    return _f5tts_model


def get_asr_model() -> WhisperModel:
    """
    ASR MUST run on CPU — GPU Whisper fails on RunPod Serverless (missing cuDNN).
    CPU mode is fast for 3–10 sec reference audio.
    """
    global _asr_model
    if _asr_model is None:
        model_name = "large-v3"

        print("========== INIT ASR MODEL ==========")
        print("[ASR] torch.__version__:", torch.__version__)
        print("[ASR] torch.cuda.is_available():", torch.cuda.is_available())
        print("[ASR] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

        # 🔥 FIX: Always use CPU for ASR
        device = "cpu"
        compute_type = "int8"
        print("[ASR] FORCING CPU MODE to avoid cuDNN crash")

        _asr_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        print(f"[get_asr_model] Loaded faster-whisper '{model_name}' on CPU (int8)")
        print("======== END INIT ASR MODEL ========")

    return _asr_model


# ==============================
# File Handling
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
    if inp.get("ref_audio_base64"):
        return _save_b64_to_temp(inp["ref_audio_base64"])

    if inp.get("ref_audio_url"):
        return _download_to_temp(inp["ref_audio_url"])

    raise ValueError("Provide either ref_audio_base64 or ref_audio_url")


# ==============================
# ASR (Transcription)
# ==============================
def _transcribe_ref_audio(ref_path: str, language: Optional[str] = None) -> str:
    model = get_asr_model()

    segments, info = model.transcribe(
        ref_path,
        language=language,
        beam_size=5,
    )

    parts = [seg.text.strip() for seg in segments if seg.text]
    transcript = " ".join(parts).strip()

    print(f"[ASR] language={info.language}, text='{transcript[:80]}...'")
    return transcript


# ==============================
# Text Processing
# ==============================
def _clean_for_tts(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(
    text: str,
    max_chars: int = 200,
    min_chars: int = 80,
) -> List[str]:

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
                chunks.append(current)
                current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


# ==============================
# TTS
# ==============================
def _tts_chunk(
    api: F5TTS,
    ref_path: str,
    ref_text: str,
    gen_text: str,
    speed: float,
    remove_silence: bool,
    nfe_step: int,
    target_rms: float,
) -> np.ndarray:

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        _, sr, _ = api.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            remove_silence=remove_silence,
            nfe_step=nfe_step,
            target_rms=target_rms,
            file_wave=out_path,
            file_spec=None,
        )

        sr_read, audio_np = wavfile.read(out_path)
        assert sr_read == sr
        return audio_np.astype(np.int16)

    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _concat_audio(segments: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(segments, axis=0)


# ==============================
# Main Handler
# ==============================
def generate_speech(job: Dict[str, Any]) -> Dict[str, Any]:

    print("### handler version: f5tts_youtube_pauses_2025-12-09 ###")

    inp = job.get("input", {})

    raw_text = (inp.get("text") or "").strip()
    print("[INPUT] text length:", len(raw_text))

    ref_path = _get_ref_audio_path(inp)

    # Reference text
    ref_text = (inp.get("ref_text") or "").strip()
    if not ref_text:
        ref_text = _transcribe_ref_audio(ref_path, language=inp.get("language"))

    # Chunking
    max_chars = int(inp.get("chunk_max_chars", 200))
    min_chars = int(inp.get("chunk_min_chars", 80))
    chunks = _chunk_text(raw_text, max_chars=max_chars, min_chars=min_chars)

    print(f"[chunking] chunks={len(chunks)} max_chars={max_chars} min_chars={min_chars}")

    # Synthesis settings
    speed = float(inp.get("speed", 0.7))
    remove_silence = bool(inp.get("remove_silence", False))

    quality = (inp.get("quality") or "standard").lower()
    nfe_step = 64 if quality == "premium" else 32

    volume = float(inp.get("volume", 1.0))
    volume = max(volume, 0.0)
    target_rms = 0.1 * volume

    pause_seconds = float(inp.get("pause_s", 0.12))

    print(
        f"[SYNTH] speed={speed}, quality={quality}, nfe_step={nfe_step}, "
        f"volume={volume}, pause_s={pause_seconds}"
    )

    api = get_f5tts_model()

    all_segments = []
    sr_final = 24000

    # Generate each chunk
    for idx, chunk in enumerate(chunks, start=1):
        cleaned = _clean_for_tts(chunk)
        print(f"[TTS] Chunk {idx}/{len(chunks)} ({len(cleaned)} chars)")

        audio_np = _tts_chunk(
            api,
            ref_path,
            ref_text,
            cleaned,
            speed,
            remove_silence,
            nfe_step,
            target_rms,
        )

        all_segments.append(audio_np)

        # Insert pause
        if idx < len(chunks) and pause_seconds > 0:
            pause_samples = int(sr_final * pause_seconds)
            silence = np.zeros(pause_samples, dtype=np.int16)
            all_segments.append(silence)

    # Final audio
    final_audio = _concat_audio(all_segments)

    # Output
    fd, final_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    wavfile.write(final_path, sr_final, final_audio)

    with open(final_path, "rb") as f:
        audio_bytes = f.read()

    os.remove(final_path)

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "sample_rate": sr_final,
        "ref_text_used": ref_text,
        "num_chunks": len(chunks),
        "quality": quality,
        "volume": volume,
        "pause_s": pause_seconds,
    }


# ==============================
# RunPod Entry
# ==============================
runpod.serverless.start({"handler": generate_speech})
