import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import runpod
from scipy.io import wavfile


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
    High-quality ASR for reference audio.
    Uses Whisper large-v3 via faster-whisper.
    """
    global _asr_model
    if _asr_model is None:
        model_name = "large-v3"  # best quality; change to "medium.en" if VRAM is tight
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        _asr_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        print(f"[get_asr_model] Loaded faster-whisper '{model_name}' on {device} ({compute_type})")
    return _asr_model


# ==============================
# Helpers: file handling
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
    ref_b64 = inp.get("ref_audio_base64")
    ref_url = inp.get("ref_audio_url")

    if ref_b64:
        return _save_b64_to_temp(ref_b64, suffix=".wav")
    if ref_url:
        return _download_to_temp(ref_url, suffix=".wav")

    raise ValueError("Provide either 'ref_audio_base64' or 'ref_audio_url' in input.")


# ==============================
# Helpers: ASR (transcriber)
# ==============================
def _transcribe_ref_audio(ref_path: str, language: Optional[str] = None) -> str:
    """
    Use faster-whisper (large-v3) to get transcript of the reference audio.
    This runs ONCE per job and is reused for all chunks.
    """
    model = get_asr_model()
    segments, info = model.transcribe(
        ref_path,
        language=language,  # e.g. "en"; or None for auto-detect
        beam_size=5,
    )
    parts = [seg.text.strip() for seg in segments if seg.text]
    transcript = " ".join(parts).strip()
    print(f"[ASR] language={info.language}, text='{transcript[:80]}...'")
    return transcript


# ==============================
# Helpers: text chunking for long YouTube scripts
# ==============================
def _split_into_sentences(text: str) -> List[str]:
    # Very simple sentence splitter based on punctuation.
    # Good enough for YouTube scripts; avoids extra deps.
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(
    text: str,
    max_chars: int = 400,
    min_chars: int = 150,
) -> List[str]:
    """
    Split long text into chunks around sentence boundaries.

    max_chars ~ how long each chunk should be (roughly).
    min_chars ~ don't create very tiny chunks unless unavoidable.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current = ""

    for s in sentences:
        if not current:
            current = s
            continue

        # If adding this sentence keeps us under max_chars, keep accumulating
        if len(current) + 1 + len(s) <= max_chars:
            current = current + " " + s
        else:
            # If current is big enough, push it as a chunk
            if len(current) >= min_chars:
                chunks.append(current)
                current = s
            else:
                # current too short, force-join and then flush
                current = current + " " + s
                chunks.append(current)
                current = ""

    if current.strip():
        chunks.append(current.strip())

    # Fallback: if somehow nothing, just return original text
    return chunks or [text.strip()]


# ==============================
# Helpers: TTS per chunk
# ==============================
def _tts_chunk(
    api: F5TTS,
    ref_path: str,
    ref_text: str,
    gen_text: str,
    speed: float,
    remove_silence: bool,
) -> np.ndarray:
    """
    Run F5TTS on a single text chunk and return it as a numpy array.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        _wav, sr, _ = api.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=out_path,
            file_spec=None,
        )
        sr_read, audio_np = wavfile.read(out_path)
        assert sr_read == sr, "Sample rate mismatch between infer() and file"
        return audio_np.astype(np.int16)  # assume 16-bit WAV
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass


def _concat_audio(segments: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple audio segments along time axis.
    Assumes all have same sample rate and channels.
    """
    if not segments:
        return np.zeros(0, dtype=np.int16)

    # Ensure all shapes are compatible (1D or (T, C))
    base = segments[0]
    if base.ndim == 1:
        return np.concatenate(segments, axis=0)

    # multi-channel: assume (T, C)
    return np.concatenate(segments, axis=0)


# ==============================
# Main RunPod job handler
# ==============================
def generate_speech(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Request:

    {
      "input": {
        "text": "Very long YouTube script here...",
        "ref_audio_url": "https://...wav",         // or "ref_audio_base64"
        "ref_audio_base64": "...",
        "ref_text": "",                            // optional; if empty, we'll transcribe
        "language": "en",                          // optional hint for ASR
        "remove_silence": true,                    // optional, default: false
        "speed": 0.7,                              // optional, default: 0.7 (slower, YouTube-friendly)
        "chunk_max_chars": 400,                    // optional
        "chunk_min_chars": 150                     // optional
      }
    }

    Response:

    {
      "audio_base64": "<base64 wav of ALL chunks>",
      "sample_rate": 24000,
      "ref_text_used": "transcribed or provided text",
      "num_chunks": 42
    }
    """
    inp: Dict[str, Any] = job.get("input") or {}

    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "Missing 'text' in input."}

    # ---- 1) Reference audio ----
    try:
        ref_path = _get_ref_audio_path(inp)
    except Exception as e:
        return {"error": f"Failed to load reference audio: {e}"}

    # ---- 2) ref_text: use client-provided or ASR transcript ----
    ref_text = (inp.get("ref_text") or "").strip()
    language_hint = inp.get("language")  # e.g. "en"

    if not ref_text:
        try:
            ref_text = _transcribe_ref_audio(ref_path, language=language_hint)
        except Exception as e:
            return {"error": f"ASR transcription failed: {e}"}

    if not ref_text:
        return {
            "error": "ASR produced empty transcript. Please provide 'ref_text' manually."
        }

    # ---- 3) Chunking parameters ----
    max_chars = int(inp.get("chunk_max_chars", 400))
    min_chars = int(inp.get("chunk_min_chars", 150))
    chunks = _chunk_text(text, max_chars=max_chars, min_chars=min_chars)
    print(f"[chunking] text length={len(text)}, num_chunks={len(chunks)}")

    # ---- 4) Synthesis settings ----
    # Default 0.7 is slower and more natural for YouTube narration.
    speed: float = float(inp.get("speed", 0.7))
    remove_silence: bool = bool(inp.get("remove_silence", False))

    api = get_f5tts_model()

    # ---- 5) Run TTS per chunk and concatenate ----
    all_segments: List[np.ndarray] = []
    sr_final: Optional[int] = None

    try:
        for idx, chunk_text in enumerate(chunks, start=1):
            print(f"[TTS] Generating chunk {idx}/{len(chunks)}: {len(chunk_text)} chars")
            audio_np = _tts_chunk(
                api=api,
                ref_path=ref_path,
                ref_text=ref_text,
                gen_text=chunk_text,
                speed=speed,
                remove_silence=remove_silence,
            )

            if audio_np.size == 0:
                continue

            # sample rate from first segment
            if sr_final is None:
                # We know F5TTS_v1_Base uses 24kHz, but we can read from API config if needed.
                sr_final = 24000

            all_segments.append(audio_np)

        if not all_segments:
            return {"error": "No audio was generated for any chunk."}

        final_audio = _concat_audio(all_segments)

        # ---- 6) Write final WAV and base64-encode ----
        fd, final_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            wavfile.write(final_path, sr_final, final_audio)
            with open(final_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        finally:
            try:
                if os.path.exists(final_path):
                    os.remove(final_path)
            except Exception:
                pass

        return {
            "audio_base64": audio_b64,
            "sample_rate": sr_final,
            "ref_text_used": ref_text,
            "num_chunks": len(chunks),
        }

    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    finally:
        try:
            if os.path.exists(ref_path):
                os.remove(ref_path)
        except Exception:
            pass


# ==============================
# RunPod serverless entry
# ==============================
runpod.serverless.start({"handler": generate_speech})
