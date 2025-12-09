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
    Load F5TTS_v1_Base once and reuse.
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
    High-quality ASR for reference audio.
    Uses Whisper large-v3 via faster-whisper.
    Prefer GPU (cuda) when available.
    """
    global _asr_model
    if _asr_model is None:
        model_name = "large-v3"  # best quality; change to "medium.en" if VRAM is tight

        print("========== INIT ASR MODEL ==========")
        print("[ASR] torch.__version__:", torch.__version__)
        print("[ASR] torch.cuda.is_available():", torch.cuda.is_available())
        print("[ASR] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"

        print(f"[ASR] Selected device={device}, compute_type={compute_type}")

        _asr_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        print(
            f"[get_asr_model] Loaded faster-whisper '{model_name}' "
            f"on {device} ({compute_type})"
        )
        print("======== END INIT ASR MODEL ========")
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
# Helpers: text normalization + chunking
# ==============================
def _clean_for_tts(text: str) -> str:
    """
    Normalize text before sending to F5TTS:
      - remove parentheses and their contents
      - collapse whitespace
    This avoids some edge cases with punctuation.
    """
    # remove (...) blocks
    text = re.sub(r"\([^)]*\)", "", text)
    # collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    # Very simple sentence splitter based on punctuation.
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(
    text: str,
    max_chars: int = 200,   # smaller default for stability
    min_chars: int = 80,
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
    nfe_step: int,
    target_rms: float,
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
            nfe_step=nfe_step,        # quality (32/64 steps)
            target_rms=target_rms,    # volume
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
        "chunk_max_chars": 200,                    // optional
        "chunk_min_chars": 80,                     // optional
        "quality": "standard" | "premium",         // optional, affects nfe_step
        "volume": 1.0,                             // optional, 0.0–2.0
        "pause_s": 0.12                            // optional, seconds of silence between chunks
      }
    }
    """
    print("### handler version: f5tts_youtube_pauses_2025-12-09 ###")

    inp: Dict[str, Any] = job.get("input") or {}

    raw_text = (inp.get("text") or "").strip()
    if not raw_text:
        return {"error": "Missing 'text' in input."}

    print("[INPUT] text length:", len(raw_text))

    # ---- 1) Reference audio ----
    try:
        ref_path = _get_ref_audio_path(inp)
        print("[REF] Loaded reference audio:", ref_path)
    except Exception as e:
        return {"error": f"Failed to load reference audio: {e}"}

    # ---- 2) ref_text: use client-provided or ASR transcript ----
    ref_text = (inp.get("ref_text") or "").strip()
    language_hint = inp.get("language")  # e.g. "en"

    if ref_text:
        print(f"[REF] Using provided ref_text (len={len(ref_text)}): '{ref_text[:80]}'")
    else:
        print("[REF] No ref_text provided, running ASR...")
        try:
            ref_text = _transcribe_ref_audio(ref_path, language=language_hint)
        except Exception as e:
            return {"error": f"ASR transcription failed: {e}"}

    if not ref_text:
        print("[ERROR] ASR produced empty transcript.")
        return {
            "error": "ASR produced empty transcript. Please provide 'ref_text' manually."
        }

    # ---- 3) Chunking parameters ----
    max_chars = int(inp.get("chunk_max_chars", 200))
    min_chars = int(inp.get("chunk_min_chars", 80))
    chunks = _chunk_text(raw_text, max_chars=max_chars, min_chars=min_chars)
    print(
        f"[chunking] text length={len(raw_text)}, num_chunks={len(chunks)}, "
        f"max_chars={max_chars}, min_chars={min_chars}"
    )

    # ---- 4) Synthesis settings ----
    speed: float = float(inp.get("speed", 0.7))  # slower default
    remove_silence: bool = bool(inp.get("remove_silence", False))

    # Generation quality: "standard" (32) vs "premium" (64)
    quality = (inp.get("quality") or "standard").lower()
    if quality == "premium":
        nfe_step = 64
    else:
        nfe_step = 32

    # Volume: 0.0–2.0 → map linearly on top of default target_rms ≈ 0.1
    volume = float(inp.get("volume", 1.0))
    if volume < 0.0:
        volume = 0.0
    target_rms = 0.1 * volume

    # Pause between chunks (seconds)
    pause_seconds = float(inp.get("pause_s", 0.12))  # default 120ms

    print(
        f"[SYNTH] speed={speed}, remove_silence={remove_silence}, "
        f"quality={quality}, nfe_step={nfe_step}, volume={volume}, "
        f"target_rms={target_rms}, pause_s={pause_seconds}"
    )

    api = get_f5tts_model()

    # ---- 5) Run TTS per chunk and concatenate ----
    all_segments: List[np.ndarray] = []
    sr_final: Optional[int] = None

    try:
        for idx, chunk_text in enumerate(chunks, start=1):
            cleaned = _clean_for_tts(chunk_text)
            if not cleaned:
                print(f"[TTS] Chunk {idx} is empty after cleaning, skipping.")
                continue

            print(f"[TTS] Generating chunk {idx}/{len(chunks)}: {len(cleaned)} chars (cleaned)")
            try:
                audio_np = _tts_chunk(
                    api=api,
                    ref_path=ref_path,
                    ref_text=ref_text,
                    gen_text=cleaned,
                    speed=speed,
                    remove_silence=remove_silence,
                    nfe_step=nfe_step,
                    target_rms=target_rms,
                )
            except Exception as e:
                print(f"[TTS] Error on chunk {idx}: {e}")
                return {"error": f"TTS failed on chunk {idx}: {e}"}

            if audio_np.size == 0:
                print(f"[TTS] Chunk {idx} produced empty audio, skipping.")
                continue

            if sr_final is None:
                sr_final = 24000  # F5TTS_v1_Base default
                print(f"[TTS] Sample rate set to {sr_final}")

            all_segments.append(audio_np)

            # ---- Pause between chunks ----
            if idx < len(chunks) and pause_seconds > 0.0:
                pause_samples = int(sr_final * pause_seconds)
                if pause_samples > 0:
                    print(f"[TTS] Inserting {pause_samples} samples of silence between chunk {idx} and {idx+1}")
                    silence = np.zeros(pause_samples, dtype=np.int16)
                    all_segments.append(silence)

        if not all_segments:
            print("[ERROR] No audio was generated for any chunk.")
            return {"error": "No audio was generated for any chunk."}

        final_audio = _concat_audio(all_segments)
        print(f"[TTS] Final audio samples={final_audio.shape[0]}")

        # ---- 6) Write final WAV and base64-encode ----
        fd, final_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        print("[OUTPUT] Writing final WAV to:", final_path)

        try:
            wavfile.write(final_path, sr_final, final_audio)
            with open(final_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        finally:
            try:
                if os.path.exists(final_path):
                    os.remove(final_path)
                    print("[OUTPUT] Deleted temp WAV:", final_path)
            except Exception as e:
                print("[WARN] Failed to delete temp WAV:", e)

        print("[DONE] Request completed successfully.")

        return {
            "audio_base64": audio_b64,
            "sample_rate": sr_final,
            "ref_text_used": ref_text,
            "num_chunks": len(chunks),
            "quality": quality,
            "volume": volume,
            "pause_s": pause_seconds,
        }

    except Exception as e:
        print("[FATAL] Inference failed:", e)
        return {"error": f"Inference failed: {e}"}

    finally:
        try:
            if os.path.exists(ref_path):
                os.remove(ref_path)
                print("[CLEANUP] Deleted temp ref audio:", ref_path)
        except Exception as e:
            print("[WARN] Failed to delete temp ref audio:", e)


# ==============================
# RunPod serverless entry
# ==============================
runpod.serverless.start({"handler": generate_speech})
