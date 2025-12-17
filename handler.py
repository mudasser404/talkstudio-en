import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import urllib.request
import re
import time

from huggingface_hub import login

# Optional: enable only for debugging (VERY slow if enabled)
if os.getenv("CUDA_LAUNCH_BLOCKING", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

# TensorFloat32 speedup (safe on RTX 4090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

model = None
hf_logged_in = False


def ensure_hf_login():
    global hf_logged_in
    if hf_logged_in:
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Hugging Face login: OK")
    else:
        print("WARNING: No HF token found (Turbo weights may fail to download).")

    hf_logged_in = True


def load_model():
    global model
    if model is None:
        ensure_hf_login()
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        model = ChatterboxTurboTTS.from_pretrained(device=device)
        print(f"Loaded class: {type(model).__name__}")
        print(f"Chatterbox TURBO model loaded on {device}")

    return model


def download_audio(url, output_path):
    print(f"Downloading audio from: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to: {output_path}")


def convert_to_wav(input_path, output_path):
    waveform, sample_rate = torchaudio.load(input_path)

    # mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 24k
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 24000)
    print(f"Converted to WAV: {output_path}")


def chunk_text(text, max_chars=2000, min_chars=800):
    """
    Strong chunker:
    - Targets near max_chars chunks (prevents fragmentation)
    - Only cuts at punctuation in the last 30% of the window
    - Otherwise cuts at whitespace near the end
    - Merges too-small tail chunks
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    n = len(text)
    if n == 0:
        return []

    max_chars = int(max_chars)
    min_chars = int(min_chars)
    if max_chars < 200:
        max_chars = 200
    if min_chars < 1:
        min_chars = 1
    if min_chars >= max_chars:
        min_chars = max(1, max_chars - 200)

    chunks = []
    start = 0

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            window = text[start:end]

            # Prefer punctuation only near the end of the window (last 30%)
            tail_start = int(len(window) * 0.7)
            tail = window[tail_start:]
            punct = list(re.finditer(r"[.!?]", tail))
            if punct:
                cut = tail_start + punct[-1].end()
                if cut >= min_chars:
                    end = start + cut
            else:
                # Otherwise break at whitespace near end (but not before min_chars)
                ws = window.rfind(" ")
                if ws >= min_chars:
                    end = start + ws

        chunk = text[start:end].strip()

        # Merge tiny remainder into previous chunk
        if len(chunk) < min_chars and chunks:
            chunks[-1] = (chunks[-1] + " " + chunk).strip()
        else:
            chunks.append(chunk)

        # Safety to avoid infinite loop
        if end <= start:
            end = min(start + max_chars, n)
        start = end

    return chunks


def remove_silence_from_audio(waveform, threshold=0.01):
    abs_waveform = torch.abs(waveform)
    non_silent = abs_waveform > threshold
    idx = torch.where(non_silent.any(dim=0))[0]
    if len(idx) > 0:
        return waveform[:, idx[0]:idx[-1] + 1]
    return waveform


def handler(job):
    job_input = job["input"]

    text = job_input.get("text")
    ref_audio_url = job_input.get("ref_audio_url")
    if not text or not ref_audio_url:
        return {"error": "text and ref_audio_url are required"}

    chunk_max_chars = int(job_input.get("chunk_max_chars", 2000))
    chunk_min_chars = int(job_input.get("chunk_min_chars", 800))
    remove_silence = str(job_input.get("remove_silence", "true")).lower() == "true"

    temp_files = []
    total_start = time.time()

    try:
        tts_model = load_model()
        sample_rate = tts_model.sr

        # Reference audio
        ref_audio_ext = os.path.splitext(ref_audio_url)[1] or ".mp3"
        ref_audio_tmp = tempfile.NamedTemporaryFile(suffix=ref_audio_ext, delete=False)
        ref_audio_tmp.close()
        temp_files.append(ref_audio_tmp.name)
        download_audio(ref_audio_url, ref_audio_tmp.name)

        ref_wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_wav_tmp.close()
        temp_files.append(ref_wav_tmp.name)
        convert_to_wav(ref_audio_tmp.name, ref_wav_tmp.name)

        # ---- CHUNK DEBUG (so we see overrides instantly) ----
        print("====== CHUNK DEBUG ======")
        print(f"text_len={len(text)}")
        print(f"chunk_max_chars={chunk_max_chars}")
        print(f"chunk_min_chars={chunk_min_chars}")
        print(f"remove_silence={remove_silence}")
        print("=========================")

        chunks = chunk_text(text, chunk_max_chars, chunk_min_chars)

        print(f"Split text into {len(chunks)} chunks")
        if chunks:
            lens_preview = [len(c) for c in chunks[:10]]
            print(f"chunk_lens(first<=10): {lens_preview}")
            print(f"chunk_len_min/max: {min(len(c) for c in chunks)}/{max(len(c) for c in chunks)}")

        audio_segments = []
        chunk_times = []

        use_cuda = torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16

        for i, chunk in enumerate(chunks):
            t0 = time.time()
            print(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")

            with torch.no_grad():
                if use_cuda:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        wav = tts_model.generate(
                            text=chunk,
                            audio_prompt_path=ref_wav_tmp.name
                        )
                else:
                    wav = tts_model.generate(
                        text=chunk,
                        audio_prompt_path=ref_wav_tmp.name
                    )

            audio_segments.append(wav)

            dt = time.time() - t0
            chunk_times.append(dt)
            print(f"Chunk {i+1} done in {dt:.2f}s")

        final_audio = torch.cat(audio_segments, dim=1)

        if remove_silence:
            final_audio = remove_silence_from_audio(final_audio)

        if final_audio.is_cuda:
            final_audio = final_audio.cpu()

        buffer = io.BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format="wav")
        buffer.seek(0)

        total_time = time.time() - total_start
        avg_chunk = (sum(chunk_times) / len(chunk_times)) if chunk_times else 0.0
        print("====== TTS SUMMARY ======")
        print(f"Chunks: {len(chunks)}")
        print(f"Avg chunk time: {avg_chunk:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print("=========================")

        return {
            "audio": base64.b64encode(buffer.read()).decode(),
            "sample_rate": sample_rate
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return {"error": str(e), "traceback": tb}

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


print("Loading Chatterbox TURBO TTS model...")
load_model()
print("Model loaded successfully!")

runpod.serverless.start({"handler": handler})
