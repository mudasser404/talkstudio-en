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

# =========================
# ENV / GLOBALS
# =========================

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


# =========================
# HUGGING FACE LOGIN
# =========================

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


# =========================
# MODEL LOADING
# =========================

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


# =========================
# AUDIO HELPERS
# =========================

def download_audio(url, output_path):
    print(f"Downloading audio from: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to: {output_path}")


def convert_to_wav(input_path, output_path):
    waveform, sample_rate = torchaudio.load(input_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 24000)
    print(f"Converted to WAV: {output_path}")


# =========================
# FIXED CHUNKER (IMPORTANT)
# =========================

def chunk_text(text, max_chars=2000, min_chars=800):
    """
    Robust chunker:
    - Produces near-max chunks
    - Prefers sentence boundaries
    - Falls back to strict slicing if fragmentation is detected
    """
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n == 0:
        return []

    def primary():
        chunks = []
        start = 0
        while start < n:
            end = min(start + max_chars, n)

            if end < n:
                window = text[start:end]
                matches = list(re.finditer(r"[.!?]", window))
                if matches:
                    last_end = matches[-1].end()
                    if last_end >= min_chars:
                        end = start + last_end

            chunk = text[start:end].strip()
            if len(chunk) < min_chars and chunks:
                chunks[-1] = (chunks[-1] + " " + chunk).strip()
            else:
                chunks.append(chunk)

            start = end
        return chunks

    def fallback():
        chunks = []
        start = 0
        while start < n:
            end = min(start + max_chars, n)
            if end < n:
                cut = text.rfind(" ", start + min_chars, end)
                if cut != -1:
                    end = cut

            chunk = text[start:end].strip()
            if len(chunk) < min_chars and chunks:
                chunks[-1] = (chunks[-1] + " " + chunk).strip()
            else:
                chunks.append(chunk)

            start = end
        return chunks

    chunks = primary()
    expected_max = max(1, (n // max_chars) + 6)

    if len(chunks) > expected_max:
        print(f"[chunk_text] Fragmentation detected ({len(chunks)} chunks). Using fallback.")
        chunks = fallback()

    return chunks


# =========================
# POST PROCESS
# =========================

def remove_silence_from_audio(waveform, threshold=0.01):
    abs_waveform = torch.abs(waveform)
    non_silent = abs_waveform > threshold
    idx = torch.where(non_silent.any(dim=0))[0]
    if len(idx) > 0:
        return waveform[:, idx[0]:idx[-1] + 1]
    return waveform


# =========================
# HANDLER
# =========================

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
        ref_audio_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        ref_audio_tmp.close()
        temp_files.append(ref_audio_tmp.name)
        download_audio(ref_audio_url, ref_audio_tmp.name)

        ref_wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_wav_tmp.close()
        temp_files.append(ref_wav_tmp.name)
        convert_to_wav(ref_audio_tmp.name, ref_wav_tmp.name)

        # Chunking
        chunks = chunk_text(text, chunk_max_chars, chunk_min_chars)
        print(f"Split text into {len(chunks)} chunks")

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
                        wav = tts_model.generate(text=chunk, audio_prompt_path=ref_wav_tmp.name)
                else:
                    wav = tts_model.generate(text=chunk, audio_prompt_path=ref_wav_tmp.name)

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
        print("====== TTS SUMMARY ======")
        print(f"Chunks: {len(chunks)}")
        print(f"Avg chunk time: {sum(chunk_times)/len(chunk_times):.2f}s")
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


# =========================
# START
# =========================

print("Loading Chatterbox TURBO TTS model...")
load_model()
print("Model loaded successfully!")

runpod.serverless.start({"handler": handler})
