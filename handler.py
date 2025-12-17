import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import urllib.request
import re
import time  # timing debug

from huggingface_hub import login

# Optional: set to "1" only for debugging; it slows down GPU
if os.getenv("CUDA_LAUNCH_BLOCKING", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

# Global model variable
model = None
hf_logged_in = False


def ensure_hf_login():
    """Login to Hugging Face once if token is provided."""
    global hf_logged_in
    if hf_logged_in:
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print("Hugging Face login: OK (token provided)")
    else:
        print(
            "WARNING: No HF token found in env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN). "
            "Turbo weights may fail to download if the repo requires authentication."
        )
    hf_logged_in = True


def load_model():
    """Load the Chatterbox TURBO TTS model."""
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
    """Download audio from URL."""
    print(f"Downloading audio from: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to: {output_path}")


def convert_to_wav(input_path, output_path):
    """Convert audio to WAV format using torchaudio."""
    waveform, sample_rate = torchaudio.load(input_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 24kHz (Chatterbox requirement)
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 24000)
    print(f"Converted to WAV: {output_path}")


# ✅ UPDATED: chunker that fills up to max_chars (instead of tiny sentence chunks)
def chunk_text(text, max_chars=2000, min_chars=800):
    """
    Split text into chunks that are close to max_chars, preferring sentence boundaries.
    This drastically reduces the number of chunks for long text (big speed win).
    """
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    chunks = []
    start = 0

    while start < n:
        end = min(start + max_chars, n)

        # If we didn't hit the end, try to cut at a sentence boundary near the end
        if end < n:
            window = text[start:end]
            matches = list(re.finditer(r"[.!?]", window))
            if matches:
                # choose the last sentence-ending punctuation in the window
                last_end = matches[-1].end()
                # only accept it if chunk is big enough
                if last_end >= min_chars:
                    end = start + last_end

        chunk = text[start:end].strip()

        # If chunk is too small, merge into previous (avoids tiny tail chunk)
        if len(chunk) < min_chars and chunks:
            chunks[-1] = (chunks[-1] + " " + chunk).strip()
        else:
            chunks.append(chunk)

        start = end

    return chunks


def adjust_audio_speed(waveform, sample_rate, speed):
    """Adjust audio speed."""
    if speed != 1.0:
        effects = [["tempo", str(speed)]]
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects
        )
    return waveform, sample_rate


def adjust_audio_volume(waveform, volume):
    """Adjust audio volume."""
    return waveform * volume


def remove_silence_from_audio(waveform, threshold=0.01):
    """Remove silence from beginning and end of audio."""
    abs_waveform = torch.abs(waveform)
    non_silent = abs_waveform > threshold
    non_silent_indices = torch.where(non_silent.any(dim=0))[0]

    if len(non_silent_indices) > 0:
        start = non_silent_indices[0].item()
        end = non_silent_indices[-1].item() + 1
        return waveform[:, start:end]
    return waveform


def handler(job):
    job_input = job["input"]

    text = job_input.get("text")
    if not text:
        return {"error": "Text is required"}

    ref_audio_url = job_input.get("ref_audio_url")
    if not ref_audio_url:
        return {"error": "ref_audio_url is required"}

    remove_silence = str(job_input.get("remove_silence", "true")).lower() == "true"

    chunk_max_chars = int(job_input.get("chunk_max_chars", 2000))
    chunk_min_chars = int(job_input.get("chunk_min_chars", 800))
    pause_s = float(job_input.get("pause_s", 0.0))

    speed = float(job_input.get("speed", 1))
    volume = float(job_input.get("volume", 1))

    temp_files = []

    total_start = time.time()

    try:
        tts_model = load_model()
        sample_rate = tts_model.sr  # 24000

        # Download + convert reference audio
        ref_audio_ext = os.path.splitext(ref_audio_url)[1] or ".mp3"
        ref_audio_temp = tempfile.NamedTemporaryFile(suffix=ref_audio_ext, delete=False)
        ref_audio_temp.close()
        temp_files.append(ref_audio_temp.name)
        download_audio(ref_audio_url, ref_audio_temp.name)

        ref_wav_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_wav_temp.close()
        temp_files.append(ref_wav_temp.name)
        convert_to_wav(ref_audio_temp.name, ref_wav_temp.name)

        # Chunk text
        chunks = chunk_text(text, chunk_max_chars, chunk_min_chars)
        print(f"Split text into {len(chunks)} chunks")

        chunk_times = []

        # Generate per chunk
        audio_segments = []
        pause_samples = int(pause_s * sample_rate)
        pause_tensor = torch.zeros(1, pause_samples)

        use_cuda = torch.cuda.is_available()
        amp_dtype = None
        if use_cuda:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            print(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")

            with torch.no_grad():
                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        # ✅ UPDATED: remove unsupported args for Turbo (less overhead / no warnings)
                        wav = tts_model.generate(
                            text=chunk,
                            audio_prompt_path=ref_wav_temp.name
                        )
                else:
                    wav = tts_model.generate(
                        text=chunk,
                        audio_prompt_path=ref_wav_temp.name
                    )

            audio_segments.append(wav)

            if i < len(chunks) - 1 and pause_samples > 0:
                audio_segments.append(pause_tensor.to(wav.device))

            chunk_elapsed = time.time() - chunk_start
            chunk_times.append(chunk_elapsed)
            print(f"Chunk {i+1} done in {chunk_elapsed:.2f}s")

        final_audio = torch.cat(audio_segments, dim=1)

        if remove_silence:
            final_audio = remove_silence_from_audio(final_audio)

        if speed != 1.0:
            if final_audio.is_cuda:
                final_audio = final_audio.cpu()
            final_audio, sample_rate = adjust_audio_speed(final_audio, sample_rate, speed)

        if volume != 1.0:
            final_audio = adjust_audio_volume(final_audio, volume)

        if final_audio.is_cuda:
            final_audio = final_audio.cpu()

        buffer = io.BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated audio: {final_audio.shape[1] / sample_rate:.2f} seconds")

        total_elapsed = time.time() - total_start
        avg_chunk = (sum(chunk_times) / len(chunk_times)) if chunk_times else 0.0
        print("====== TTS TIMING SUMMARY ======")
        print(f"Chunks: {len(chunks)}")
        print(f"Avg time per chunk: {avg_chunk:.2f}s")
        print(f"Total time: {total_elapsed:.2f}s")
        print("================================")

        return {"audio": audio_b64, "sample_rate": sample_rate}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error: {str(e)}\n{tb}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"error": str(e), "traceback": tb}

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


print("Loading Chatterbox TURBO TTS model...")
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

runpod.serverless.start({"handler": handler})
