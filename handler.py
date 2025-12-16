import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import urllib.request
import re
import time  # ✅ NEW: timing debug

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


# ✅ Updated defaults here (max_chars/min_chars)
def chunk_text(text, max_chars=2000, min_chars=800):
    """Split text into chunks for processing."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk and len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        if len(current_chunk) < min_chars and chunks:
            chunks[-1] += " " + current_chunk
        else:
            chunks.append(current_chunk.strip())

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
    """
    RunPod serverless handler for Chatterbox TURBO TTS.

    Input:
        - text: Text to synthesize (required)
        - ref_audio_url: URL to reference audio for voice cloning (required)
        - remove_silence: Remove silence from output (default: "true")
        - chunk_max_chars: Max characters per chunk (default: 2000)
        - chunk_min_chars: Min characters per chunk (default: 800)
        - pause_s: Pause between chunks in seconds (default: 0.0)
        - speed: Playback speed multiplier (default: 1)
        - volume: Volume multiplier (default: 1)

    Output:
        - audio: Base64 encoded WAV audio
        - sample_rate: Sample rate of the output audio
    """
    job_input = job["input"]

    text = job_input.get("text")
    if not text:
        return {"error": "Text is required"}

    ref_audio_url = job_input.get("ref_audio_url")
    if not ref_audio_url:
        return {"error": "ref_audio_url is required"}

    remove_silence = str(job_input.get("remove_silence", "true")).lower() == "true"

    # ✅ Updated defaults here
    chunk_max_chars = int(job_input.get("chunk_max_chars", 2000))
    chunk_min_chars = int(job_input.get("chunk_min_chars", 800))
    pause_s = float(job_input.get("pause_s", 0.0))

    speed = float(job_input.get("speed", 1))
    volume = float(job_input.get("volume", 1))

    temp_files = []

    # ✅ NEW: timing (total)
    total_start = time.time()

    try:
        # ❌ Removed torch.cuda.empty_cache() from normal path (keeps allocator warm)

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

        # ✅ NEW: chunk timing summary accumulator
        chunk_times = []

        # Generate per chunk
        audio_segments = []
        pause_samples = int(pause_s * sample_rate)
        pause_tensor = torch.zeros(1, pause_samples)

        use_cuda = torch.cuda.is_available()
        amp_dtype = None
        if use_cuda:
            # Prefer bf16 if supported, otherwise fp16
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()  # ✅ NEW: per-chunk timer
            print(f"Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")

            with torch.no_grad():
                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        wav = tts_model.generate(
                            text=chunk,
                            audio_prompt_path=ref_wav_temp.name,
                            exaggeration=0.5,
                            cfg_weight=0.5
                        )
                else:
                    wav = tts_model.generate(
                        text=chunk,
                        audio_prompt_path=ref_wav_temp.name,
                        exaggeration=0.5,
                        cfg_weight=0.5
                    )

            # ✅ Do NOT .cpu() per chunk; keep on device and concatenate first
            audio_segments.append(wav)

            if i < len(chunks) - 1 and pause_samples > 0:
                # If wav is on GPU, move pause to same device for concat
                audio_segments.append(pause_tensor.to(wav.device))

            chunk_elapsed = time.time() - chunk_start  # ✅ NEW
            chunk_times.append(chunk_elapsed)
            print(f"Chunk {i+1} done in {chunk_elapsed:.2f}s")  # ✅ NEW

        # Concatenate all on the same device
        final_audio = torch.cat(audio_segments, dim=1)

        # Silence trim only once at the end
        if remove_silence:
            final_audio = remove_silence_from_audio(final_audio)

        if speed != 1.0:
            # tempo adjustment requires CPU tensor in torchaudio/sox path
            if final_audio.is_cuda:
                final_audio = final_audio.cpu()
            final_audio, sample_rate = adjust_audio_speed(final_audio, sample_rate, speed)

        if volume != 1.0:
            final_audio = adjust_audio_volume(final_audio, volume)

        # One .cpu() transfer at the end (needed for torchaudio.save to buffer)
        if final_audio.is_cuda:
            final_audio = final_audio.cpu()

        buffer = io.BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated audio: {final_audio.shape[1] / sample_rate:.2f} seconds")

        # ✅ NEW: timing summary
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

        # Optional: clear cache on error only
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"error": str(e), "traceback": tb}

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


# Pre-load model on cold start
print("Loading Chatterbox TURBO TTS model...")
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

runpod.serverless.start({"handler": handler})
