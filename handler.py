import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import urllib.request
import re

# Enable CUDA debugging (NOTE: this can slow things down; consider disabling once stable)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

model = None

def load_model():
    """Load the Chatterbox TURBO TTS model."""
    global model
    if model is None:
        # ✅ TURBO import
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from huggingface_hub import login

        # Login to Hugging Face if token is available
        if HF_TOKEN:
            print("Logging in to Hugging Face...")
            login(token=HF_TOKEN)
        else:
            print("WARNING: No HF_TOKEN found. Set HF_TOKEN environment variable in RunPod.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # ✅ TURBO load
        model = ChatterboxTurboTTS.from_pretrained(device=device)

        # Helpful sanity check
        print(f"Loaded class: {type(model).__name__}")
        print(f"Chatterbox TURBO model loaded on {device}")

    return model

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

def chunk_text(text, max_chars=400, min_chars=150):
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
    if speed != 1.0:
        effects = [["tempo", str(speed)]]
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    return waveform, sample_rate

def adjust_audio_volume(waveform, volume):
    return waveform * volume

def remove_silence_from_audio(waveform, sample_rate, threshold=0.01):
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

    language = job_input.get("language", "en")
    remove_silence = str(job_input.get("remove_silence", "true")).lower() == "true"
    chunk_max_chars = int(job_input.get("chunk_max_chars", 400))
    chunk_min_chars = int(job_input.get("chunk_min_chars", 150))
    pause_s = float(job_input.get("pause_s", 0.15))
    speed = float(job_input.get("speed", 1))
    volume = float(job_input.get("volume", 1))

    temp_files = []

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tts_model = load_model()
        sample_rate = tts_model.sr

        ref_audio_ext = os.path.splitext(ref_audio_url)[1] or ".mp3"
        ref_audio_temp = tempfile.NamedTemporaryFile(suffix=ref_audio_ext, delete=False)
        ref_audio_temp.close()
        temp_files.append(ref_audio_temp.name)
        download_audio(ref_audio_url, ref_audio_temp.name)

        ref_wav_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_wav_temp.close()
        temp_files.append(ref_wav_temp.name)
        convert_to_wav(ref_audio_temp.name, ref_wav_temp.name)

        chunks = chunk_text(text, chunk_max_chars, chunk_min_chars)
        print(f"Split text into {len(chunks)} chunks")

        audio_segments = []
        pause_samples = int(pause_s * sample_rate)
        pause_tensor = torch.zeros(1, pause_samples)

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

            with torch.no_grad():
                wav = tts_model.generate(
                    text=chunk,
                    audio_prompt_path=ref_wav_temp.name,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )

            if wav.is_cuda:
                wav = wav.cpu()

            if remove_silence:
                wav = remove_silence_from_audio(wav, sample_rate)

            audio_segments.append(wav)

            if i < len(chunks) - 1:
                audio_segments.append(pause_tensor)

        final_audio = torch.cat(audio_segments, dim=1)

        if speed != 1.0:
            final_audio, sample_rate = adjust_audio_speed(final_audio, sample_rate, speed)

        if volume != 1.0:
            final_audio = adjust_audio_volume(final_audio, volume)

        buffer = io.BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated audio: {final_audio.shape[1] / sample_rate:.2f} seconds")

        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

        return {"audio": audio_b64, "sample_rate": sample_rate}

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}\n{traceback.format_exc()}")

        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"error": str(e), "traceback": traceback.format_exc()}

print("Loading Chatterbox TURBO TTS model...")
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

runpod.serverless.start({"handler": handler})
