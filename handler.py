import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import urllib.request
import re

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Global model variable
model = None

def load_model():
    """Load the Chatterbox TTS model."""
    global model
    if model is None:
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        model = ChatterboxTTS.from_pretrained(device=device)
        print(f"Chatterbox model loaded on {device}")
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
    # Resample to 24kHz if needed (Chatterbox requirement)
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, 24000)
    print(f"Converted to WAV: {output_path}")

def chunk_text(text, max_chars=400, min_chars=150):
    """Split text into chunks for processing."""
    # Split by sentences
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

    # Add remaining text
    if current_chunk:
        if len(current_chunk) < min_chars and chunks:
            # Merge with last chunk if too short
            chunks[-1] += " " + current_chunk
        else:
            chunks.append(current_chunk.strip())

    return chunks

def adjust_audio_speed(waveform, sample_rate, speed):
    """Adjust audio speed."""
    if speed != 1.0:
        effects = [["tempo", str(speed)]]
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
    return waveform, sample_rate

def adjust_audio_volume(waveform, volume):
    """Adjust audio volume."""
    return waveform * volume

def remove_silence_from_audio(waveform, sample_rate, threshold=0.01):
    """Remove silence from beginning and end of audio."""
    # Find non-silent parts
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
    RunPod serverless handler for Chatterbox TTS.

    Input:
        - text: Text to synthesize (required)
        - ref_audio_url: URL to reference audio for voice cloning (required)
        - language: Language code (default: "en")
        - remove_silence: Remove silence from output (default: "true")
        - chunk_max_chars: Max characters per chunk (default: 400)
        - chunk_min_chars: Min characters per chunk (default: 150)
        - pause_s: Pause between chunks in seconds (default: 0.15)
        - speed: Playback speed multiplier (default: 1)
        - volume: Volume multiplier (default: 1)

    Output:
        - audio: Base64 encoded WAV audio
        - sample_rate: Sample rate of the output audio
    """
    job_input = job["input"]

    # Get input parameters
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
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load model
        tts_model = load_model()
        sample_rate = tts_model.sr  # 24000 for Chatterbox

        # Download and convert reference audio
        ref_audio_ext = os.path.splitext(ref_audio_url)[1] or ".mp3"
        ref_audio_temp = tempfile.NamedTemporaryFile(suffix=ref_audio_ext, delete=False)
        ref_audio_temp.close()
        temp_files.append(ref_audio_temp.name)
        download_audio(ref_audio_url, ref_audio_temp.name)

        # Convert to WAV
        ref_wav_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_wav_temp.close()
        temp_files.append(ref_wav_temp.name)
        convert_to_wav(ref_audio_temp.name, ref_wav_temp.name)

        # Chunk text for processing
        chunks = chunk_text(text, chunk_max_chars, chunk_min_chars)
        print(f"Split text into {len(chunks)} chunks")

        # Generate audio for each chunk
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

            # Ensure on CPU
            if wav.is_cuda:
                wav = wav.cpu()

            # Remove silence if requested
            if remove_silence:
                wav = remove_silence_from_audio(wav, sample_rate)

            audio_segments.append(wav)

            # Add pause between chunks (except last)
            if i < len(chunks) - 1:
                audio_segments.append(pause_tensor)

        # Concatenate all segments
        final_audio = torch.cat(audio_segments, dim=1)

        # Adjust speed
        if speed != 1.0:
            final_audio, sample_rate = adjust_audio_speed(final_audio, sample_rate, speed)

        # Adjust volume
        if volume != 1.0:
            final_audio = adjust_audio_volume(final_audio, volume)

        # Convert to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format="wav")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        print(f"Generated audio: {final_audio.shape[1] / sample_rate:.2f} seconds")

        # Cleanup temp files
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

        return {
            "audio": audio_b64,
            "sample_rate": sample_rate
        }

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error: {error_msg}")

        # Cleanup temp files
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

        # Clear CUDA cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"error": str(e), "traceback": traceback.format_exc()}

# Pre-load model on cold start
print("Loading Chatterbox TTS model...")
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

runpod.serverless.start({"handler": handler})
