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
import paramiko

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
# Streaming VPS Upload (Memory Efficient)
# ==============================
def _upload_to_vps_streaming(file_path: str, storage_config: Dict[str, Any]) -> str:
    """
    Upload COMPLETE audio file to VPS using STREAMING to avoid memory issues.
    File is uploaded in chunks without loading entire file in memory.
    """
    host = storage_config.get("host", "72.61.125.201")
    port = int(storage_config.get("port", 22))
    username = storage_config.get("username", "root")
    password = storage_config.get("password", "Ryk112233@@@")
    key_file = storage_config.get("key_file")
    remote_path = storage_config.get("remote_path", "/media/runpod_audio")
    base_url = storage_config.get("base_url", "https://demo.talkstudio.ai")

    if not all([host, username, remote_path, base_url]):
        raise ValueError("Storage config must include host, username, remote_path, and base_url")

    # Generate unique filename
    import uuid
    from datetime import datetime
    
    timestamp = datetime.utcnow().strftime("%Y/%m/%d")
    unique_id = str(uuid.uuid4())
    filename = f"runpod_{unique_id}.wav"
    
    # Create remote directory path
    remote_dir = os.path.join(remote_path, timestamp)
    remote_file_path = os.path.join(remote_dir, filename)
    
    # Public URL
    public_url = f"{base_url.rstrip('/')}/{timestamp}/{filename}"

    ssh = None
    sftp = None
    
    try:
        print(f"[UPLOAD] Connecting to {host}:{port} as {username}...")
        print(f"[UPLOAD] Using password authentication: {bool(password)}")
        print(f"[UPLOAD] Using key file: {key_file or 'None'}")

        # Setup SSH connection with compression
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect
        connect_kwargs = {
            "hostname": host,
            "port": port,
            "username": username,
            "compress": True,
            "look_for_keys": False,  # Don't use SSH keys from ~/.ssh
            "allow_agent": False,    # Don't use SSH agent
        }

        if key_file:
            connect_kwargs["key_filename"] = key_file
        elif password:
            connect_kwargs["password"] = password
        else:
            raise ValueError("Either password or key_file must be provided")

        ssh.connect(**connect_kwargs)
        
        print(f"[UPLOAD] Connected successfully")
        
        # Create directory
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_dir}")
        stdout.channel.recv_exit_status()
        
        # Open SFTP
        sftp = ssh.open_sftp()
        
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"[UPLOAD] Uploading COMPLETE audio file: {file_size_mb:.2f}MB")
        print(f"[UPLOAD] This is the FINAL combined audio, not chunks")
        
        # Stream upload
        CHUNK_SIZE = 32768  # 32KB chunks for network transfer
        uploaded_bytes = 0
        
        with sftp.open(remote_file_path, 'wb') as remote_file:
            with open(file_path, 'rb') as local_file:
                while True:
                    chunk = local_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    remote_file.write(chunk)
                    uploaded_bytes += len(chunk)
                    
                    # Progress every 5MB
                    if uploaded_bytes % (5 * 1024 * 1024) < CHUNK_SIZE:
                        progress = (uploaded_bytes / file_size) * 100
                        print(f"[UPLOAD] Progress: {progress:.1f}% ({uploaded_bytes / (1024*1024):.2f}MB / {file_size_mb:.2f}MB)")
        
        print(f"[UPLOAD] ✓ Complete audio uploaded: {file_size_mb:.2f}MB")
        
        # Set permissions
        sftp.chmod(remote_file_path, 0o644)
        
        # Close connections
        sftp.close()
        ssh.close()
        
        print(f"[UPLOAD] ✓ File accessible at: {public_url}")
        
        return public_url
        
    except Exception as e:
        print(f"[UPLOAD] ✗ Upload failed: {e}")
        if sftp:
            sftp.close()
        if ssh:
            ssh.close()
        raise Exception(f"Failed to upload to VPS: {str(e)}")


# ==============================
# HTTP Streaming Upload
# ==============================
def _upload_to_vps_http_streaming(file_path: str, storage_config: Dict[str, Any]) -> str:
    """
    Upload COMPLETE file via HTTP POST with streaming
    """
    upload_endpoint = storage_config.get("upload_endpoint")
    api_key = storage_config.get("api_key")

    if not upload_endpoint:
        raise ValueError("Storage config must include upload_endpoint")

    try:
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"[HTTP_UPLOAD] Uploading COMPLETE audio: {file_size_mb:.2f}MB")
        
        headers = {"Content-Type": "audio/wav"}
        if api_key:
            headers["X-API-Key"] = api_key
        
        # Stream upload
        def file_generator():
            CHUNK_SIZE = 65536  # 64KB
            uploaded = 0
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    uploaded += len(chunk)
                    
                    if uploaded % (5 * 1024 * 1024) < CHUNK_SIZE:
                        progress = (uploaded / file_size) * 100
                        print(f"[HTTP_UPLOAD] Progress: {progress:.1f}%")
                    
                    yield chunk
        
        response = requests.post(
            upload_endpoint,
            data=file_generator(),
            headers=headers,
            timeout=300
        )
        
        response.raise_for_status()

        # Handle different response formats
        try:
            result = response.json()
            audio_url = result.get("url") or result.get("file_url") or result.get("audio_url")
        except:
            # If not JSON, assume the response text is the URL
            audio_url = response.text.strip()

        if not audio_url:
            raise Exception("Upload endpoint did not return URL")
        
        print(f"[HTTP_UPLOAD] ✓ Complete audio uploaded: {audio_url}")
        return audio_url
        
    except Exception as e:
        print(f"[HTTP_UPLOAD] ✗ Upload failed: {e}")
        raise Exception(f"Failed to upload to VPS: {str(e)}")


# ==============================
# Main Upload Router
# ==============================
def _upload_to_storage(file_path: str, storage_config: Dict[str, Any]) -> str:
    """
    Route to appropriate streaming upload method
    """
    method = storage_config.get("method", "sftp").lower()
    
    if method == "sftp":
        return _upload_to_vps_streaming(file_path, storage_config)
    elif method == "http":
        return _upload_to_vps_http_streaming(file_path, storage_config)
    else:
        raise ValueError(f"Unsupported upload method: {method}")


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
    """
    Split text into chunks for TTS processing.
    NOTE: These chunks are only for TTS processing - final output is ONE combined audio file.
    """
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


def _concat_audio_streaming(segments: List[np.ndarray], output_path: str, sr: int) -> int:
    """
    Combine ALL audio segments into ONE complete file.
    This creates the FINAL audio that will be uploaded.
    Memory-efficient: writes directly to disk.
    """
    import struct
    
    print(f"[COMBINE] Merging {len(segments)} audio segments into ONE complete file...")
    
    # Calculate total samples
    total_samples = sum(len(seg) for seg in segments)
    total_duration = total_samples / sr
    
    print(f"[COMBINE] Total duration: {total_duration:.2f} seconds")
    
    # Write WAV file
    with open(output_path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 0))  # Placeholder
        f.write(b'WAVE')
        
        # Format chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', 1))   # Mono
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        
        # Data chunk
        f.write(b'data')
        data_size = total_samples * 2
        f.write(struct.pack('<I', data_size))
        
        # Write all segments
        for i, segment in enumerate(segments):
            segment.tofile(f)
            del segment
            
            if (i + 1) % 10 == 0:
                print(f"[COMBINE] Progress: {i + 1}/{len(segments)} segments merged")
        
        # Update file size
        file_size = f.tell()
        f.seek(4)
        f.write(struct.pack('<I', file_size - 8))
    
    print(f"[COMBINE] ✓ Complete audio file created: {file_size / (1024*1024):.2f}MB")
    
    return file_size


# ==============================
# Main Handler
# ==============================
def generate_speech(job: Dict[str, Any]) -> Dict[str, Any]:

    print("### handler version: f5tts_complete_audio_2025-12-09 ###")
    print("### NOTE: All chunks will be COMBINED into ONE complete audio file ###")

    inp = job.get("input", {})

    raw_text = (inp.get("text") or "").strip()
    print("[INPUT] text length:", len(raw_text))

    ref_path = _get_ref_audio_path(inp)

    # Reference text
    ref_text = (inp.get("ref_text") or "").strip()
    if not ref_text:
        ref_text = _transcribe_ref_audio(ref_path, language=inp.get("language"))

    # Chunking (for TTS processing only - output will be ONE file)
    max_chars = int(inp.get("chunk_max_chars", 150))
    min_chars = int(inp.get("chunk_min_chars", 50))
    chunks = _chunk_text(raw_text, max_chars=max_chars, min_chars=min_chars)

    print(f"[PROCESSING] Will process {len(chunks)} text chunks")
    print(f"[PROCESSING] These will be COMBINED into ONE complete audio file")

    # Synthesis settings
    # Higher default speed to reduce file size for long texts
    speed = float(inp.get("speed", 1.0))  # Changed from 0.7 to 1.0 for faster/smaller files
    remove_silence = bool(inp.get("remove_silence", True))  # Enable by default to reduce size
    quality = (inp.get("quality") or "standard").lower()
    nfe_step = 64 if quality == "premium" else 32
    volume = float(inp.get("volume", 1.0))
    volume = max(volume, 0.0)
    target_rms = 0.1 * volume
    pause_seconds = float(inp.get("pause_s", 0.12))

    print(f"[SYNTH] speed={speed}, quality={quality}, nfe_step={nfe_step}, volume={volume}, pause_s={pause_seconds}")

    api = get_f5tts_model()

    all_segments = []
    sr_final = 24000

    # Generate each chunk
    for idx, chunk in enumerate(chunks, start=1):
        cleaned = _clean_for_tts(chunk)
        print(f"[TTS] Processing chunk {idx}/{len(chunks)} ({len(cleaned)} chars)")

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

        # Insert pause between chunks
        if idx < len(chunks) and pause_seconds > 0:
            pause_samples = int(sr_final * pause_seconds)
            silence = np.zeros(pause_samples, dtype=np.int16)
            all_segments.append(silence)

    # Create ONE complete audio file
    fd, final_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    print("[FINAL] Creating complete combined audio file...")
    file_size = _concat_audio_streaming(all_segments, final_path, sr_final)
    
    # Clear memory
    del all_segments
    
    file_size_mb = file_size / (1024 * 1024)
    print(f"[FINAL] Complete audio ready: {file_size_mb:.2f}MB")

    # Storage config - create default if not provided
    storage_config = inp.get("storage")
    auto_upload_threshold_mb = float(inp.get("auto_upload_threshold_mb", 8.0))

    # If no storage config but file will likely be large, create default config
    if not storage_config and file_size_mb > auto_upload_threshold_mb:
        print(f"[AUTO-CONFIG] File is {file_size_mb:.2f}MB, creating default VPS storage config")
        storage_config = {
            "method": "sftp",
            "host": "72.61.125.201",
            "port": 22,
            "username": "root",
            "password": "Ryk112233@@@",
            "remote_path": "/media/runpod_audio",
            "base_url": "https://demo.talkstudio.ai"
        }

    result = {
        "sample_rate": sr_final,
        "ref_text_used": ref_text,
        "num_chunks_processed": len(chunks),
        "quality": quality,
        "volume": volume,
        "pause_s": pause_seconds,
        "file_size_mb": round(file_size_mb, 2),
        "output_type": "complete_audio"  # Clarify this is ONE complete file
    }

    try:
        # Upload if storage config provided OR file too large
        should_upload = storage_config is not None

        if should_upload and file_size_mb > auto_upload_threshold_mb:
            print(f"[OUTPUT] Uploading COMPLETE audio file to VPS...")
            
            audio_url = _upload_to_storage(final_path, storage_config)
            result["audio_url"] = audio_url
            result["uploaded"] = True
            
            if file_size_mb > auto_upload_threshold_mb:
                result["upload_reason"] = f"file_size_exceeds_{auto_upload_threshold_mb}mb"
            else:
                result["upload_reason"] = "storage_config_provided"
            
            print(f"[OUTPUT] ✓ Complete audio available at: {audio_url}")
            
        else:
            # Base64 for small files
            max_base64_mb = 8.0
            
            if file_size_mb > max_base64_mb:
                error_msg = (
                    f"File too large ({file_size_mb:.2f}MB) for base64 response. "
                    f"Maximum: {max_base64_mb}MB. "
                    f"Please provide 'storage' configuration to upload to VPS."
                )
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

            print(f"[OUTPUT] Encoding complete audio as base64...")
            with open(final_path, "rb") as f:
                audio_bytes = f.read()
            
            base64_str = base64.b64encode(audio_bytes).decode("utf-8")
            result["audio_base64"] = base64_str
            result["uploaded"] = False
            
            print(f"[OUTPUT] ✓ Complete audio encoded as base64")

    finally:
        # Cleanup
        if os.path.exists(final_path):
            os.remove(final_path)
            print("[CLEANUP] Temp files removed")

    print("[SUCCESS] ONE complete audio file generated and delivered!")
    
    return result


# ==============================
# RunPod Entry
# ==============================
runpod.serverless.start({"handler": generate_speech})
   



