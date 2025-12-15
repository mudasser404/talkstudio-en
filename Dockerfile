# RunPod Serverless Dockerfile for OpenVoice V2 Voice Cloning

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install torchaudio
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install RunPod and audio processing
RUN pip install runpod soundfile librosa pydub requests numpy huggingface_hub

# Clone and install OpenVoice V2
RUN git clone https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && \
    cd /tmp/OpenVoice && \
    pip install -e . && \
    rm -rf /tmp/OpenVoice/.git

# Clone and install MeloTTS (required for base TTS)
RUN git clone https://github.com/myshell-ai/MeloTTS.git /tmp/MeloTTS && \
    cd /tmp/MeloTTS && \
    pip install -e . && \
    rm -rf /tmp/MeloTTS/.git

# Download unidic for MeloTTS
RUN python -m unidic download

# Download OpenVoice V2 checkpoints
RUN mkdir -p /app/checkpoints_v2
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='/app/checkpoints_v2')"

# Pre-download MeloTTS model
RUN python -c "from melo.api import TTS; TTS(language='EN')"

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
