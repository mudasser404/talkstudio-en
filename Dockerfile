# RunPod Serverless Dockerfile for Speaker Embedding Voice Cloning
# Supports: OpenVoice, XTTS-v2, Kokoro-82M

FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
RUN pip install -r requirements.txt

# Install TTS models
# 1. Coqui TTS (XTTS-v2)
RUN pip install TTS

# 2. OpenVoice
RUN pip install git+https://github.com/myshell-ai/OpenVoice.git
RUN pip install git+https://github.com/myshell-ai/MeloTTS.git

# 3. Kokoro-82M
RUN pip install kokoro>=0.3.4 soundfile

# Download model checkpoints
RUN mkdir -p /app/checkpoints_v2

# Download OpenVoice V2 checkpoints
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='/app/checkpoints_v2')"

# Pre-download XTTS-v2 model
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Pre-download Kokoro model
RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a')"

# Copy handler
COPY handler.py .
COPY test_input.json .

# RunPod specific
ENV RUNPOD_DEBUG_LEVEL=DEBUG

# Start handler
CMD ["python", "-u", "handler.py"]
