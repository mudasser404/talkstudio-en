# RunPod Serverless Dockerfile for Chatterbox TTS

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Chatterbox TTS
RUN pip install --no-cache-dir chatterbox-tts

# Install RunPod and other dependencies
RUN pip install --no-cache-dir runpod requests torchaudio

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
