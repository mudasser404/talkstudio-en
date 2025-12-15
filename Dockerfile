# RunPod Serverless Dockerfile for XTTS-v2 Voice Cloning

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install packages
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install runpod transformers==4.39.3 TTS

# Create TOS file
RUN mkdir -p /root/.local/share/tts && echo "1" > /root/.local/share/tts/coqui_tos_agreed.txt

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
