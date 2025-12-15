# RunPod Serverless Dockerfile for Chatterbox TTS

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Clone and install Chatterbox from GitHub
RUN git clone https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox && \
    cd /tmp/chatterbox && \
    pip install --no-cache-dir -e . && \
    cp -r /tmp/chatterbox/chatterbox /app/chatterbox

# Install RunPod
RUN pip install --no-cache-dir runpod requests

# Create models directory
RUN mkdir -p /app/models

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
