# RunPod Serverless Dockerfile for XTTS-v2 Voice Cloning

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install torchaudio
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install RunPod
RUN pip install runpod

# Install Coqui TTS with correct transformers version
RUN pip install transformers==4.39.3
RUN pip install TTS

# Accept Coqui TTS TOS BEFORE downloading model
RUN mkdir -p /root/.local/share/tts && \
    echo "agreed" > /root/.local/share/tts/coqui_tos_agreed.txt

# Pre-download XTTS-v2 model with TOS bypass
RUN python -c "import os; os.makedirs('/root/.local/share/tts', exist_ok=True); open('/root/.local/share/tts/coqui_tos_agreed.txt', 'w').write('agreed'); from TTS.utils.manage import ModelManager; m = ModelManager(); setattr(m, 'ask_tos', lambda *a: True); from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
