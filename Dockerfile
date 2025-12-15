# RunPod Serverless Dockerfile for OpenVoice V2 Voice Cloning

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install torchaudio
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install all Python dependencies in one step
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    pydub \
    requests \
    huggingface_hub \
    numpy \
    librosa \
    faster-whisper \
    wavmark \
    pypinyin \
    cn2an \
    jieba \
    inflect \
    unidecode \
    eng_to_ipa \
    langid

# Install MeloTTS from GitHub
RUN pip install --no-cache-dir git+https://github.com/myshell-ai/MeloTTS.git && \
    python -m unidic download

# Clone OpenVoice and install
RUN git clone https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && \
    cd /tmp/OpenVoice && \
    pip install -e . --no-deps && \
    cp -r /tmp/OpenVoice/openvoice /app/openvoice && \
    rm -rf /tmp/OpenVoice

# Download OpenVoice V2 checkpoints
RUN mkdir -p /app/checkpoints_v2 && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='/app/checkpoints_v2')"

# Pre-download MeloTTS model
RUN python -c "from melo.api import TTS; TTS(language='EN')"

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
