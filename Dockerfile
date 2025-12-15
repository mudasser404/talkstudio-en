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

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install torchaudio
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
RUN pip install runpod soundfile pydub requests huggingface_hub numpy

# Install audio processing
RUN pip install librosa

# Install MeloTTS from GitHub
RUN pip install git+https://github.com/myshell-ai/MeloTTS.git
RUN python -m unidic download

# Install OpenVoice dependencies
RUN pip install wavmark pypinyin cn2an jieba inflect unidecode eng_to_ipa langid

# Install faster-whisper for VAD in se_extractor
RUN pip install faster-whisper

# Clone OpenVoice and install
RUN git clone https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && \
    cd /tmp/OpenVoice && \
    pip install -e . --no-deps && \
    cp -r /tmp/OpenVoice/openvoice /app/openvoice

# Download OpenVoice V2 checkpoints
RUN mkdir -p /app/checkpoints_v2
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='/app/checkpoints_v2')"

# Pre-download MeloTTS model
RUN python -c "from melo.api import TTS; TTS(language='EN')"

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
