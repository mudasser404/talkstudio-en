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

# Install all Python dependencies
RUN pip install --no-cache-dir \
    runpod soundfile pydub requests huggingface_hub numpy librosa scipy \
    faster-whisper whisper-timestamped openai-whisper \
    wavmark pypinyin cn2an jieba inflect unidecode eng_to_ipa langid \
    transformers mecab-python3 num2words gruut g2p-en anyascii jamo gruut-ipa \
    webrtcvad pyworld pyloudnorm praat-parselmouth torchcrepe \
    cached-path tensorboard

# Install MeloTTS from GitHub
RUN pip install --no-cache-dir git+https://github.com/myshell-ai/MeloTTS.git && \
    python -m unidic download

# Clone OpenVoice (Dec 2024 stable)
RUN git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && \
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
