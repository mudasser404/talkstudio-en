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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install torchaudio
RUN pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install RunPod and core dependencies first
RUN pip install runpod soundfile pydub requests huggingface_hub

# Install OpenVoice dependencies manually (to avoid conflicts)
RUN pip install librosa==0.9.1 numpy==1.22.0
RUN pip install wavmark pypinyin cn2an jieba inflect unidecode eng_to_ipa langid

# Install OpenVoice from GitHub
RUN pip install git+https://github.com/myshell-ai/OpenVoice.git --no-deps

# Install MeloTTS
RUN pip install git+https://github.com/myshell-ai/MeloTTS.git

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
