FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ARG DEBIAN_FRONTEND=noninteractive
LABEL github_repo="https://github.com/coqui-ai/TTS"

# ---- System deps (including ffmpeg & audio libs) ----
RUN set -x \
 && apt-get update \
 && apt-get -y install \
      wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
      openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
      librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
      espeak-ng \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

WORKDIR /workspace

# ---- Install Coqui TTS ----
RUN pip install --upgrade pip \
 && pip install TTS --no-cache-dir \
 # Extra deps we need: SciPy, RunPod SDK, HTTP client, faster-whisper, boto3 for storage, and paramiko for SFTP
 && pip install scipy requests runpod faster-whisper boto3 paramiko --no-cache-dir

# Hugging Face cache directory (Coqui TTS also uses HF for model downloads)
ENV HF_HOME=/root/.cache/huggingface

# Pre-download XTTS v2 model to speed up first run
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')" \
  || echo 'Model download will happen at runtime'

VOLUME /root/.cache/huggingface/hub/

# ---- Serverless worker setup ----
WORKDIR /workspace

# Copy handler
COPY handler.py /workspace/handler.py

# Default command for RunPod Serverless
CMD ["python", "-u", "handler.py"]
