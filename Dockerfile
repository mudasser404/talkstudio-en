FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ARG DEBIAN_FRONTEND=noninteractive
LABEL github_repo="https://github.com/SWivid/F5-TTS"

# ---- System deps (including ffmpeg & audio libs) ----
RUN set -x \
 && apt-get update \
 && apt-get -y install \
      wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
      openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
      librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

WORKDIR /workspace

# ---- Clone F5-TTS and install Python deps ----
RUN git clone https://github.com/SWivid/F5-TTS.git \
 && cd F5-TTS \
 && pip install --upgrade pip \
 # Editable install of official F5-TTS (works with torch==2.4.0 from base image)
 && pip install -e . --no-cache-dir \
 # Extra deps we need: SciPy, RunPod SDK, HTTP client, faster-whisper, and boto3 for storage
 && pip install scipy requests runpod faster-whisper boto3 --no-cache-dir

# Hugging Face cache directory
ENV HF_HOME=/root/.cache/huggingface

# (Optional) Pre-download F5TTS_v1_Base weights to speed up first run
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SWivid/F5-TTS', allow_patterns=['F5TTS_v1_Base/*'])" \
  || echo 'Model download will happen at runtime'

VOLUME /root/.cache/huggingface/hub/

# ---- Serverless worker setup ----
WORKDIR /workspace/F5-TTS

# Copy handler into the cloned F5-TTS repo
COPY handler.py /workspace/F5-TTS/handler.py

# Default command for RunPod Serverless
CMD ["python", "-u", "handler.py"]
