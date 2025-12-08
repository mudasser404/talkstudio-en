FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ARG DEBIAN_FRONTEND=noninteractive
LABEL github_repo="https://github.com/SWivid/F5-TTS"

# ---- System deps (same as your original, just formatted) ----
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
 && git submodule update --init --recursive \
 # install F5-TTS (editable)
 && pip install -e . --no-cache-dir \
 # IMPORTANT: make TorchCodec match torch==2.4.*  (fixes libtorchcodec error)
 && pip install "torchcodec==0.0.3" --no-cache-dir \
 # SciPy is required by your handler + patch (scipy.io.wavfile)
 && pip install scipy --no-cache-dir \
 # RunPod client
 && pip install runpod --no-cache-dir

ENV SHELL=/bin/bash

# ---- Optional: pre-download models into HF cache ----
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SWivid/F5-TTS', allow_patterns=['F5TTS_Base/*', 'E2TTS_Base/*', 'F5TTS_v1_Base/*'])" \
  || echo 'Model download will happen at runtime'

VOLUME /root/.cache/huggingface/hub/
EXPOSE 7860

# ---- Serverless worker setup ----
WORKDIR /workspace/F5-TTS

# Copy your existing handler + patch into the cloned F5-TTS repo
COPY handler.py /workspace/F5-TTS/handler.py
COPY patch_weights_only.py /workspace/F5-TTS/patch_weights_only.py

# Default command for RunPod Serverless
CMD ["python", "-u", "handler.py"]
