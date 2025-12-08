FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root

# Updated for Gradio 6.0 compatibility

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /workspace

RUN git clone https://github.com/SWivid/F5-TTS.git \
    && cd F5-TTS \
    && git submodule update --init --recursive \
    && pip install -e . --no-cache-dir \
    && pip install runpod --no-cache-dir

ENV SHELL=/bin/bash

# Pre-download models and cache them
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SWivid/F5-TTS', allow_patterns=['F5TTS_Base/*', 'E2TTS_Base/*'])" || echo "Model download will happen at runtime"

VOLUME /root/.cache/huggingface/hub/

EXPOSE 7860

WORKDIR /workspace/F5-TTS

# Copy handler file
COPY handler.py /workspace/F5-TTS/handler.py

# Default CMD for serverless, can be overridden for Gradio
CMD ["python", "-u", "handler.py"]
