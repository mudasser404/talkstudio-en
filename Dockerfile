# CUDA 12.4 runtime, good for modern GPUs (A40/4090 etc.)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Basic OS deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg libsndfile1-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install PyTorch + torchaudio (CUDA 12.4 build)
RUN pip install "torch==2.4.0+cu124" "torchaudio==2.4.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# Install F5-TTS (includes F5TTS_v1_Base support) + RunPod SDK + requests
RUN pip install f5-tts runpod requests --no-cache-dir

# Hugging Face cache location (optional)
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /workspace/app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
