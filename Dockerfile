FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install Chatterbox
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox && \
    cd /app/chatterbox && \
    pip install --no-cache-dir -e .

# Copy handler
COPY handler.py .

# Pre-download model weights during build
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"

# Start the handler
CMD ["python", "-u", "handler.py"]
