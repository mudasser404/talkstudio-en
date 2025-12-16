# Chatterbox TTS - RunPod Serverless

RunPod serverless deployment for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) voice cloning.

## Setup

### 1. Build Docker Image

```bash
docker build -t chatterbox-runpod .
```

### 2. Push to Docker Hub

```bash
docker tag chatterbox-runpod YOUR_DOCKERHUB_USERNAME/chatterbox-runpod:latest
docker push YOUR_DOCKERHUB_USERNAME/chatterbox-runpod:latest
```

### 3. Get Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with `read` access
3. Copy the token

### 4. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select your Docker image
4. Configure:
   - GPU: RTX 3090 or better recommended
   - Min Workers: 0 (scale to zero)
   - Max Workers: As needed
   - **Environment Variables**: Add `HF_TOKEN` with your Hugging Face token

## API Usage

### Request Format

```json
{
  "input": {
    "text": "Hello, this is a test of voice cloning.",
    "ref_audio_url": "https://example.com/voice_sample.mp3",
    "language": "en",
    "remove_silence": "true",
    "chunk_max_chars": 400,
    "chunk_min_chars": 150,
    "pause_s": 0.15,
    "speed": 1,
    "volume": 1
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| text | string | Yes | - | Text to synthesize |
| ref_audio_url | string | Yes | - | URL to reference audio for voice cloning |
| language | string | No | "en" | Language code |
| remove_silence | string/bool | No | "true" | Remove silence from output |
| chunk_max_chars | int | No | 400 | Max characters per chunk |
| chunk_min_chars | int | No | 150 | Min characters per chunk |
| pause_s | float | No | 0.15 | Pause between chunks (seconds) |
| speed | float | No | 1 | Playback speed multiplier |
| volume | float | No | 1 | Volume multiplier |

### Response Format

```json
{
  "audio": "<base64_encoded_wav>",
  "sample_rate": 24000
}
```

### Python Client Example

```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Make request with reference audio URL
result = endpoint.run_sync({
    "text": "Hello world! This is a test of voice cloning.",
    "ref_audio_url": "https://example.com/voice_sample.mp3",
    "remove_silence": "true",
    "speed": 1.0,
    "volume": 1.0
})

# Save output
audio_bytes = base64.b64decode(result["audio"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

## Important Notes

1. **Hugging Face Token Required**: Set `HF_TOKEN` environment variable in RunPod endpoint settings
2. **First Request Slow**: Model downloads on first cold start (~20-30 seconds)
3. **Chunking**: Long text is automatically split into chunks for better quality
4. **Audio Format**: Reference audio can be MP3, WAV, etc. (auto-converted to 24kHz mono WAV)
