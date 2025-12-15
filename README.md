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

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select your Docker image
4. Configure:
   - GPU: RTX 3090 or better recommended
   - Min Workers: 0 (scale to zero)
   - Max Workers: As needed

## API Usage

### Request Format

```json
{
  "input": {
    "text": "Hello, this is a test of voice cloning.",
    "audio_prompt": "<base64_encoded_wav_audio>",
    "exaggeration": 0.5,
    "cfg_weight": 0.5
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| text | string | Yes | - | Text to synthesize |
| audio_prompt | string | No | - | Base64 encoded WAV for voice cloning |
| exaggeration | float | No | 0.5 | Exaggeration factor (0.0-1.0) |
| cfg_weight | float | No | 0.5 | CFG weight for generation |

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

# Read audio prompt file
with open("voice_sample.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Make request
result = endpoint.run_sync({
    "text": "Hello world!",
    "audio_prompt": audio_b64,
    "exaggeration": 0.5
})

# Save output
audio_bytes = base64.b64decode(result["audio"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```
