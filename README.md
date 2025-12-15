# RunPod Voice Cloning - Speaker Embedding Based

Fast voice cloning using speaker embeddings. Supports multiple models:

| Model | Speed | Reference Audio | Best For |
|-------|-------|-----------------|----------|
| **Kokoro-82M** | Fastest | Not needed | Quick TTS, no cloning |
| **XTTS-v2** | Fast | 6 sec | Multilingual cloning |
| **OpenVoice V2** | Fast | 10 sec | Emotion/tone control |

## Quick Start

### 1. Build Docker Image

```bash
docker build -t voice-cloning-runpod .
```

### 2. Push to Docker Hub

```bash
docker tag voice-cloning-runpod:latest YOUR_DOCKERHUB/voice-cloning-runpod:latest
docker push YOUR_DOCKERHUB/voice-cloning-runpod:latest
```

### 3. Deploy on RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Create New Endpoint
3. Select your Docker image
4. Choose GPU (RTX 3090/4090 recommended)
5. Set container disk to 20GB+
6. Deploy!

## API Usage

### Endpoint: `/runsync` or `/run`

### Request Format

```json
{
    "input": {
        "model": "xtts",
        "text": "Hello, this is a test.",
        "reference_audio": "BASE64_ENCODED_AUDIO",
        "language": "en"
    }
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | `kokoro`, `xtts`, or `openvoice` |
| `text` | string | Yes | Text to synthesize |
| `reference_audio` | string | For xtts/openvoice | Base64 encoded WAV audio |
| `language` | string | No | Language code (default: `en`) |
| `voice` | string | For kokoro | Voice preset (default: `af_heart`) |

### Response Format

```json
{
    "audio": "BASE64_ENCODED_WAV",
    "model_used": "xtts",
    "status": "success"
}
```

## Examples

### Python - Kokoro (No Reference Audio)

```python
import requests
import base64

ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID"
API_KEY = "your_api_key"

response = requests.post(
    f"{ENDPOINT}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "model": "kokoro",
            "text": "Hello, this is a fast voice synthesis test.",
            "voice": "af_heart"
        }
    }
)

result = response.json()
audio_bytes = base64.b64decode(result["output"]["audio"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### Python - XTTS Voice Cloning

```python
import requests
import base64

# Read reference audio
with open("reference.wav", "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{ENDPOINT}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "input": {
            "model": "xtts",
            "text": "This voice was cloned from the reference audio.",
            "reference_audio": ref_audio_b64,
            "language": "en"
        }
    }
)

result = response.json()
audio_bytes = base64.b64decode(result["output"]["audio"])
with open("cloned_output.wav", "wb") as f:
    f.write(audio_bytes)
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "model": "kokoro",
      "text": "Hello from RunPod!",
      "voice": "af_heart"
    }
  }'
```

## Kokoro Voice Presets

| Voice | Description |
|-------|-------------|
| `af_heart` | American Female (warm) |
| `af_bella` | American Female (professional) |
| `af_sarah` | American Female (casual) |
| `am_adam` | American Male |
| `am_michael` | American Male (deep) |
| `bf_emma` | British Female |
| `bm_george` | British Male |

## Supported Languages (XTTS)

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Polish (pl)
- Turkish (tr)
- Russian (ru)
- Dutch (nl)
- Czech (cs)
- Arabic (ar)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Hindi (hi)

## Local Testing

```bash
# Test locally without RunPod
python test_local.py --local

# Test deployed endpoint
python test_local.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY

# Test with voice cloning
python test_local.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY --reference my_voice.wav
```

## GPU Requirements

| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| Kokoro-82M | ~2GB | Any |
| XTTS-v2 | ~4GB | RTX 3060+ |
| OpenVoice V2 | ~4GB | RTX 3060+ |
| All Models | ~8GB | RTX 3090/4090 |

## Troubleshooting

### Out of Memory
- Use a larger GPU
- Load only one model at a time (modify handler.py)

### Slow First Request
- First request downloads/loads models
- Subsequent requests are much faster

### Audio Quality Issues
- Use clean reference audio (no background noise)
- Reference audio should be 5-15 seconds
- Use WAV format, 16kHz+ sample rate

## Cost Estimation (RunPod)

- Cold start: ~60-120 seconds
- Inference: ~2-5 seconds per request
- Approximate cost: $0.0002-0.001 per request (depends on GPU)
