# RunPod Deployment Setup

## Step 1: Docker Hub Setup

1. **Docker Hub account banao** (agar nahi hai): https://hub.docker.com/
2. **Access Token generate karo**:
   - Docker Hub pe login karo
   - Account Settings > Security > New Access Token
   - Token copy kar lo

## Step 2: GitHub Secrets Setup

1. GitHub repository pe jao: https://github.com/mudasser404/talkstudio-en
2. **Settings** > **Secrets and variables** > **Actions**
3. Do secrets add karo:
   - `DOCKER_USERNAME`: Tumhara Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub access token (jo step 1 me banaya)

## Step 3: Workflow Trigger

Ab jab bhi tum `Dockerfile` ko push karoge, automatically:
- GitHub Actions image build karega
- Docker Hub pe push karega
- Tag hoga: `your-username/f5-tts:latest`

## Step 4: RunPod Serverless Deployment

### For Serverless Endpoints (Recommended for API usage):

1. **RunPod** pe login karo: https://www.runpod.io/
2. **Serverless** > **+ New Endpoint**
3. Configuration:
   - **Container Image**: `your-docker-username/f5-tts:latest`
   - **Container Disk**: 10 GB minimum
   - **GPU**: Select GPU type (RTX 4090, A100, etc.)
   - **Active Workers**: 0 (auto-scale based on demand)
   - **Max Workers**: 3-5
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 600 seconds (for long audio generation)

4. **Environment Variables** (optional):
   ```
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

5. **Deploy** karo

### For Gradio Web UI (GPU Pods):

1. **Pods** > **+ Deploy**
2. **Container Image**: `your-docker-username/f5-tts:latest`
3. **Override Launch Command**:
   ```
   python -m f5_tts.infer.infer_gradio --host 0.0.0.0 --port 7860 --share
   ```
4. **Expose HTTP Ports**: `7860`
5. **Deploy** karo

## Step 5: Using the Serverless API

### Python Example:

```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Prepare input
with open("reference_audio.wav", "rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode()

# Run inference
result = endpoint.run_sync({
    "text": "Hello, this is a test of F5-TTS voice cloning!",
    "ref_audio": ref_audio_b64,
    "ref_text": "Reference text that was spoken in the audio",
    "remove_silence": True,
    "speed": 1.0
})

# Save output
if "audio" in result:
    audio_data = base64.b64decode(result["audio"])
    with open("output.wav", "wb") as f:
        f.write(audio_data)
```

### cURL Example:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Your text here",
      "ref_text": "Reference text",
      "remove_silence": true,
      "speed": 1.0
    }
  }'
```

## Access Gradio Interface (GPU Pods only)

- RunPod dashboard me **Connect** button pe click karo
- **Connect to HTTP Service [7860]** select karo
- Gradio interface khul jayega!

## Manual Build (Optional)

Agar manually build karna ho:

```bash
docker build -t your-username/f5-tts:latest .
docker push your-username/f5-tts:latest
```

## Troubleshooting

- **Build fail ho raha hai**: GitHub Actions logs check karo
- **RunPod pe start nahi ho raha**: Container logs dekho RunPod dashboard me
- **Models download nahi ho rahe**: Internet access check karo container me
