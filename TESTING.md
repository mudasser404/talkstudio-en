# F5-TTS Handler Testing Guide

## Local Testing (Recommended before deployment)

### Prerequisites

```bash
# Install dependencies
pip install torch torchaudio numpy cached_path requests safetensors vocos
pip install -e .
```

### Run Test

```bash
# Test with test.json
python test_handler.py
```

### Expected Output

```
============================================================
Testing F5-TTS Handler
============================================================

Input text: Hello, this is a test of the F5 TTS multilingual model...
Reference audio URL: https://talkstudio.ai/media/library_voices/...
Speed: 1.0

Starting inference...

Loading F5-TTS model...
Downloading checkpoint from HuggingFace...
✓ Patched utils_infer.py: weights_only=True → weights_only=False
Checkpoint cached at: ...
Loading vocoder...
Models loaded successfully!

Downloading audio from https://talkstudio.ai/...
Converting .mp3 to WAV...
Generating speech for: Hello, this is a test of the F5 TTS multilingu...

✅ SUCCESS!
   Output saved: test_output.wav
   Sample rate: 24000 Hz
   Audio size: XXXXX bytes
```

## Test Different Scenarios

### Test 1: With URL (Current test.json)
```json
{
  "input": {
    "text": "Your text here",
    "ref_audio_url": "https://example.com/audio.mp3",
    "ref_text": "",
    "speed": 1.0
  }
}
```

### Test 2: With Base64 Audio
```json
{
  "input": {
    "text": "Your text here",
    "ref_audio_base64": "BASE64_ENCODED_AUDIO_HERE",
    "ref_text": "Reference text",
    "speed": 1.0
  }
}
```

### Test 3: Without Reference Audio (Use default voice)
```json
{
  "input": {
    "text": "Your text here",
    "speed": 1.0
  }
}
```

## Docker Test

### Build Image
```bash
docker build -t f5-tts:test .
```

### Run Container
```bash
# Start container
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/test.json:/workspace/F5-TTS/test.json \
  f5-tts:test
```

### Test with RunPod Format
```bash
# Install runpod locally
pip install runpod

# Create test_runpod.py
python -c "
import runpod
import json

with open('test.json') as f:
    test_data = json.load(f)

# Simulate RunPod job
from handler import generate_speech
result = generate_speech(test_data)
print(result)
"
```

## Common Issues & Solutions

### Issue 1: Model download fails
**Solution**: Check internet connection and HuggingFace access

### Issue 2: weights_only error
**Solution**: Patch should automatically apply. Check logs for "Patched utils_infer.py"

### Issue 3: MP3 conversion fails
**Solution**: Install ffmpeg
```bash
# Ubuntu/Debian
apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Issue 4: CUDA out of memory
**Solution**: Use smaller batch or lower precision
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Issue 5: Invalid load key error
**Solution**: Handler should auto-fix with symlink. Check logs for ".safetensors" extension

## Performance Benchmarks

Expected timings (RTX 4090):
- Cold start (first request): ~30-45 seconds (model download + load)
- Warm start (cached model): ~3-5 seconds (model load only)
- Generation (per 10 seconds of audio): ~2-4 seconds

## Verify Before Deployment

✅ Local test passes
✅ Output audio plays correctly
✅ No errors in logs
✅ Models load successfully
✅ Reference audio downloads/converts
✅ Docker build succeeds

## Deploy to RunPod

Once all tests pass:

1. Push code to GitHub
2. GitHub Actions builds Docker image
3. Deploy to RunPod Serverless
4. Test with real API calls

See [RUNPOD_SETUP.md](RUNPOD_SETUP.md) for deployment instructions.
