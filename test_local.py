"""
Local testing script for RunPod Voice Cloning Handler
Run this before deploying to RunPod to verify everything works
"""

import requests
import base64
import json
import os
import sys

# For local testing without runpod
def test_handler_locally():
    """Test the handler function directly"""
    print("=" * 50)
    print("Testing Handler Locally")
    print("=" * 50)

    # Import handler
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Mock runpod to prevent startup
    import types
    runpod_mock = types.ModuleType('runpod')
    runpod_mock.serverless = types.SimpleNamespace()
    runpod_mock.serverless.start = lambda x: None
    sys.modules['runpod'] = runpod_mock

    from handler import handler, clone_with_kokoro

    # Test 1: Kokoro (no reference audio needed)
    print("\n[TEST 1] Testing Kokoro-82M...")
    test_job = {
        "input": {
            "model": "kokoro",
            "text": "Hello, this is a test of the Kokoro voice synthesis system.",
            "voice": "af_heart"
        }
    }

    result = handler(test_job)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success! Model used: {result['model_used']}")
        print(f"   Audio length: {len(result['audio'])} base64 chars")

        # Save output
        audio_bytes = base64.b64decode(result['audio'])
        with open("test_output_kokoro.wav", "wb") as f:
            f.write(audio_bytes)
        print(f"   Saved to: test_output_kokoro.wav")

    print("\n" + "=" * 50)
    print("Local Testing Complete")
    print("=" * 50)


def test_runpod_endpoint(endpoint_url: str, api_key: str):
    """Test deployed RunPod endpoint"""
    print("=" * 50)
    print("Testing RunPod Endpoint")
    print("=" * 50)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Test Kokoro
    payload = {
        "input": {
            "model": "kokoro",
            "text": "This is a test from the RunPod serverless endpoint.",
            "voice": "af_heart"
        }
    }

    print(f"\nSending request to: {endpoint_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(
        f"{endpoint_url}/runsync",
        headers=headers,
        json=payload,
        timeout=120
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        if "output" in result:
            output = result["output"]
            if "audio" in output:
                print("✅ Success!")
                audio_bytes = base64.b64decode(output["audio"])
                with open("test_output_runpod.wav", "wb") as f:
                    f.write(audio_bytes)
                print(f"   Saved to: test_output_runpod.wav")
            else:
                print(f"❌ No audio in response: {output}")
        else:
            print(f"Response: {result}")
    else:
        print(f"❌ Error: {response.text}")


def test_with_reference_audio(endpoint_url: str, api_key: str, audio_path: str):
    """Test voice cloning with reference audio"""
    print("=" * 50)
    print("Testing Voice Cloning with Reference Audio")
    print("=" * 50)

    # Read and encode reference audio
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Test XTTS
    payload = {
        "input": {
            "model": "xtts",
            "text": "This voice was cloned from the reference audio using XTTS version 2.",
            "reference_audio": audio_b64,
            "language": "en"
        }
    }

    print(f"\nSending request to: {endpoint_url}")
    print(f"Model: XTTS-v2")
    print(f"Reference audio: {audio_path}")

    response = requests.post(
        f"{endpoint_url}/runsync",
        headers=headers,
        json=payload,
        timeout=180
    )

    if response.status_code == 200:
        result = response.json()
        if "output" in result and "audio" in result["output"]:
            print("✅ Success!")
            audio_bytes = base64.b64decode(result["output"]["audio"])
            with open("test_output_cloned.wav", "wb") as f:
                f.write(audio_bytes)
            print(f"   Saved to: test_output_cloned.wav")
        else:
            print(f"Response: {result}")
    else:
        print(f"❌ Error: {response.text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Voice Cloning Handler")
    parser.add_argument("--local", action="store_true", help="Test locally without RunPod")
    parser.add_argument("--endpoint", type=str, help="RunPod endpoint URL")
    parser.add_argument("--api-key", type=str, help="RunPod API key")
    parser.add_argument("--reference", type=str, help="Path to reference audio for cloning")

    args = parser.parse_args()

    if args.local:
        test_handler_locally()
    elif args.endpoint and args.api_key:
        if args.reference:
            test_with_reference_audio(args.endpoint, args.api_key, args.reference)
        else:
            test_runpod_endpoint(args.endpoint, args.api_key)
    else:
        print("Usage:")
        print("  Local test:  python test_local.py --local")
        print("  RunPod test: python test_local.py --endpoint <URL> --api-key <KEY>")
        print("  With clone:  python test_local.py --endpoint <URL> --api-key <KEY> --reference <audio.wav>")
