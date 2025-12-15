"""
RunPod Voice Cloning Client
Use this to call your deployed RunPod endpoint from any Python project
"""

import requests
import base64
import time
import os
from typing import Optional, Dict, Any
from pathlib import Path


class RunPodVoiceCloning:
    """Client for RunPod Voice Cloning Endpoint"""

    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize the client

        Args:
            endpoint_id: Your RunPod endpoint ID
            api_key: Your RunPod API key
        """
        self.endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _encode_audio(self, audio_path: str) -> str:
        """Encode audio file to base64"""
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _decode_audio(self, audio_b64: str, output_path: str) -> str:
        """Decode base64 audio and save to file"""
        audio_bytes = base64.b64decode(audio_b64)
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path

    def generate_sync(
        self,
        text: str,
        model: str = "kokoro",
        reference_audio: Optional[str] = None,
        language: str = "en",
        voice: str = "af_heart",
        output_path: Optional[str] = None,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Generate speech synchronously (waits for result)

        Args:
            text: Text to synthesize
            model: Model to use (kokoro, xtts, openvoice)
            reference_audio: Path to reference audio file (for xtts/openvoice)
            language: Language code
            voice: Voice preset (for kokoro)
            output_path: Where to save the output audio
            timeout: Request timeout in seconds

        Returns:
            Dict with 'audio_path' and 'model_used'
        """
        payload = {
            "input": {
                "model": model,
                "text": text,
                "language": language,
                "voice": voice
            }
        }

        # Add reference audio if provided
        if reference_audio:
            payload["input"]["reference_audio"] = self._encode_audio(reference_audio)

        # Make request
        response = requests.post(
            f"{self.endpoint_url}/runsync",
            headers=self.headers,
            json=payload,
            timeout=timeout
        )

        if response.status_code != 200:
            raise Exception(f"RunPod error: {response.text}")

        result = response.json()

        if "error" in result:
            raise Exception(f"RunPod error: {result['error']}")

        output = result.get("output", {})

        if "error" in output:
            raise Exception(f"Handler error: {output['error']}")

        # Save audio
        if output_path is None:
            output_path = f"output_{model}_{int(time.time())}.wav"

        self._decode_audio(output["audio"], output_path)

        return {
            "audio_path": output_path,
            "model_used": output.get("model_used", model),
            "status": "success"
        }

    def generate_async(
        self,
        text: str,
        model: str = "kokoro",
        reference_audio: Optional[str] = None,
        language: str = "en",
        voice: str = "af_heart"
    ) -> str:
        """
        Start async generation (returns job ID)

        Args:
            text: Text to synthesize
            model: Model to use
            reference_audio: Path to reference audio
            language: Language code
            voice: Voice preset

        Returns:
            Job ID string
        """
        payload = {
            "input": {
                "model": model,
                "text": text,
                "language": language,
                "voice": voice
            }
        }

        if reference_audio:
            payload["input"]["reference_audio"] = self._encode_audio(reference_audio)

        response = requests.post(
            f"{self.endpoint_url}/run",
            headers=self.headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"RunPod error: {response.text}")

        result = response.json()
        return result["id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of async job"""
        response = requests.get(
            f"{self.endpoint_url}/status/{job_id}",
            headers=self.headers
        )

        return response.json()

    def wait_for_job(
        self,
        job_id: str,
        output_path: Optional[str] = None,
        poll_interval: float = 1.0,
        max_wait: int = 300
    ) -> Dict[str, Any]:
        """
        Wait for async job to complete

        Args:
            job_id: The job ID from generate_async
            output_path: Where to save output audio
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            Dict with 'audio_path' and 'model_used'
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_job_status(job_id)

            if status["status"] == "COMPLETED":
                output = status.get("output", {})

                if "error" in output:
                    raise Exception(f"Handler error: {output['error']}")

                if output_path is None:
                    output_path = f"output_{job_id}.wav"

                self._decode_audio(output["audio"], output_path)

                return {
                    "audio_path": output_path,
                    "model_used": output.get("model_used"),
                    "status": "success"
                }

            elif status["status"] == "FAILED":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")

            time.sleep(poll_interval)

        raise Exception(f"Job timed out after {max_wait} seconds")

    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        output_path: Optional[str] = None,
        model: str = "xtts",
        language: str = "en"
    ) -> str:
        """
        Convenience method for voice cloning

        Args:
            text: Text to speak
            reference_audio: Path to reference audio file
            output_path: Where to save output
            model: Which model to use (xtts or openvoice)
            language: Language code

        Returns:
            Path to generated audio file
        """
        result = self.generate_sync(
            text=text,
            model=model,
            reference_audio=reference_audio,
            language=language,
            output_path=output_path
        )
        return result["audio_path"]

    def quick_tts(
        self,
        text: str,
        voice: str = "af_heart",
        output_path: Optional[str] = None
    ) -> str:
        """
        Quick TTS without cloning using Kokoro

        Args:
            text: Text to speak
            voice: Voice preset
            output_path: Where to save output

        Returns:
            Path to generated audio file
        """
        result = self.generate_sync(
            text=text,
            model="kokoro",
            voice=voice,
            output_path=output_path
        )
        return result["audio_path"]


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = RunPodVoiceCloning(
        endpoint_id="YOUR_ENDPOINT_ID",
        api_key="YOUR_API_KEY"
    )

    # Quick TTS (no reference audio needed)
    print("Testing quick TTS...")
    audio_path = client.quick_tts(
        text="Hello, this is a quick test of the voice synthesis system.",
        voice="af_heart",
        output_path="quick_tts_output.wav"
    )
    print(f"Saved to: {audio_path}")

    # Voice cloning (requires reference audio)
    # print("Testing voice cloning...")
    # audio_path = client.clone_voice(
    #     text="This voice was cloned from the reference audio.",
    #     reference_audio="path/to/reference.wav",
    #     output_path="cloned_output.wav"
    # )
    # print(f"Saved to: {audio_path}")
