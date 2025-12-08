"""
Local test script for F5-TTS handler
Tests the handler without RunPod infrastructure
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_handler():
    """Test the handler with test.json input"""

    # Import handler after path is set
    from handler import generate_speech

    # Load test input
    with open("test.json", "r") as f:
        test_data = json.load(f)

    print("=" * 60)
    print("Testing F5-TTS Handler")
    print("=" * 60)
    print(f"\nInput text: {test_data['input']['text'][:100]}...")
    print(f"Reference audio URL: {test_data['input']['ref_audio_url']}")
    print(f"Speed: {test_data['input']['speed']}")
    print("\nStarting inference...\n")

    # Create job format
    job = test_data

    # Run handler
    try:
        result = generate_speech(job)

        if "error" in result:
            print(f"\n❌ ERROR: {result['error']}\n")
            return False

        if "audio" in result:
            # Save output audio
            import base64

            audio_data = base64.b64decode(result["audio"])
            output_file = "test_output.wav"
            with open(output_file, "wb") as f:
                f.write(audio_data)

            print("\n✅ SUCCESS!")
            print(f"   Output saved: {output_file}")
            print(f"   Sample rate: {result['sample_rate']} Hz")
            print(f"   Audio size: {len(audio_data)} bytes")
            return True
        else:
            print(f"\n❌ No audio in result: {result}")
            return False

    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_handler()
    sys.exit(0 if success else 1)
