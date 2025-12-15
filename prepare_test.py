"""
Reference Audio ko Base64 mein convert karo aur test_input.json generate karo
Usage: python prepare_test.py --audio reference.wav --text "Hello world"
"""

import argparse
import base64
import json
import os


def audio_to_base64(audio_path: str) -> str:
    """Convert audio file to base64 string"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_test_input(
    audio_path: str,
    text: str,
    model: str = "xtts",
    language: str = "en",
    output_file: str = "test_input.json"
):
    """Create test_input.json with base64 encoded reference audio"""

    print(f"📁 Reading audio file: {audio_path}")
    audio_b64 = audio_to_base64(audio_path)
    print(f"✅ Audio encoded: {len(audio_b64)} characters")

    test_input = {
        "input": {
            "model": model,
            "text": text,
            "reference_audio": audio_b64,
            "language": language
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_input, f, indent=4)

    print(f"✅ Saved to: {output_file}")
    print(f"\n📋 Test Input Details:")
    print(f"   Model: {model}")
    print(f"   Text: {text[:50]}...")
    print(f"   Language: {language}")
    print(f"   Audio size: {os.path.getsize(audio_path) / 1024:.1f} KB")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Prepare test input with reference audio")
    parser.add_argument("--audio", "-a", required=True, help="Path to reference audio file (WAV/MP3)")
    parser.add_argument("--text", "-t", default="Hello, this is a voice cloning test.", help="Text to synthesize")
    parser.add_argument("--model", "-m", default="xtts", choices=["xtts", "openvoice"], help="Model to use")
    parser.add_argument("--language", "-l", default="en", help="Language code (en, hi, etc.)")
    parser.add_argument("--output", "-o", default="test_input.json", help="Output JSON file")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ Error: Audio file not found: {args.audio}")
        return

    create_test_input(
        audio_path=args.audio,
        text=args.text,
        model=args.model,
        language=args.language,
        output_file=args.output
    )

    print(f"\n🚀 Ab test kar sakte ho:")
    print(f"   python test_local.py --local")


if __name__ == "__main__":
    main()
