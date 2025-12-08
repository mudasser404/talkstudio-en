"""
Patch script to fix weights_only issue in F5-TTS
Run this before importing F5-TTS modules
"""

import sys
from pathlib import Path

def patch_utils_infer():
    """Patch utils_infer.py to use weights_only=False"""
    utils_file = Path(__file__).parent / "src" / "f5_tts" / "infer" / "utils_infer.py"

    if not utils_file.exists():
        print(f"Warning: {utils_file} not found")
        return

    content = utils_file.read_text()

    # Replace weights_only=True with weights_only=False
    if "weights_only=True" in content:
        new_content = content.replace("weights_only=True", "weights_only=False")
        utils_file.write_text(new_content)
        print("✓ Patched utils_infer.py: weights_only=True → weights_only=False")
    else:
        print("✓ utils_infer.py already patched or doesn't need patching")

if __name__ == "__main__":
    patch_utils_infer()
