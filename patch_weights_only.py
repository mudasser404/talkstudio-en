"""
Patch script to fix weights_only and torchaudio issues in F5-TTS.

Run this BEFORE importing F5-TTS modules, or call patch_utils_infer()
from your handler.
"""

from pathlib import Path


def patch_utils_infer() -> None:
    """
    Patch src/f5_tts/infer/utils_infer.py to:

    1. Replace weights_only=True with weights_only=False so we don't require
       torchcodec's weights_only loader.

    2. Replace torchaudio.load(...) with a scipy-based loader to avoid
       libtorchcodec / ffmpeg compatibility issues.
    """
    utils_file = Path(__file__).parent / "src" / "f5_tts" / "infer" / "utils_infer.py"

    if not utils_file.exists():
        print(f"[patch_weights_only] Warning: {utils_file} not found")
        return

    content = utils_file.read_text()
    modified = False

    # ---- Fix 1: weights_only=True → weights_only=False ----------------------
    if "weights_only=True" in content:
        content = content.replace("weights_only=True", "weights_only=False")
        print("[patch_weights_only] ✓ Patched: weights_only=True → weights_only=False")
        modified = True

    # ---- Fix 2: torchaudio.load(...) → scipy.io.wavfile.read(...) ----------
    old_infer_process = '''def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)'''

    new_infer_process = '''def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    # Load audio using scipy to avoid torchcodec dependency
    from scipy.io import wavfile
    import numpy as np
    import torch

    sr, audio_np = wavfile.read(ref_audio)
    audio = torch.from_numpy(audio_np.astype(np.float32) / 32767.0)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        # (time, channels) -> (channels, time)
        audio = audio.T'''

    if old_infer_process in content:
        content = content.replace(old_infer_process, new_infer_process)
        print("[patch_weights_only] ✓ Patched: torchaudio.load() → scipy.io.wavfile.read()")
        modified = True

    # ---- Write back if changed ----------------------------------------------
    if modified:
        utils_file.write_text(content)
        print("[patch_weights_only] ✓ Successfully patched utils_infer.py")
    else:
        print("[patch_weights_only] ✓ No changes needed (already patched?)")


if __name__ == "__main__":
    patch_utils_infer()
