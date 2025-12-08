"""
Patch script to fix weights_only and torchaudio issues in F5-TTS
Run this before importing F5-TTS modules
"""

from pathlib import Path


def patch_utils_infer():
    """Patch utils_infer.py to fix weights_only and torchaudio.load issues"""
    utils_file = Path(__file__).parent / "src" / "f5_tts" / "infer" / "utils_infer.py"

    if not utils_file.exists():
        print(f"Warning: {utils_file} not found")
        return

    content = utils_file.read_text()
    modified = False

    # Fix 1: Replace weights_only=True with weights_only=False
    if "weights_only=True" in content:
        content = content.replace("weights_only=True", "weights_only=False")
        print("✓ Patched: weights_only=True → weights_only=False")
        modified = True

    # Fix 2: Replace torchaudio.load() with scipy-based loading to avoid torchcodec
    old_infer_process = """def infer_process(
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
    audio, sr = torchaudio.load(ref_audio)"""

    new_infer_process = """def infer_process(
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
    sr, audio_np = wavfile.read(ref_audio)
    audio = torch.from_numpy(audio_np.astype(np.float32) / 32767.0)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T"""

    if old_infer_process in content:
        content = content.replace(old_infer_process, new_infer_process)
        print("✓ Patched: torchaudio.load() → scipy.io.wavfile.read()")
        modified = True

    if modified:
        utils_file.write_text(content)
        print("✓ Successfully patched utils_infer.py")
    else:
        print("✓ utils_infer.py already patched or doesn't need patching")


if __name__ == "__main__":
    patch_utils_infer()
