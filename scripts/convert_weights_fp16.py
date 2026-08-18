#!/usr/bin/env python3
"""Convert SA3 torch checkpoints to fp16 for redistribution.

Run on the release machine, not shipped to users. Produces the directory layout
that sa3_pipeline_torch.load_local_model() expects:

    <out>/medium/model_config.json
    <out>/medium/model.safetensors            fp16
    <out>/medium/t5gemma-b-b-ul2/...          fp16 weights + tokenizer files

Why fp16 is safe here: the plugin loads every model with model_half=True on
CUDA, so the weights are cast to fp16 at load time anyway. Converting on disk
just moves the same rounding earlier — the tensors the model runs with are
bit-identical. It halves both the download (medium: 9.2 GB -> ~4.6 GB) and the
load time. The one behavioural difference is a CPU fallback, which would
otherwise have run true fp32; it now upcasts fp16 -> fp32 instead.

Buffers and anything integer/bool stay untouched: casting a position index or a
mask to fp16 would corrupt it.

Usage:
    python scripts/convert_weights_fp16.py --out dist/models
    python scripts/convert_weights_fp16.py --out dist/models --models medium

Then verify before uploading — this is the step that could silently degrade
audio quality, so it is worth the two minutes:

    python scripts/convert_weights_fp16.py --out dist/models --verify <source.wav>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Must be set before huggingface_hub is imported. Windows refuses os.symlink
# without Developer Mode or admin rights, and snapshot_download() hard-fails on
# that (hf_hub_download degrades to copies, but snapshot_download does not).
# Copies cost disk we're about to stop using anyway — this is a one-shot
# release step.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

# The plugin's three model kinds, by their upstream repo names.
MODELS = {
    "medium": "stabilityai/stable-audio-3-medium",
    "small-music": "stabilityai/stable-audio-3-small-music",
    "small-sfx": "stabilityai/stable-audio-3-small-sfx",
}


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def cast_safetensors(src: Path, dst: Path) -> tuple[int, int]:
    """Cast every floating-point tensor in `src` to fp16. Returns (before, after)."""
    tensors = load_file(str(src))
    out = {}
    for k, v in tensors.items():
        # Only real floating types. Integer indices, masks and bools must survive
        # untouched, and anything already fp16 is left alone.
        out[k] = v.to(torch.float16) if v.dtype in (torch.float32, torch.float64) else v
    dst.parent.mkdir(parents=True, exist_ok=True)
    # safetensors refuses shared storage; contiguous clones avoid that entirely.
    save_file({k: v.contiguous().clone() for k, v in out.items()}, str(dst),
              metadata={"format": "pt"})
    return src.stat().st_size, dst.stat().st_size


def convert(name: str, repo: str, out_root: Path) -> Path:
    print(f"\n=== {name} ({repo}) ===")
    snap = Path(snapshot_download(repo))
    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=True)

    before, after = cast_safetensors(snap / "model.safetensors", dest / "model.safetensors")
    print(f"  model.safetensors  {human(before)} -> {human(after)}")

    shutil.copy2(snap / "model_config.json", dest / "model_config.json")

    # The text encoder ships as a subfolder of the same repo. Cast its weights
    # too and carry the tokenizer/config files across verbatim.
    t5_src = snap / "t5gemma-b-b-ul2"
    if t5_src.is_dir():
        t5_dst = dest / "t5gemma-b-b-ul2"
        t5_dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(t5_src.iterdir()):
            if not f.is_file():
                continue
            if f.suffix == ".safetensors":
                b, a = cast_safetensors(f, t5_dst / f.name)
                print(f"  {f.name:<26} {human(b)} -> {human(a)}")
            else:
                shutil.copy2(f, t5_dst / f.name)
    else:
        print("  warning: no t5gemma-b-b-ul2/ subfolder — the local load will need "
              "a text encoder from somewhere else", file=sys.stderr)

    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"  total: {human(total)}  ->  {dest}")
    return dest


def verify(name: str, out_root: Path, source_wav: Path) -> bool:
    """Generate the same seeds from the hub fp32 model and the converted one.

    Correlation should be ~1.0. Anything materially below that means the cast
    changed the audio, not just the storage.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import sa3_pipeline_torch as spt

    audio = spt.read_wav_44k_stereo_s16(str(source_wav))
    secs = audio.shape[-1] / spt.SAMPLE_RATE
    dit = next(k for k, v in spt.DIT_CHOICES.items() if v["model"] == name)
    decoder = spt.DIT_CHOICES[dit]["decoder"]

    def run(models_dir: str | None) -> list:
        import os
        prev = os.environ.get("SA3_MODELS_DIR")
        if models_dir is None:
            os.environ.pop("SA3_MODELS_DIR", None)
        else:
            os.environ["SA3_MODELS_DIR"] = models_dir
        try:
            pipe = spt.Pipeline(dit=dit, decoder=decoder, seconds=secs, verbose=True)
            return [pipe.generate(prompt="", seed=s, steps=8, init_noise_level=0.45,
                                  cfg=1.0, init_audio_np=audio) for s in (1234, 2243, 3252)]
        finally:
            if prev is None:
                os.environ.pop("SA3_MODELS_DIR", None)
            else:
                os.environ["SA3_MODELS_DIR"] = prev

    print("\n=== verifying fp16 conversion ===")
    hub = run(None)
    local = run(str(out_root))
    ok = True
    for i, (a, b) in enumerate(zip(hub, local)):
        n = min(a.shape[-1], b.shape[-1])
        c = float(np.corrcoef(a[:, :n].ravel(), b[:, :n].ravel())[0, 1])
        flag = "" if c > 0.999 else "   <-- TOO LOW"
        ok &= c > 0.999
        print(f"  seed {i}: corr(hub fp32, local fp16) = {c:+.6f}{flag}")
    print("  PASS" if ok else "  FAIL — do not upload this conversion")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                    help="which models to convert (default: all three)")
    ap.add_argument("--verify", type=Path, metavar="SOURCE_WAV",
                    help="after converting, compare fp16 output against the hub fp32 model")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        convert(name, MODELS[name], args.out)

    if args.verify:
        if not verify("medium" if "medium" in args.models else args.models[0],
                      args.out, args.verify):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
