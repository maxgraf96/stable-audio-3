"""One-time npz → safetensors converter.

mlx-c++ exposes load_safetensors() but no public load_npz(). So we re-export
the .npz weight files as .safetensors once, then the C++ plugin loads them
directly.

Usage:
    python cpp_spike/convert_npz.py optimized/mlx/models/mlx/same_l_encoder_f32.npz

Writes <input_stem>.safetensors next to the input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert(npz_path: Path) -> Path:
    out_path = npz_path.with_suffix(".safetensors")
    z = np.load(npz_path)
    tensors: dict[str, mx.array] = {}
    for key in z.files:
        arr = z[key]
        # Preserve original dtype (incl. uint8 for embedded SentencePiece bytes).
        tensors[key] = mx.array(arr)
    mx.save_safetensors(str(out_path), tensors)
    total_mb = sum(a.nbytes for a in tensors.values()) / 1024**2
    print(f"  {npz_path.name:40s} → {out_path.name}")
    print(f"  arrays: {len(tensors):>4d}    size: {total_mb:>7.1f} MB")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+", help="One or more .npz files to convert")
    args = ap.parse_args()
    for p in args.npz:
        if not p.exists():
            raise SystemExit(f"error: {p} not found")
        convert(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
