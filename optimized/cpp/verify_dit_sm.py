"""Bit-exactness reference for the sm-music DiT port (Python side).

Runs the MLX sm-music DiT on deterministic inputs and dumps shape + first 16
values + stats. The C++ test_dit_sm must match this output.

Usage:
    python verify_dit_sm.py [T_lat]
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx"))

from models.defs.dit_mlx import load_dit  # noqa: E402  (sm-music DiT)

NPZ = (Path(__file__).resolve().parent.parent / "mlx" / "models" / "mlx"
       / "dit_sm-music_f16.npz")


def dump(name, a):
    a = np.array(a.astype(mx.float32))
    print(f"{name} shape={list(a.shape)}")
    flat = a.flatten()
    print("  first16:", " ".join(f"{v:.6f}" for v in flat[:16]))
    print(f"  mean={flat.mean():.6f} std={flat.std():.6f} "
          f"absmax={np.abs(flat).max():.6f} n={flat.size}")


def main():
    T_lat = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    dt = mx.float16 if (len(sys.argv) > 2 and sys.argv[2] == "fp16") else mx.float32
    m = load_dit(str(NPZ), T_lat=T_lat, dtype=dt)

    def make(shape):
        total = int(np.prod(shape))
        return (mx.reshape(mx.arange(total, dtype=mx.float32), shape) / total).astype(dt)

    x = make([1, 256, T_lat])
    t = mx.array([0.5]).astype(dt)
    cross = make([1, 257, 768])
    gcond = make([1, 768])

    v = m(x, t, cross, gcond)
    dump("v", v)
    print(f"seed-free deterministic check, T_lat={T_lat}")


if __name__ == "__main__":
    main()
