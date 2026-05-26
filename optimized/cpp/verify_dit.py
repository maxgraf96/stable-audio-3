"""Python reference for optimized/cpp/test_dit.cpp.

Same fp32 forward pass with the same deterministic inputs (arange/total).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "optimized/mlx"))
sys.path.insert(0, str(REPO / "optimized/mlx/scripts"))

from models.defs.dit_mlx_medium import load_dit  # noqa: E402

NPZ = REPO / "optimized/mlx/models/mlx/dit_medium_f16.npz"
T_LAT = 16


def dump(label: str, a: mx.array, n: int = 16) -> None:
    a_fp32 = a.astype(mx.float32)
    mean = mx.mean(a_fp32)
    std_ = mx.sqrt(mx.mean((a_fp32 - mean) * (a_fp32 - mean)))
    absmax = mx.abs(a_fp32).max()
    mx.eval(mean, std_, absmax)
    shape = tuple(a.shape)
    print(f"[{label}] shape=({', '.join(str(d) for d in shape)})")
    flat = np.array(a_fp32).reshape(-1)
    limit = min(len(flat), n)
    for i in range(limit):
        print(f"  {label}[{i:3d}] = {flat[i]:.10g}")
    print(f"  {label} mean    = {float(mean):.10g}")
    print(f"  {label} std     = {float(std_):.10g}")
    print(f"  {label} abs.max = {float(absmax):.10g}")


def main() -> int:
    model = load_dit(str(NPZ), T_lat=T_LAT, dtype=mx.float32)

    def make(shape):
        total = 1
        for d in shape:
            total *= d
        return (mx.arange(0, total, dtype=mx.float32) / float(total)).reshape(shape)

    x     = make((1, 256, T_LAT))
    t     = mx.array([0.5])
    cross = make((1, 257, 768))
    gcond = make((1, 768))

    out = model(x, t, cross, gcond)
    mx.eval(out)
    dump("DiT", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
