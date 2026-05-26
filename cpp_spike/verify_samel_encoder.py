"""Python reference for cpp_spike/test_samel_encoder.cpp.

Loads the same npz, builds the same input (arange / total_size shaped 1x512x128),
runs the SAME-L encoder at fp32, and prints the first 16 output values plus
summary stats. Outputs should match the C++ side bit-exactly (or within fp32
rounding at worst — the deep transformer stack may accumulate ulp-level
differences from op-ordering even at the same Metal kernels).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "optimized/mlx"))
sys.path.insert(0, str(REPO / "optimized/mlx/scripts"))

from models.defs.same_l_encoder import load_model  # noqa: E402

NPZ = REPO / "optimized/mlx/models/mlx/same_l_encoder_f32.npz"


def main() -> int:
    model = load_model(weights_path=str(NPZ), dtype=mx.float32)

    B, C, T_aud = 1, 512, 128
    total = B * C * T_aud
    x = mx.arange(0, total, dtype=mx.float32) / float(total)
    x = mx.reshape(x, (B, C, T_aud))

    y = model(x)
    mx.eval(y)

    shape = tuple(y.shape)
    print(f"y shape=({', '.join(str(d) for d in shape)})")
    flat = np.array(y).reshape(-1)
    limit = min(len(flat), 16)
    for i in range(limit):
        print(f"  y[{i:3d}] = {flat[i]:.10g}")

    print(f"  mean    = {float(y.mean()):.10g}")
    # NB: mlx's std and our manual std calc are slightly different — use the same
    # formula as C++ (mean((x - mean)^2)) for bit-exact comparability.
    y_mean = mx.mean(y)
    y_std = mx.sqrt(mx.mean((y - y_mean) * (y - y_mean)))
    print(f"  std     = {float(y_std):.10g}")
    print(f"  abs.max = {float(mx.abs(y).max()):.10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
