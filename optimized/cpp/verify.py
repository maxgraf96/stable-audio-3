"""Python reference for the C++ spike.

Loads same_l_encoder_f32.npz, picks `mapping.weight` (shape 1536x512),
multiplies a deterministic input by W.T, and prints the first 10 values
of y[0,:] at full float32 precision so the C++ output can be diffed
against it bit-exactly.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
NPZ = REPO / "optimized/mlx/models/mlx/same_l_encoder_f32.npz"
WEIGHT_NAME = "mapping.weight"


def deterministic_input() -> mx.array:
    """x of shape (2, 512), values 0.00, 0.01, …, 5.11, 5.12, …, 10.23."""
    x = mx.arange(0, 1024, dtype=mx.float32) * 0.01
    return mx.reshape(x, (2, 512))


def main() -> int:
    z = np.load(NPZ)
    W = mx.array(z[WEIGHT_NAME])
    x = deterministic_input()
    y = x @ mx.transpose(W)
    mx.eval(y)

    print(f"W: {WEIGHT_NAME}  shape={tuple(W.shape)}  dtype={W.dtype}")
    print(f"x: shape={tuple(x.shape)}  dtype={x.dtype}")
    print(f"y: shape={tuple(y.shape)}  dtype={y.dtype}")
    print()
    print("First 10 values of y[0, :10] at full precision:")
    y_np = np.array(y[0, :10])
    for i, v in enumerate(y_np):
        # %.10g prints float32 with enough precision to be unique
        print(f"  y[0,{i:2d}] = {v:.10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
