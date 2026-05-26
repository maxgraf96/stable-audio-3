"""Python reference for cpp_spike/test_t5gemma.cpp.

Runs the T5Gemma encoder against the same deterministic token IDs and prints
the same per-value + summary stats. Skips SentencePiece tokenization (the C++
side does too — that's a separate concern from encoder correctness).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "optimized/mlx"))
sys.path.insert(0, str(REPO / "optimized/mlx/scripts"))

from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402

NPZ = REPO / "optimized/mlx/models/mlx/t5gemma_f16.npz"


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
    t5 = T5Gemma.from_npz(str(NPZ))

    B, S = 1, 30
    ids = mx.arange(1, B * S + 1, dtype=mx.int32).reshape(B, S)
    mask = mx.ones((B, S), dtype=mx.int32)

    out = t5.encoder(ids, mask)
    mx.eval(out)
    dump("T5G", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
