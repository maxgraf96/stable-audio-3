"""Python reference for optimized/cpp/test_samel_decoder.cpp.

Runs both:
  T1 single-shot: input (1, 256, 16), bit-exact decode
  T2 chunked:     input (1, 256, 64), decode_chunked with chunk=8 overlap=8
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "optimized/mlx"))
sys.path.insert(0, str(REPO / "optimized/mlx/scripts"))

from models.defs.same_l_decoder import decode_chunked, load_model  # noqa: E402

NPZ = REPO / "optimized/mlx/models/mlx/same_l_decoder_f32.npz"


def dump(label: str, a: mx.array, n: int = 16) -> None:
    f = a.astype(mx.float32)
    mx.eval(f)
    shape = tuple(f.shape)
    print(f"[{label}] shape=({', '.join(str(d) for d in shape)})")
    flat = np.array(f).reshape(-1)
    limit = min(len(flat), n)
    for i in range(limit):
        print(f"  {label}[{i:3d}] = {flat[i]:.10g}")
    mean = mx.mean(f)
    std_ = mx.sqrt(mx.mean((f - mean) * (f - mean)))
    absmax = mx.abs(f).max()
    print(f"  {label} mean    = {float(mean):.10g}")
    print(f"  {label} std     = {float(std_):.10g}")
    print(f"  {label} abs.max = {float(absmax):.10g}")


def main() -> int:
    model = load_model(weights_path=str(NPZ), dtype=mx.float32)

    # T1 single-shot
    B, C, T_lat = 1, 256, 16
    total = B * C * T_lat
    x1 = (mx.arange(0, total, dtype=mx.float32) / float(total)).reshape(B, C, T_lat)
    y1 = model(x1)
    mx.eval(y1)
    dump("T1 single", y1)

    # T2 chunked
    B, C, T_lat = 1, 256, 64
    total = B * C * T_lat
    x2 = (mx.arange(0, total, dtype=mx.float32) / float(total)).reshape(B, C, T_lat)
    y2 = decode_chunked(model, x2, chunk_size=8, overlap=8)
    mx.eval(y2)
    dump("T2 chunked", y2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
