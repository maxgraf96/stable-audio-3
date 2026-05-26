"""Python reference for optimized/cpp/test_sa3_pipeline.cpp.

Mirrors each test exactly with the original sa3_pipeline.py functions. Outputs
should match the C++ side bit-exactly at fp32.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "optimized/mlx"))
sys.path.insert(0, str(REPO / "optimized/mlx/scripts"))

from models.defs.sa3_pipeline import (  # noqa: E402
    SecondsTotalEmbedder,
    apply_prompt_padding,
    build_pingpong_schedule,
    expo_fourier_features,
    logsnr_shift,
    patched_decode,
    sample_flow_pingpong,
)


def dump(label: str, a: mx.array, n: int = -1) -> None:
    f = a.astype(mx.float32)
    mx.eval(f)
    flat = np.array(f).reshape(-1)
    shape = tuple(f.shape)
    print(f"[{label}] shape=({', '.join(str(d) for d in shape)})")
    limit = len(flat) if n < 0 else min(n, len(flat))
    for i in range(limit):
        # %.10g matches std::cout.precision(10) on the C++ side
        print(f"  [{i:3d}] {flat[i]:.10g}")


def main() -> int:
    # T1
    s = build_pingpong_schedule(8, sigma_max=1.0, use_logsnr_shift=True)
    dump("T1 schedule(8, 1.0)", s)

    # T2
    s = build_pingpong_schedule(4, sigma_max=0.52, use_logsnr_shift=True)
    dump("T2 schedule(4, 0.52)", s)

    # T3
    t = mx.arange(0, 11, dtype=mx.float32) / 10.0
    dump("T3 logsnr_shift", logsnr_shift(t))

    # T4
    t = mx.arange(0, 4, dtype=mx.float32) / 4.0
    dump("T4 fourier_features", expo_fourier_features(t, dim=8))

    # T5
    embeds = mx.arange(1, 5, dtype=mx.float32)
    embeds = mx.expand_dims(mx.expand_dims(embeds, 0), -1)
    embeds = mx.broadcast_to(embeds, (1, 4, 3))
    mask = mx.array([[1, 1, 0, 0]], dtype=mx.int32)
    pad = mx.array([9.0, 8.0, 7.0])
    dump("T5 apply_prompt_padding", apply_prompt_padding(embeds, mask, pad))

    # T6
    p = mx.arange(0, 12, dtype=mx.float32).reshape(1, 4, 3)
    dump("T6 patched_decode", patched_decode(p, patch_size=2, channels=2))

    # T7
    def model_fn(x, t):
        return 0.5 * x

    sigmas = build_pingpong_schedule(4, sigma_max=1.0, use_logsnr_shift=True)
    x0 = mx.ones((1, 4), dtype=mx.float32)
    y = sample_flow_pingpong(model_fn, x0, sigmas, seed=42)
    dump("T7 sample_flow_pingpong", y)

    # T8
    OUT, IN = 768, 256
    W = mx.array([(i % 13) * 0.001 for i in range(OUT * IN)], dtype=mx.float32).reshape(OUT, IN)
    b = mx.array([(i % 7) * 0.01 for i in range(OUT)], dtype=mx.float32)
    emb = SecondsTotalEmbedder(W, b, min_val=0.0, max_val=384.0, fourier_dim=256)
    out = emb(30.0)
    dump("T8 seconds_embedder(30.0)", out, n=12)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
