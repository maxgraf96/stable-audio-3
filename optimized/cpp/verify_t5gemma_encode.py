"""Python reference for optimized/cpp/test_t5gemma_encode.cpp.

Tokenizes the exact same prompts via SentencePiece + runs the T5Gemma encoder.
Prints the same labelled IDs / mask / embedding values so a `diff` against the
C++ output is meaningful.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "optimized/mlx"))
sys.path.insert(0, str(REPO / "optimized/mlx/scripts"))

from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402

NPZ = REPO / "optimized/mlx/models/mlx/t5gemma_f16.npz"


def print_ids(label: str, ids: mx.array) -> None:
    mx.eval(ids)
    arr = np.array(ids)
    B, S = arr.shape
    print(f"[{label}] shape=({B}, {S})")
    for i in range(B):
        row = ", ".join(str(int(x)) for x in arr[i])
        print(f"  row {i}: [{row}]")


def dump_embeds(label: str, a: mx.array, n: int = 16) -> None:
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

    prompts = [
        "A beautiful piano arpeggio grows into a grand cinematic climax",
        "lofi house loop",
        "Amen break 174 BPM",
        "",
    ]
    max_len = 20

    ids, mask = t5.tokenize(prompts, max_len=max_len)
    print_ids("ids", ids)
    print_ids("mask", mask)

    embeds, _mask = t5.encode(prompts, max_len=max_len)
    mx.eval(embeds)
    dump_embeds("T5G", embeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
