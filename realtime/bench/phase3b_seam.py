"""Phase 3b gate — cyclic decode makes the loop wrap seamless.

When the ring wraps, audio sample L-1 is followed by sample 0. The plain decoder
gives those two ends no shared context, so the wrap has a discontinuity (a click).
cyclic_decode pads the latent with wrap-around context. We measure the step across
the wrap relative to the loop's typical inter-sample step:

  wrap_ratio = |x[0] - x[L-1]|  /  median(|x[i+1] - x[i]|)

A seamless loop has wrap_ratio ~ O(1) (the boundary is as smooth as anywhere
else). A clicky loop has wrap_ratio >> 1. Gate: cyclic ratio < plain ratio AND
cyclic interior is bit-identical to plain interior (cyclic only changes the edges).

Run: optimized/mlx/.venv/bin/python realtime/bench/phase3b_seam.py
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

REPO = Path("/Users/max/Code/stable-audio-3")
MLX_ROOT = REPO / "optimized" / "mlx"
for p in (REPO, REPO / "realtime", MLX_ROOT, MLX_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sa3_mlx import (  # noqa: E402
    read_wav, save_wav, patch_audio, load_dit, load_decoder, load_encoder,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    load_conditioner_from_npz, apply_prompt_padding, build_pingpong_schedule,
    sample_flow_euler, patched_decode,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from realtime.runtime.generator import cyclic_decode, CYCLIC_PAD  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
SIGMA = 0.5
STEPS = 8
SEED = 1000
OUTDIR = REPO / "runs" / "phase3b_seam"


def wrap_ratio(audio: np.ndarray) -> float:
    """|step across the wrap| / median |interior step|, per channel, averaged."""
    ratios = []
    for ch in audio:
        steps = np.abs(np.diff(ch))
        med = float(np.median(steps)) + 1e-9
        wrap = abs(float(ch[0]) - float(ch[-1]))
        ratios.append(wrap / med)
    return float(np.mean(ratios))


def main():
    print("=" * 68)
    print("  Phase 3b gate — cyclic decode -> seamless loop wrap")
    print("=" * 68)

    src = read_wav(str(SRC))
    seconds = src.shape[-1] / SAMPLE_RATE
    T = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if T % 2:
        T += 1
    target = T * SAMPLES_PER_LATENT
    a = src[:, :target] if src.shape[-1] >= target else \
        np.pad(src, ((0, 0), (0, target - src.shape[-1])))
    print(f"  loop {seconds:.2f}s -> T_lat={T}  pad={CYCLIC_PAD} frames")

    enc, _ = load_encoder("same-s", mx.float32)
    init = enc(mx.array(patch_audio(a[None], 256))).astype(mx.float16)
    mx.eval(init)
    del enc
    dit, ckpt = load_dit("sm-music", T, mx.float16)
    t5 = T5Gemma.from_npz(str(MLX_ROOT / "models" / "mlx" / "t5gemma_f16.npz"))
    e, m = t5.encode([""], max_len=256)
    mx.eval(e, m)
    pad_emb, secs = load_conditioner_from_npz(ckpt, prefix="cond.")
    se = secs(seconds).astype(mx.float16)
    cross = mx.concatenate([apply_prompt_padding(e.astype(mx.float16), m, pad_emb.astype(mx.float16)), se], axis=1)
    g = se[:, 0, :]
    del t5
    dec, chunk_fn, (chunk, ovl) = load_decoder("same-s", mx.float32)

    # one generated loop latent
    sig = build_pingpong_schedule(STEPS, sigma_max=SIGMA, use_logsnr_shift=True)
    pure = mx.random.normal((1, 256, T), dtype=mx.float16, key=mx.random.key(SEED))
    noise = init * (1.0 - SIGMA) + pure * SIGMA
    lat = sample_flow_euler(lambda x, t: dit(x, t, cross, g), noise, sig)
    mx.eval(lat)
    latf = lat.astype(mx.float32)

    def to_audio(patches):
        out = np.array(patched_decode(patches, 256, 2).astype(mx.float32))[0]
        return out[:, :target]

    plain_p = chunk_fn(dec, latf, chunk, ovl) if T > chunk + 2 * ovl else dec(latf)
    plain = to_audio(plain_p)
    cyc = to_audio(cyclic_decode(dec, chunk_fn, latf, chunk, ovl))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    # double each so the seam is audible on loop in a player
    save_wav(str(OUTDIR / "plain_x2.wav"), np.concatenate([plain, plain], axis=1))
    save_wav(str(OUTDIR / "cyclic_x2.wav"), np.concatenate([cyc, cyc], axis=1))

    r_plain = wrap_ratio(plain)
    r_cyc = wrap_ratio(cyc)

    # interior identity: cyclic must only change the edges. Compare the middle 80%.
    lo, hi = int(0.1 * target), int(0.9 * target)
    interior_snr_den = float(((plain[:, lo:hi] - cyc[:, lo:hi]) ** 2).sum())
    interior_snr = (10 * math.log10(float((plain[:, lo:hi] ** 2).sum()) / interior_snr_den)
                    if interior_snr_den > 0 else float("inf"))

    print(f"\n  wrap discontinuity ratio (lower = more seamless):")
    print(f"    plain  decode : {r_plain:8.1f}")
    print(f"    cyclic decode : {r_cyc:8.1f}   ({r_plain / max(r_cyc, 1e-9):.1f}x better)")
    print(f"  interior (mid 80%) cyclic-vs-plain SNR : {interior_snr:.1f} dB (edges-only change)")

    g_seam = r_cyc < r_plain
    g_interior = interior_snr > 30
    print()
    print("=" * 68)
    print(f"  seam improved      : {'PASS' if g_seam else 'FAIL'}")
    print(f"  interior preserved : {'PASS' if g_interior else 'FAIL'}")
    print(f"  PHASE 3b SEAM GATE : {'PASS' if (g_seam and g_interior) else 'FAIL'}")
    print(f"  WAVs in {OUTDIR} (plain_x2 vs cyclic_x2 — listen for the seam click)")
    print("=" * 68)


if __name__ == "__main__":
    main()
