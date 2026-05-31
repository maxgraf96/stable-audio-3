"""Phase 3b gate — interior windowing evens the loop-point energy.

The live "gap" is SA3's intro/outro envelope (the clip is loud at the start and
decays toward the end), not a click. The proper fix is interior windowing:
GENERATE loop + 2*margin and play only the even interior body, so the intro lands
in the leading margin and the outro in the trailing margin, both cropped away.

We pull a SETTLED live ring buffer (what the ear hears) at three margin settings
and measure loop-point energy evenness + wrap continuity:

  edge_ratio = rms(last 0.5s) / rms(first 0.5s)   (1.0 = even loop point)
  wrap       = |x[0]-x[L-1]| / median|step|       (low = continuous)

margin=0 (no windowing) should be uneven (edge_ratio far from 1.0, the bug).
margin>0 should pull edge_ratio toward 1.0 while staying continuous + finite.

Run: optimized/mlx/.venv/bin/python realtime/bench/phase3b_interior.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/max/Code/stable-audio-3")
MLX_ROOT = REPO / "optimized" / "mlx"
for p in (REPO, REPO / "realtime", MLX_ROOT, MLX_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sa3_mlx import read_wav, save_wav, SAMPLE_RATE  # noqa: E402
from realtime.runtime.audio_ring import AudioRing  # noqa: E402
from realtime.runtime.generator import GeneratorThread, GenConfig, plan_lengths  # noqa: E402
from realtime.runtime.live_morph import build  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
OUTDIR = REPO / "runs" / "phase3b_interior"


def edge_metrics(buf):
    mono = buf.mean(0)
    w = int(0.5 * SAMPLE_RATE)
    head = float(np.sqrt((mono[:w] ** 2).mean()))
    tail = float(np.sqrt((mono[-w:] ** 2).mean()))
    edge_ratio = tail / (head + 1e-9)
    steps = np.abs(np.diff(mono))
    wrap = abs(float(mono[0]) - float(mono[-1])) / (float(np.median(steps)) + 1e-9)
    return edge_ratio, wrap, bool(np.isfinite(buf).all())


def settled(margin_s):
    src = read_wav(str(SRC))
    seconds = src.shape[-1] / SAMPLE_RATE
    cfg = GenConfig(seconds=seconds, steps=8, depth=2, denoise=0.5,
                    margin_s=margin_s, loop_xfade_s=0.15 if margin_s > 0 else 0.0)
    ring, gen, L = build(str(SRC), cfg)
    gen.start()
    assert gen.wait_ready(timeout=120)
    time.sleep(2.5)
    buf = ring._buf.copy()
    gen.stop(); gen.join(timeout=5)
    gen_T, margin_T, loop_T = plan_lengths(seconds, margin_s)
    return buf, (gen_T, margin_T, loop_T)


def main():
    print("=" * 70)
    print("  Phase 3b gate — interior windowing evens the loop point")
    print("=" * 70)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for margin in (0.0, 1.5, 3.0):
        buf, plan = settled(margin)
        er, wrap, fin = edge_metrics(buf)
        gen_T, margin_T, loop_T = plan
        rows.append((margin, er, wrap, fin, buf.shape[-1]))
        save_wav(str(OUTDIR / f"margin_{margin:.1f}_x2.wav"),
                 np.concatenate([buf, buf], axis=1))
        print(f"  margin={margin:>4.1f}s  gen_T={gen_T} margin_T={margin_T} loop_T={loop_T}"
              f"  ->  edge_ratio={er:.3f}  wrap={wrap:.1f}  finite={fin}  "
              f"L={buf.shape[-1]} ({buf.shape[-1]/SAMPLE_RATE:.2f}s)")

    base = next(r for r in rows if r[0] == 0.0)
    win = next(r for r in rows if r[0] == 1.5)
    # windowing must pull edge_ratio closer to 1.0 than no-windowing, stay
    # continuous, finite, and the default 1.5s should be reasonably even.
    g_better = abs(win[1] - 1.0) < abs(base[1] - 1.0)
    g_even = 0.6 <= win[1] <= 1.6
    g_cont = win[2] < 5.0
    g_fin = win[3]
    print()
    print("=" * 70)
    print(f"  windowing more even than none : {'PASS' if g_better else 'FAIL'} "
          f"(|{win[1]:.3f}-1| < |{base[1]:.3f}-1|)")
    print(f"  default 1.5s reasonably even  : {'PASS' if g_even else 'FAIL'} ({win[1]:.3f})")
    print(f"  wrap continuous + finite      : {'PASS' if (g_cont and g_fin) else 'FAIL'}")
    ok = g_better and g_even and g_cont and g_fin
    print(f"  PHASE 3b INTERIOR GATE        : {'PASS' if ok else 'FAIL'}")
    print(f"  WAVs in {OUTDIR} (margin_0.0 vs margin_1.5 vs margin_3.0, each x2)")
    print("=" * 70)


if __name__ == "__main__":
    main()
