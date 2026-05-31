"""Phase 3b gate — loop-crossfade fold evens the loop-point energy.

The real live "gap" is NOT a click (cyclic decode already makes the wrap
continuous) — it's SA3's intro/outro: the clip is loud at the start and decays
toward the end, so looping it pumps loud->quiet->loud. make_loopable folds the
faded tail back over the faded head, equalizing the level at the loop point.

We pull a SETTLED live ring buffer (what the ear actually hears) with the fold
OFF vs ON and measure loop-point energy evenness:

  edge_ratio = rms(last 0.5s) / rms(first 0.5s)   (1.0 = perfectly even)

Fold ON should push edge_ratio much closer to 1.0 than fold OFF, while keeping
the wrap sample-continuous and the audio finite.

Run: optimized/mlx/.venv/bin/python realtime/bench/phase3b_loopfold.py
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
from realtime.runtime.generator import GeneratorThread, GenConfig  # noqa: E402
from realtime.runtime.live_morph import loop_length_samples  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
OUTDIR = REPO / "runs" / "phase3b_loopfold"


def edge_metrics(buf):
    mono = buf.mean(0)
    L = mono.shape[-1]
    w = int(0.5 * SAMPLE_RATE)
    head = float(np.sqrt((mono[:w] ** 2).mean()))
    tail = float(np.sqrt((mono[-w:] ** 2).mean()))
    edge_ratio = tail / (head + 1e-9)
    steps = np.abs(np.diff(mono))
    wrap = abs(float(mono[0]) - float(mono[-1])) / (float(np.median(steps)) + 1e-9)
    finite = bool(np.isfinite(buf).all())
    return edge_ratio, wrap, finite, L


def settled(loop_xfade_s):
    src = read_wav(str(SRC))
    seconds = src.shape[-1] / SAMPLE_RATE
    cfg = GenConfig(seconds=seconds, steps=8, depth=2, denoise=0.5,
                    loop_xfade_s=loop_xfade_s)
    L = loop_length_samples(seconds)
    ring = AudioRing(length=L, channels=2, xfade=2048,
                     loop_xfade=int(loop_xfade_s * SAMPLE_RATE))
    gen = GeneratorThread(ring, src, cfg)
    gen.start()
    assert gen.wait_ready(timeout=120)
    time.sleep(2.5)
    buf = ring._buf.copy()
    gen.stop(); gen.join(timeout=5)
    return buf


def main():
    print("=" * 68)
    print("  Phase 3b gate — loop-crossfade fold evens loop-point energy")
    print("=" * 68)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    off = settled(0.0)
    on = settled(0.75)
    r_off, w_off, f_off, L_off = edge_metrics(off)
    r_on, w_on, f_on, L_on = edge_metrics(on)
    save_wav(str(OUTDIR / "fold_off_x2.wav"), np.concatenate([off, off], axis=1))
    save_wav(str(OUTDIR / "fold_on_x2.wav"), np.concatenate([on, on], axis=1))

    print(f"\n  edge_ratio = rms(last 0.5s)/rms(first 0.5s), 1.0 = even loop point")
    print(f"    fold OFF : edge_ratio={r_off:.3f}  wrap={w_off:.1f}  "
          f"finite={f_off}  L={L_off} ({L_off/SAMPLE_RATE:.2f}s)")
    print(f"    fold ON  : edge_ratio={r_on:.3f}  wrap={w_on:.1f}  "
          f"finite={f_on}  L={L_on} ({L_on/SAMPLE_RATE:.2f}s)")

    # ON should be closer to 1.0 (even) than OFF
    g_even = abs(r_on - 1.0) < abs(r_off - 1.0)
    g_cont = w_on < 5.0
    g_fin = f_on
    print()
    print("=" * 68)
    print(f"  loop-point more even : {'PASS' if g_even else 'FAIL'}")
    print(f"  wrap continuous      : {'PASS' if g_cont else 'FAIL'}")
    print(f"  finite               : {'PASS' if g_fin else 'FAIL'}")
    print(f"  PHASE 3b LOOPFOLD GATE: {'PASS' if (g_even and g_cont and g_fin) else 'FAIL'}")
    print(f"  WAVs in {OUTDIR} (fold_off_x2 vs fold_on_x2 — listen at the loop point)")
    print("=" * 68)


if __name__ == "__main__":
    main()
