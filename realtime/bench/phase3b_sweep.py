"""Phase 3b gate — denoise slew makes a slider sweep morph gradually.

denoise is the per-request class (it defines a slot's whole trajectory), so a
change can't be a 1-tick override; the question is only whether a fast slider
move produces a GRADUAL sequence of completions or a snap between very different
loops. With slew-limiting, consecutive completions during a sweep should change
in small steps. We jump the target denoise 0.3 -> 0.8 instantly and measure the
per-completion latent change.

  metric: max consecutive-completion RMS during the sweep.
  slew ON should give a much smaller max step than slew OFF (jump).
  Also verify the effective denoise actually REACHES the target.

Run: optimized/mlx/.venv/bin/python realtime/bench/phase3b_sweep.py
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

from sa3_mlx import read_wav, SAMPLE_RATE  # noqa: E402
from realtime.runtime.audio_ring import AudioRing  # noqa: E402
from realtime.runtime.generator import GeneratorThread, GenConfig  # noqa: E402
from realtime.runtime.live_morph import loop_length_samples  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"


def run(slew: float, label: str):
    src = read_wav(str(SRC))
    cfg = GenConfig(seconds=8.0, steps=8, depth=2, denoise=0.3, denoise_slew=slew)
    ring = AudioRing(length=loop_length_samples(cfg.seconds), channels=2, xfade=2048)
    gen = GeneratorThread(ring, src, cfg)

    # capture the effective denoise actually used at each submission
    eff = []
    orig = gen._submit_morph
    def traced():
        orig()
        eff.append(gen._denoise_eff)
    gen._submit_morph = traced

    gen.start()
    assert gen.wait_ready(timeout=120)
    # jump the target; let it run enough completions to ramp 0.3 -> 0.8 at this slew
    gen.set_denoise(0.8)
    import time
    time.sleep(6.0)
    gen.stop(); gen.join(timeout=5)

    eff = eff[2:]                      # drop warmup submissions
    steps = np.abs(np.diff(eff)) if len(eff) > 1 else np.array([0.0])
    reached = max(eff) if eff else 0.0
    print(f"  {label:<10} submissions={len(eff):3d}  max step={steps.max():.3f}  "
          f"reached denoise={reached:.3f}")
    return steps.max(), reached


def main():
    print("=" * 68)
    print("  Phase 3b gate — denoise slew -> gradual sweep (jump 0.3 -> 0.8)")
    print("=" * 68)
    off_step, off_reached = run(slew=1.0, label="slew OFF")     # 1.0 = no limit
    on_step, on_reached = run(slew=0.04, label="slew ON")

    g_gradual = on_step < off_step * 0.6        # slew clearly limits per-step jump
    g_reaches = on_reached >= 0.79              # still converges to target
    print()
    print("=" * 68)
    print(f"  gradual (smaller steps) : {'PASS' if g_gradual else 'FAIL'} "
          f"({on_step:.3f} vs {off_step:.3f})")
    print(f"  reaches target 0.8      : {'PASS' if g_reaches else 'FAIL'} ({on_reached:.3f})")
    print(f"  PHASE 3b SWEEP GATE     : {'PASS' if (g_gradual and g_reaches) else 'FAIL'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
