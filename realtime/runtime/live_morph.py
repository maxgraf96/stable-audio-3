"""LiveMorph — the living-loop morph instrument (Phase 3a).

Drop a loop, it plays continuously while fresh morphs fade in. The generator
thread owns MLX and fills an AudioRing; a PortAudio output stream plays the ring
on its own real-time thread. Live controls (denoise, SDE source-blend, velocity,
prompt) mutate shared state and take effect on the next tick.

CLI:
  # live playback (default 8s loop, depth 2)
  optimized/mlx/.venv/bin/python -m realtime.runtime.live_morph SRC.wav

  # headless self-test: no audio device, verify the ring fills + telemetry
  optimized/mlx/.venv/bin/python -m realtime.runtime.live_morph SRC.wav --selftest

Interactive (live mode), single keys + Enter:
  d <0..1>   denoise = variation amount (how far the loop departs from the source)
  s <0..1>   SDE source-blend: 1=explore/regenerate, 0=anchor to source (flat curve)
  v <x>      velocity_scale: per-step transformation RATE. 1.0=trained; >1 lunges
             harder toward the generated result (more aggressive, unstable past ~1.3);
             <1 stays softer/closer. A texture knob, not a headline control.
  e <0|1>    evolve: 0 (default) = FIXED seed, your controls morph ONE coherent loop;
             1 = roll a new random variation each loop ("browse stations")
  p <text>   set prompt (Enter alone after 'p' = empty/unconditional)
  q          quit
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MLX_ROOT = REPO / "optimized" / "mlx"
for p in (REPO / "realtime", MLX_ROOT, MLX_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sa3_mlx import read_wav, SAMPLE_RATE, SAMPLES_PER_LATENT  # noqa: E402
from realtime.runtime.audio_ring import AudioRing  # noqa: E402
from realtime.runtime.generator import GeneratorThread, GenConfig, plan_lengths  # noqa: E402

import math  # noqa: E402


def loop_length_samples(seconds: float) -> int:
    """Playable interior-loop length in samples (the generator crops to this)."""
    _, _, loop_T = plan_lengths(seconds, 0.0)
    return loop_T * SAMPLES_PER_LATENT


def build(src_path: str, cfg: GenConfig):
    src = read_wav(src_path)                       # [2, T_audio]
    # The generator decodes gen = loop + 2*margin and crops to the interior loop,
    # so the AudioRing holds exactly that interior loop. The tail->head fold
    # (loop_xfade) smooths the loop point; with margin=0 (default) the generator's
    # cyclic-wrap decode keeps the seam continuous and the fold evens the level.
    _, _, loop_T = plan_lengths(cfg.seconds, cfg.margin_s)
    L = loop_T * SAMPLES_PER_LATENT
    loop_xf = int(cfg.loop_xfade_s * SAMPLE_RATE)
    ring = AudioRing(length=L, channels=2, xfade=2048, loop_xfade=loop_xf)
    gen = GeneratorThread(ring, src, cfg)
    return ring, gen, ring.L


def run_selftest(src_path: str, cfg: GenConfig, seconds_run: float = 12.0):
    print("=" * 68)
    print("  LiveMorph self-test (headless, no audio device)")
    print("=" * 68)
    ring, gen, L = build(src_path, cfg)
    print(f"  loop={cfg.seconds}s ({L} samples)  depth={cfg.depth} steps={cfg.steps} "
          f"denoise={cfg.denoise}  dit={'sm-music'}")
    t0 = time.perf_counter()
    gen.start()
    if not gen.wait_ready(timeout=120):
        print("  FAIL: generator did not become ready")
        return False
    print(f"  warmup ready in {time.perf_counter() - t0:.1f}s — first loop audible")

    # Simulate an audio device pulling blocks at real time (1024 frames @ 44.1k).
    block = 1024
    period = block / SAMPLE_RATE
    nblocks = int(seconds_run / period)
    underruns = 0
    peak = 0.0
    finite = True
    # exercise the live controls mid-run
    events = {int(nblocks * 0.25): ("sde", 0.2),
              int(nblocks * 0.5): ("denoise", 0.7),
              int(nblocks * 0.75): ("sde", 1.0)}
    next_pull = time.perf_counter()
    for i in range(nblocks):
        next_pull += period
        buf = ring.read(block)                     # what the callback would get
        if not np.isfinite(buf).all():
            finite = False
        peak = max(peak, float(np.abs(buf).max()))
        if not ring.has_audio:
            underruns += 1
        if i in events:
            kind, val = events[i]
            if kind == "denoise":
                gen.set_denoise(val)
            else:
                gen.set_shared_curve("sde_denoise_curve", val)
        sleep = next_pull - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

    st = gen.stats()
    gen.stop()
    gen.join(timeout=5)
    print(f"  ran {seconds_run:.0f}s of playback ({nblocks} blocks @ {block})")
    print(f"  ticks={st['ticks']} completions={st['completions']} "
          f"decodes={st['decodes']} skips={st['skips']}")
    print(f"  tick {st['tick_ms']:.1f} ms   decode {st['decode_ms']:.1f} ms")
    print(f"  audio: peak={peak:.3f}  finite={finite}  underruns(after warmup)={underruns}")
    gen_rate = st['completions'] / seconds_run
    play_rate = 1.0 / cfg.seconds
    print(f"  morph rate {gen_rate:.2f} loops/s vs playback {play_rate:.2f} loops/s "
          f"({gen_rate / play_rate:.0f}x headroom)")
    ok = finite and underruns == 0 and peak > 1e-3 and st['completions'] >= 2
    print("=" * 68)
    print(f"  SELF-TEST: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    return ok


def run_live(src_path: str, cfg: GenConfig):
    import sounddevice as sd
    ring, gen, L = build(src_path, cfg)
    print(f"loading models + warming up ({cfg.seconds}s loop, depth {cfg.depth}) ...")
    gen.start()
    if not gen.wait_ready(timeout=120):
        print("generator failed to start"); return
    print("ready — playing. controls: d=denoise s=source-blend v=velocity "
          "e=evolve(0/1) p=prompt, q=quit")

    def callback(outdata, frames, time_info, status):
        buf = ring.read(frames)                    # [2, frames]
        outdata[:] = buf.T                          # PortAudio wants [frames, ch]

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2,
                             blocksize=1024, dtype="float32", callback=callback)
    stop = threading.Event()

    def telemetry():
        while not stop.wait(2.0):
            s = gen.stats()
            print(f"  [tick {s['tick_ms']:.0f}ms decode {s['decode_ms']:.0f}ms  "
                  f"gens={s['completions']} skips={s['skips']}]")
    threading.Thread(target=telemetry, daemon=True).start()

    with stream:
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                cmd, _, arg = line.partition(" ")
                arg = arg.strip()
                if cmd == "q":
                    break
                elif cmd == "d":
                    gen.set_denoise(float(arg)); print(f"  denoise -> {arg}")
                elif cmd == "s":
                    gen.set_shared_curve("sde_denoise_curve", float(arg)); print(f"  sde -> {arg}")
                elif cmd == "v":
                    gen.set_shared_curve("velocity_scale", float(arg)); print(f"  vel -> {arg}")
                elif cmd == "e":
                    on = arg not in ("0", "off", "false", "")
                    gen.set_evolve(on); print(f"  evolve -> {'on (browse)' if on else 'off (fixed seed)'}")
                elif cmd == "p":
                    gen.set_prompt(arg); print(f"  prompt -> {arg!r}")
                else:
                    print("  ? d/s/v/e/p <val>, q")
        except KeyboardInterrupt:
            pass
    stop.set()
    gen.stop()
    gen.join(timeout=5)
    print("stopped.")


def main():
    ap = argparse.ArgumentParser(description="SA3 living-loop morph instrument (Phase 3a)")
    ap.add_argument("src", help="source loop WAV (44.1kHz)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="playable loop length (DEFAULT = source length). SA3 "
                         "actually generates this PLUS 2x --margin and only the "
                         "even interior is played, so the intro/outro envelope is "
                         "cropped out.")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="interior-windowing margin (s) per side, cropped away. "
                         "Default 0 (off) -> cyclic-wrap decode. Measured "
                         "INEFFECTIVE on a decaying source (the loud->quiet shape "
                         "is the source loop's own envelope, not SA3 intro/outro), "
                         "so only useful for genuinely full-energy source loops.")
    ap.add_argument("--loop-xfade", type=float, default=0.15,
                    help="small tail->head fold (s) for the interior seam "
                         "(default 0.15; interior windowing does the heavy lifting).")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--denoise", type=float, default=0.5)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--evolve", action="store_true",
                    help="start in evolve/browse mode (new random variation each loop); "
                         "default is fixed-seed coherent morph")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-seconds", type=float, default=12.0)
    a = ap.parse_args()
    seconds = a.seconds
    if seconds is None:                       # default: match the source length
        seconds = read_wav(a.src).shape[-1] / SAMPLE_RATE
        print(f"loop length = source length ({seconds:.2f}s)")
    cfg = GenConfig(seconds=seconds, steps=a.steps, depth=a.depth,
                    denoise=a.denoise, prompt=a.prompt, evolve=a.evolve,
                    margin_s=a.margin, loop_xfade_s=a.loop_xfade)
    if a.selftest:
        ok = run_selftest(a.src, cfg, a.selftest_seconds)
        sys.exit(0 if ok else 1)
    run_live(a.src, cfg)


if __name__ == "__main__":
    main()
