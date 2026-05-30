"""Phase 0 feasibility spike — single-forward "tick" latency for the SA3
sm-music DiT, the number that gates the whole real-time streaming project.

Measures the cost of ONE batched DiT forward pass (the ring-buffer tick) across
loop length x batch depth x {eager, mx.compile}, plus SAME-S decode latency,
then derives the DEMON-style operating points (tick ms, gens/sec at S=8/S=4,
per-step vs per-request onset).

No T5Gemma / no real conditioning: forward cost is shape-driven, so dummy
tensors isolate the pure tick. One loaded model is timed at every sequence
length by always passing an explicit local_add_cond zeros tensor (bypassing the
model's T_lat-bound internal buffer).

Run:  optimized/mlx/.venv/bin/python realtime/bench/phase0_tick.py
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

MLX_ROOT = Path("/Users/max/Code/stable-audio-3/optimized/mlx")
sys.path.insert(0, str(MLX_ROOT))

from models.defs.dit_mlx import load_dit                              # noqa: E402
from models.defs.same_s_decoder import load_model as load_same_s, decode_chunked  # noqa: E402

DIT_NPZ = MLX_ROOT / "models" / "mlx" / "dit_sm-music_f16.npz"
SAMES_NPZ = MLX_ROOT / "models" / "mlx" / "same_s_decoder_f32.npz"

SR = 44100
SPL = 4096                      # audio samples per latent frame
LAT_HZ = SR / SPL              # ~10.77 latent frames / sec

SECONDS = [2.0, 4.0, 8.0, 12.0]
BATCHES = [1, 2, 4]            # ring-buffer depth proxy
DTYPE = mx.float16


def secs_to_T(s: float) -> int:
    T = max(1, math.ceil(s * SR / SPL))
    if T % 2:                  # SAME-S needs even T
        T += 1
    return T


def bench(fn, iters: int = 12, warmup: int = 4) -> tuple[float, float]:
    """Return (median_ms, min_ms). mx.eval forces materialization (MLX is lazy)."""
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def make_inputs(B: int, T: int):
    x = mx.random.normal((B, 256, T)).astype(DTYPE)
    t = mx.full((B,), 0.5, dtype=DTYPE)
    c = mx.random.normal((B, 257, 768)).astype(DTYPE)
    g = mx.random.normal((B, 768)).astype(DTYPE)
    l = mx.zeros((B, T, 257), dtype=DTYPE)
    mx.eval(x, t, c, g, l)
    return x, t, c, g, l


def main():
    print("=" * 72)
    print("  Phase 0 — SA3 sm-music streaming tick latency (Apple Silicon / MLX)")
    print("=" * 72)
    try:
        import mlx
        ver = getattr(mlx, "__version__", "?")
    except Exception:
        ver = "?"
    print(f"  device: {mx.default_device()}   mlx: {ver}   dtype: float16")
    print(f"  latent rate: {LAT_HZ:.2f} frames/s   ({SPL} samples/frame @ {SR} Hz)")
    print()

    t0 = time.perf_counter()
    dit = load_dit(str(DIT_NPZ), T_lat=64, dtype=DTYPE, compile_=False)
    mx.eval(dit.parameters())
    print(f"  loaded sm-music DiT (fp16) in {time.perf_counter() - t0:.1f}s")

    def fwd_fn(x, t, c, g, l):
        return dit(x, t, c, g, local_add_cond=l)

    compiled = mx.compile(fwd_fn)

    Ts = [secs_to_T(s) for s in SECONDS]
    eager: dict[tuple[int, int], float] = {}
    comp: dict[tuple[int, int], float] = {}

    print()
    print("  Measuring DiT forward (one tick) ...")
    for s, T in zip(SECONDS, Ts):
        for B in BATCHES:
            x, t, c, g, l = make_inputs(B, T)
            med, mn = bench(lambda: fwd_fn(x, t, c, g, l))
            eager[(T, B)] = med
            tag = f"T={T:<4}({s:>4.1f}s) B={B}"
            line = f"    {tag}  eager {med:6.1f} ms (min {mn:5.1f})"
            try:
                cmed, cmn = bench(lambda: compiled(x, t, c, g, l))
                comp[(T, B)] = cmed
                line += f"   compile {cmed:6.1f} ms (min {cmn:5.1f})   {med / cmed:4.2f}x"
            except Exception as e:
                line += f"   compile FAILED ({type(e).__name__})"
            print(line)

    def table(title, data):
        print()
        print(f"  --- {title} (median ms) ---")
        hdr = "  T \\ B    " + "".join(f"{'B='+str(B):>10}" for B in BATCHES)
        print(hdr)
        for s, T in zip(SECONDS, Ts):
            row = f"  {T:>4}({s:>4.1f}s)"
            for B in BATCHES:
                v = data.get((T, B))
                row += f"{(f'{v:.1f}' if v else '-'):>10}"
            print(row)

    table("Eager forward", eager)
    if comp:
        table("Compiled forward", comp)

    # Batching sub-linearity (key: does higher depth pay off like CUDA's ~6.1x@B8?)
    best = comp if comp else eager
    print()
    print("  --- Batching curve: cost(B) / cost(B=1)  [compiled if available] ---")
    for s, T in zip(SECONDS, Ts):
        b1 = best.get((T, 1))
        if not b1:
            continue
        ratios = "  ".join(
            f"B{B}:{best.get((T, B)) / b1:4.2f}x" for B in BATCHES if best.get((T, B))
        )
        print(f"    T={T:<4}({s:>4.1f}s)  {ratios}")

    # ---- SAME-S decode ----
    print()
    print("  Measuring SAME-S decode (fp32) ...")
    t0 = time.perf_counter()
    dec = load_same_s(weights_path=str(SAMES_NPZ), dtype=mx.float32)
    mx.eval(dec.parameters())
    print(f"    loaded SAME-S (fp32) in {time.perf_counter() - t0:.1f}s")
    win_T = secs_to_T(3.0)     # ~3s playback window
    for s, T in zip(SECONDS, Ts):
        lat = mx.random.normal((1, 256, T)).astype(mx.float32)
        mx.eval(lat)
        full_med, _ = bench(lambda: decode_chunked(dec, lat, 8, 2), iters=8, warmup=2)
        line = f"    T={T:<4}({s:>4.1f}s)  full-loop decode {full_med:6.1f} ms"
        if T >= win_T:
            win = lat[:, :, :win_T]
            mx.eval(win)
            win_med, _ = bench(lambda: dec(win), iters=8, warmup=2)
            line += f"   |  {win_T}-frame (~3s) window {win_med:6.1f} ms"
        print(line)

    # ---- Derived operating points ----
    print()
    print("  --- Derived ring-buffer operating points (cfg=1, compiled tick) ---")
    print("  gens/sec = (1000/tick) * (depth/steps);  per-step onset = 1 tick;")
    print("  per-request onset ~= steps ticks (new slot must run its full schedule)")
    print()
    print(f"  {'config':<18}{'tick ms':>9}{'g/s S=8':>9}{'g/s S=4':>9}"
          f"{'step onset':>12}{'req onset(S8)':>15}")
    for s, T in zip(SECONDS, Ts):
        for B in BATCHES:
            tick = best.get((T, B))
            if not tick:
                continue
            gps8 = (1000.0 / tick) * (B / 8.0)
            gps4 = (1000.0 / tick) * (B / 4.0)
            cfg = f"{s:.0f}s loop, D={B}"
            print(f"  {cfg:<18}{tick:>9.1f}{gps8:>9.2f}{gps4:>9.2f}"
                  f"{tick:>10.0f}ms{tick * 8:>13.0f}ms")

    print()
    print("=" * 72)
    print("  Read: a 'tick' is one denoise step across all in-flight slots. The")
    print("  per-step shared-mutable curves (SDE source-blend, velocity) onset in")
    print("  ONE tick at any depth; denoise/prompt changes take ~S ticks.")
    print("=" * 72)


if __name__ == "__main__":
    main()
