"""Phase 0 — sampler A/B for the streaming engine's per-slot step.

The ring buffer advances each slot one step per tick. DEMON uses a deterministic
ODE/SDE step; SA3's shipped sampler is stochastic pingpong (fresh noise every
step). This generates the SAME a2a variation of one loop under four regimes so
we can listen and pick the engine's default step:

    pingpong-8 (current product baseline) | pingpong-4 | euler-8 | euler-4

All four share the identical init noise (same seed + same a2a mix), so the only
variable is sampler x step-count. fewer steps ~halves per-request onset and
doubles throughput, IF quality holds.

Run: optimized/mlx/.venv/bin/python realtime/bench/phase0_euler_ab.py [SRC.wav]
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

REPO = Path("/Users/max/Code/stable-audio-3")
MLX_ROOT = REPO / "optimized" / "mlx"
sys.path.insert(0, str(MLX_ROOT))
sys.path.insert(0, str(MLX_ROOT / "scripts"))

from sa3_mlx import (  # noqa: E402  (reuse the product's battle-tested helpers)
    read_wav, save_wav, patch_audio, load_dit, load_decoder, load_encoder,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    load_conditioner_from_npz, apply_prompt_padding, build_pingpong_schedule,
    sample_flow_pingpong, sample_flow_euler, patched_decode,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
PROMPT = ""            # free-variation regime (empty prompt, the morph default)
SIGMA = 0.5            # a2a noise level (free preset ~0.45-0.52)
SEED = 1528
OUTDIR = REPO / "runs" / "phase0_euler_ab"

CONFIGS = [("pingpong", 8), ("pingpong", 4), ("euler", 8), ("euler", 4)]


def to_mono(a: np.ndarray) -> np.ndarray:
    return a.mean(0) if a.ndim == 2 else a


def mean_mag_spec(x: np.ndarray, n: int = 2048, hop: int = 512) -> np.ndarray:
    w = np.hanning(n)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    mags = [np.abs(np.fft.rfft(x[i:i + n] * w)) for i in range(0, len(x) - n + 1, hop)]
    return np.mean(mags, axis=0) if mags else np.abs(np.fft.rfft(x[:n] * w))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print("=" * 72)
    print("  Phase 0 — Euler vs pingpong sampler A/B  (SA3 sm-music, a2a)")
    print("=" * 72)
    print(f"  source : {SRC}")
    print(f"  prompt : {PROMPT!r}   sigma_max={SIGMA}   seed={SEED}")

    src = read_wav(str(SRC))                              # (2, T_audio)
    seconds = src.shape[-1] / SAMPLE_RATE
    T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if T_lat % 2:
        T_lat += 1
    target = T_lat * SAMPLES_PER_LATENT
    if src.shape[-1] >= target:
        a = src[:, :target]
    else:
        a = np.pad(src, ((0, 0), (0, target - src.shape[-1])))
    print(f"  loop   : {seconds:.2f}s  ->  T_lat={T_lat}  ({target / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz)")
    req = int(round(seconds * SAMPLE_RATE))

    # ---- one-time loads ----
    t0 = time.perf_counter()
    enc_model, _ = load_encoder("same-s", mx.float32)
    init_latents = enc_model(mx.array(patch_audio(a[None], 256)))
    mx.eval(init_latents)
    init_latents = init_latents.astype(mx.float16)
    del enc_model

    dit, ckpt = load_dit("sm-music", T_lat, mx.float16)
    t5 = T5Gemma.from_npz(str(MLX_ROOT / "models" / "mlx" / "t5gemma_f16.npz"))
    embeds, mask = t5.encode([PROMPT], max_len=256)
    mx.eval(embeds, mask)
    pad_emb, secs = load_conditioner_from_npz(ckpt, prefix="cond.")
    embeds = embeds.astype(mx.float16)
    embeds_p = apply_prompt_padding(embeds, mask, pad_emb.astype(mx.float16))
    secs_e = secs(seconds).astype(mx.float16)                      # (1,1,768)
    cross = mx.concatenate([embeds_p, secs_e], axis=1)            # (1,257,768)
    gcond = secs_e[:, 0, :]                                        # (1,768)
    del t5

    dec, chunk_fn, (chunk, ovl) = load_decoder("same-s", mx.float32)
    print(f"  loaded encoder+DiT+T5Gemma+decoder in {time.perf_counter() - t0:.1f}s")

    def model_fn(x, t):
        return dit(x, t, cross, gcond)

    def decode(lat):
        latf = lat.astype(mx.float32)
        kernel = chunk + 2 * ovl
        patches = chunk_fn(dec, latf, chunk, ovl) if T_lat > kernel else dec(latf)
        audio = patched_decode(patches, 256, 2)
        mx.eval(audio)
        out = np.array(audio.astype(mx.float32))[0]
        return out[..., :req] if out.shape[-1] > req else out

    # Identical init for every config: same seed -> same pure noise -> same a2a mix.
    key = mx.random.key(SEED)
    pure = mx.random.normal((1, 256, T_lat), dtype=mx.float16, key=key)
    noise = init_latents * (1.0 - SIGMA) + pure * SIGMA
    mx.eval(noise)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    src_trim = a[:, :req]
    save_wav(str(OUTDIR / "0_source.wav"), src_trim)
    src_spec = mean_mag_spec(to_mono(src_trim))

    outputs: dict[str, np.ndarray] = {}
    rows = []
    print()
    print("  Generating ...")
    for i, (name, steps) in enumerate(CONFIGS, start=1):
        sampler = sample_flow_pingpong if name == "pingpong" else sample_flow_euler
        sigmas = build_pingpong_schedule(steps, sigma_max=SIGMA, use_logsnr_shift=True)
        t0 = time.perf_counter()
        lat = sampler(model_fn, noise, sigmas, seed=SEED + 1)
        mx.eval(lat)
        sample_ms = (time.perf_counter() - t0) * 1000
        audio = decode(lat)
        tag = f"{name}{steps}"
        fname = f"{i}_{tag}.wav"
        save_wav(str(OUTDIR / fname), audio)
        outputs[tag] = audio
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt((audio ** 2).mean()))
        finite = bool(np.isfinite(audio).all())
        cos_src = cos(mean_mag_spec(to_mono(audio)), src_spec)
        rows.append([tag, sample_ms, sample_ms / steps, peak, rms, cos_src, finite, fname])
        print(f"    {tag:<11} sample {sample_ms:6.0f} ms ({sample_ms / steps:5.0f}/step)  "
              f"peak {peak:.3f} rms {rms:.3f} {'' if finite else 'NON-FINITE!'}")

    base = outputs["pingpong8"]
    base_spec = mean_mag_spec(to_mono(base))

    print()
    print(f"  {'config':<11}{'ms':>7}{'ms/step':>9}{'peak':>7}{'rms':>7}"
          f"{'cos→src':>9}{'cos→pp8':>9}  file")
    for tag, ms, mps, peak, rms, cos_src, finite, fname in rows:
        cos_pp8 = cos(mean_mag_spec(to_mono(outputs[tag])), base_spec)
        print(f"  {tag:<11}{ms:>7.0f}{mps:>9.0f}{peak:>7.3f}{rms:>7.3f}"
              f"{cos_src:>9.3f}{cos_pp8:>9.3f}  {fname}")

    print()
    print(f"  WAVs in: {OUTDIR}")
    print("  Listen 0_source -> 1_pingpong8 (baseline) -> 2/3/4. cos->src = how much")
    print("  timbre is preserved; cos->pp8 = how far each sampler departs from the")
    print("  current product default. The ear is the judge; these are triage only.")


if __name__ == "__main__":
    main()
