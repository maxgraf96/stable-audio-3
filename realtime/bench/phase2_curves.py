"""Phase 2a gates — per-frame curves + shared-mutable overrides.

  (1) no-op regression : velocity_scale=1.0 is bit-exact to plain euler
  (2) hard anchor      : flat sde_denoise_curve=0.0 -> output == source latent
  (3) source gradient  : sde ramp 0->1 -> per-segment similarity-to-source descends
                         (the per-frame preservation gradient SA3 scalar a2a can't do)
  (4) shared-mutable   : set_shared_curve overrides a slot, beats the per-slot field,
                         reverts on None, and a mid-stream set anchors the remaining steps

Run: optimized/mlx/.venv/bin/python realtime/bench/phase2_curves.py
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

REPO = Path("/Users/max/Code/stable-audio-3")
MLX_ROOT = REPO / "optimized" / "mlx"
for p in (REPO / "realtime", MLX_ROOT, MLX_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sa3_mlx import (  # noqa: E402
    read_wav, save_wav, patch_audio, load_dit, load_decoder, load_encoder,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    load_conditioner_from_npz, apply_prompt_padding, patched_decode,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from engine.stream import StreamPipeline, SlotRequest  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
PROMPT = ""
SIGMA = 0.5
STEPS = 8
SEED = 1528
OUTDIR = REPO / "runs" / "phase2_curves"


def maxabs(a, b):
    return float(np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32))).max())


def to_mono(a):
    return a.mean(0) if a.ndim == 2 else a


def mean_mag_spec(x, n=2048, hop=512):
    w = np.hanning(n)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    mags = [np.abs(np.fft.rfft(x[i:i + n] * w)) for i in range(0, len(x) - n + 1, hop)]
    return np.mean(mags, axis=0) if mags else np.abs(np.fft.rfft(x[:n] * w))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def seg_cos_to_source(audio, src, n=4):
    L = min(audio.shape[-1], src.shape[-1])
    out = []
    for i in range(n):
        a0, a1 = i * L // n, (i + 1) * L // n
        out.append(cos(mean_mag_spec(to_mono(audio[..., a0:a1])),
                       mean_mag_spec(to_mono(src[..., a0:a1]))))
    return out


def main():
    print("=" * 72)
    print("  Phase 2a gates — per-frame curves + shared-mutable (SA3 sm-music)")
    print("=" * 72)

    src = read_wav(str(SRC))
    seconds = src.shape[-1] / SAMPLE_RATE
    T = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if T % 2:
        T += 1
    target = T * SAMPLES_PER_LATENT
    a = src[:, :target] if src.shape[-1] >= target else \
        np.pad(src, ((0, 0), (0, target - src.shape[-1])))
    req_samples = int(round(seconds * SAMPLE_RATE))
    print(f"  loop {seconds:.2f}s -> T_lat={T}  sigma={SIGMA} steps={STEPS} seed={SEED}")

    enc_model, _ = load_encoder("same-s", mx.float32)
    init_latents = enc_model(mx.array(patch_audio(a[None], 256))).astype(mx.float16)
    mx.eval(init_latents)
    del enc_model
    dit, ckpt = load_dit("sm-music", T, mx.float16)
    t5 = T5Gemma.from_npz(str(MLX_ROOT / "models" / "mlx" / "t5gemma_f16.npz"))
    embeds, mask = t5.encode([PROMPT], max_len=256)
    mx.eval(embeds, mask)
    pad_emb, secs = load_conditioner_from_npz(ckpt, prefix="cond.")
    embeds_p = apply_prompt_padding(embeds.astype(mx.float16), mask, pad_emb.astype(mx.float16))
    secs_e = secs(seconds).astype(mx.float16)
    cross = mx.concatenate([embeds_p, secs_e], axis=1)
    gcond = secs_e[:, 0, :]
    del t5
    dec, chunk_fn, (chunk, ovl) = load_decoder("same-s", mx.float32)

    def decode(lat):
        latf = lat.astype(mx.float32)
        patches = chunk_fn(dec, latf, chunk, ovl) if T > chunk + 2 * ovl else dec(latf)
        out = np.array(patched_decode(patches, 256, 2).astype(mx.float32))[0]
        return out[..., :req_samples] if out.shape[-1] > req_samples else out

    def run_slot(shared=None, **req_kwargs):
        pipe = StreamPipeline(dit, T, steps=STEPS, depth=1)
        if shared:
            for k, v in shared.items():
                pipe.set_shared_curve(k, v)
        pipe.submit(SlotRequest(cross_attn=cross, global_cond=gcond, seed=SEED,
                                denoise=SIGMA, source_latent=init_latents, **req_kwargs))
        for _ in range(STEPS + 4):
            r = pipe.tick()
            if r is not None:
                return r
        raise RuntimeError("no completion")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    src_recon = decode(init_latents)
    save_wav(str(OUTDIR / "0_source_recon.wav"), src_recon)

    euler = run_slot()                       # baseline: no curves -> euler

    # (1) no-op regression
    vs1 = run_slot(velocity_scale=1.0)
    d1 = maxabs(euler, vs1)
    g1 = d1 == 0.0
    print(f"\n  (1) velocity_scale=1.0 == plain euler : max|d|={d1:.1e}  {'PASS' if g1 else 'FAIL'}")

    # (2) hard anchor: flat sde=0 -> source latent
    sde0 = run_slot(sde_denoise_curve=0.0)
    d2 = maxabs(sde0, init_latents)
    g2 = d2 == 0.0
    aud2 = decode(sde0)
    da2 = float(np.abs(aud2 - src_recon).max())
    print(f"  (2) sde flat 0.0 == source latent     : max|d|={d2:.1e} (audio {da2:.1e})  "
          f"{'PASS' if g2 else 'FAIL'}")

    # (3) source-preservation gradient: sde ramp 0->1
    ramp = mx.linspace(0.0, 1.0, T)
    sde_ramp = run_slot(sde_denoise_curve=ramp)
    aud3 = decode(sde_ramp)
    save_wav(str(OUTDIR / "1_sde_ramp_0to1.wav"), aud3)
    segs = seg_cos_to_source(aud3, src_recon)
    grad = segs[-1] - segs[0]
    finite = bool(np.isfinite(aud3).all())
    g3 = grad < 0 and finite
    print(f"  (3) sde ramp 0->1 segment cos->src    : "
          f"[{', '.join(f'{c:.3f}' for c in segs)}]  grad={grad:+.3f}  "
          f"{'PASS' if g3 else 'FAIL'}")

    # (4a) shared override of a no-curve slot
    sh = run_slot(shared={"sde_denoise_curve": 0.0})
    d4a = maxabs(sh, init_latents)
    g4a = d4a == 0.0
    print(f"  (4a) shared sde=0 overrides slot      : max|d|={d4a:.1e}  {'PASS' if g4a else 'FAIL'}")

    # (4b) precedence: shared beats per-slot field
    prec = run_slot(shared={"sde_denoise_curve": 0.0}, sde_denoise_curve=1.0)
    d4b = maxabs(prec, init_latents)
    g4b = d4b == 0.0
    print(f"  (4b) shared beats per-slot field      : max|d|={d4b:.1e}  {'PASS' if g4b else 'FAIL'}")

    # (4c) revert: set then clear -> back to euler
    pipe = StreamPipeline(dit, T, steps=STEPS, depth=1)
    pipe.set_shared_curve("sde_denoise_curve", 0.0)
    pipe.set_shared_curve("sde_denoise_curve", None)
    pipe.submit(SlotRequest(cross_attn=cross, global_cond=gcond, seed=SEED,
                            denoise=SIGMA, source_latent=init_latents))
    rev = next(r for r in (pipe.tick() for _ in range(STEPS + 4)) if r is not None)
    d4c = maxabs(rev, euler)
    g4c = d4c == 0.0
    print(f"  (4c) clear reverts to euler           : max|d|={d4c:.1e}  {'PASS' if g4c else 'FAIL'}")

    # (4d) mid-stream onset: euler for 4 steps, then shared sde=0 anchors the rest
    pipe = StreamPipeline(dit, T, steps=STEPS, depth=1)
    pipe.submit(SlotRequest(cross_attn=cross, global_cond=gcond, seed=SEED,
                            denoise=SIGMA, source_latent=init_latents))
    mid = None
    for i in range(STEPS + 4):
        if i == 4:
            pipe.set_shared_curve("sde_denoise_curve", 0.0)
        r = pipe.tick()
        if r is not None:
            mid = r
            break
    cos_mid = cos(mean_mag_spec(to_mono(decode(mid))), mean_mag_spec(to_mono(src_recon)))
    cos_eul = cos(mean_mag_spec(to_mono(decode(euler))), mean_mag_spec(to_mono(src_recon)))
    g4d = cos_mid > cos_eul
    print(f"  (4d) mid-stream set anchors remainder : cos->src euler={cos_eul:.3f} "
          f"-> mid-set={cos_mid:.3f}  {'PASS' if g4d else 'FAIL'}")

    gates = [g1, g2, g3, g4a, g4b, g4c, g4d]
    print()
    print("=" * 72)
    print(f"  PHASE 2a CURVE GATES: {'PASS' if all(gates) else 'FAIL'} "
          f"({sum(gates)}/{len(gates)})")
    print(f"  demo WAVs in {OUTDIR}  (0_source_recon, 1_sde_ramp_0to1)")
    print("=" * 72)


if __name__ == "__main__":
    main()
