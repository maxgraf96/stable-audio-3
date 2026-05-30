"""Phase 2c gates — CFG + APG + RCFG-self.

  (A) cfg==1.0 unchanged : a cfg=1 slot is bit-exact to the Phase-1 euler path
  (B) CFG parity         : a depth-1 cfg>1 ring slot reproduces a standalone
                           sample_flow_euler whose model_fn does cond+uncond as
                           TWO separate B=1 forwards (the engine's exact call
                           structure). Bit-exact target. [info] also report vs the
                           product's B=2-FUSED model_fn: differs at the sub-PCM
                           fp16 batch-kernel floor, amplified by cfg/APG.
  (C) APG parity         : same, with apg=0.5 (vanilla/orth blend)
  (D) neg-prompt parity  : same, cfg>1 with a real negative prompt
  (E) RCFG-self          : zero uncond forwards, finite output, distinct from cfg=1
                           and from full-CFG (a real, cheaper guidance variant)

Run: optimized/mlx/.venv/bin/python realtime/bench/phase2c_cfg.py
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
    read_wav, patch_audio, load_dit, load_encoder,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    load_conditioner_from_npz, apply_prompt_padding, build_pingpong_schedule,
    sample_flow_euler,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from engine.stream import StreamPipeline, SlotRequest  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
PROMPT = "vintage electric piano riff"
NEG_PROMPT = "drums, vocals"
SIGMA = 0.6
STEPS = 8
SEED = 1528


def maxabs(a, b):
    return float(np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32))).max())


def snr(a, b):
    a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
    den = float(((a - b) ** 2).sum())
    return 10 * math.log10(float((a ** 2).sum()) / den) if den > 0 else float("inf")


def main():
    print("=" * 72)
    print("  Phase 2c gates — CFG + APG + RCFG-self (SA3 sm-music)")
    print("=" * 72)

    src = read_wav(str(SRC))
    seconds = src.shape[-1] / SAMPLE_RATE
    T = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if T % 2:
        T += 1
    target = T * SAMPLES_PER_LATENT
    a = src[:, :target] if src.shape[-1] >= target else \
        np.pad(src, ((0, 0), (0, target - src.shape[-1])))
    print(f"  loop {seconds:.2f}s -> T_lat={T}  sigma={SIGMA} steps={STEPS}")
    print(f"  prompt={PROMPT!r}  neg={NEG_PROMPT!r}")

    enc_model, _ = load_encoder("same-s", mx.float32)
    init_latents = enc_model(mx.array(patch_audio(a[None], 256))).astype(mx.float16)
    mx.eval(init_latents)
    del enc_model
    dit, ckpt = load_dit("sm-music", T, mx.float16)
    t5 = T5Gemma.from_npz(str(MLX_ROOT / "models" / "mlx" / "t5gemma_f16.npz"))
    pad_emb, secs = load_conditioner_from_npz(ckpt, prefix="cond.")
    secs_e = secs(seconds).astype(mx.float16)
    gcond = secs_e[:, 0, :]

    def encode(text):
        e, m = t5.encode([text], max_len=256)
        mx.eval(e, m)
        ep = apply_prompt_padding(e.astype(mx.float16), m, pad_emb.astype(mx.float16))
        return mx.concatenate([ep, secs_e], axis=1)          # [1,257,768]

    cross = encode(PROMPT)
    neg_cross = encode(NEG_PROMPT)
    zeros_cross = mx.zeros_like(cross)
    del t5

    # init noise matches engine _init_slot exactly: key(seed), a2a mix at SIGMA
    pure = mx.random.normal((1, 256, T), dtype=mx.float16, key=mx.random.key(SEED))
    noise = init_latents * (1.0 - SIGMA) + pure * SIGMA
    mx.eval(noise)
    sigmas = build_pingpong_schedule(STEPS, sigma_max=SIGMA, use_logsnr_shift=True)

    def _apg_combine(x, cond_v, unc_v, t, cfg, apg):
        sigma = t.reshape(-1, 1, 1).astype(mx.float32)
        cd = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
        ud = x.astype(mx.float32) - unc_v.astype(mx.float32) * sigma
        diff = cd - ud
        if apg <= 0.0:
            cdf = diff
        else:
            nrm = mx.sqrt((cd * cd).sum(axis=(-2, -1), keepdims=True))
            unit = cd / mx.maximum(nrm, 1e-8)
            par = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
            orth = diff - par
            cdf = orth if apg >= 1.0 else (apg * orth + (1.0 - apg) * diff)
        cfg_d = cd + (cfg - 1.0) * cdf
        return ((x.astype(mx.float32) - cfg_d) / sigma).astype(x.dtype)

    def reference(cfg, apg, null_cross):
        # SPLIT structure: cond and uncond as TWO separate B=1 forwards, matching
        # the engine's two-pass _advance. Isolates CFG math from batch-kernel rounding.
        def fn(x, t):
            if cfg == 1.0:
                return dit(x, t, cross, gcond)
            cv = dit(x, t, cross, gcond)
            uv = dit(x, t, null_cross, gcond)
            return _apg_combine(x, cv, uv, t, cfg, apg)
        out = sample_flow_euler(fn, noise, sigmas)
        mx.eval(out)
        return out

    def reference_fused(cfg, apg, null_cross):
        # The product's B=2-fused model_fn (cat([x,x])). Informational only.
        def fn(x, t):
            x2 = mx.concatenate([x, x], axis=0)
            t2 = mx.concatenate([t, t], axis=0)
            c2 = mx.concatenate([cross, null_cross], axis=0)
            g2 = mx.concatenate([gcond, gcond], axis=0)
            vb = dit(x2, t2, c2, g2)
            cv, uv = mx.split(vb, 2, axis=0)
            return _apg_combine(x, cv, uv, t, cfg, apg)
        out = sample_flow_euler(fn, noise, sigmas)
        mx.eval(out)
        return out

    def ring(cfg, apg, rcfg_mode="full", neg=None):
        pipe = StreamPipeline(dit, T, steps=STEPS, depth=1)
        pipe.submit(SlotRequest(cross_attn=cross, global_cond=gcond, seed=SEED,
                                denoise=SIGMA, source_latent=init_latents,
                                cfg=cfg, apg=apg, rcfg_mode=rcfg_mode,
                                neg_cross_attn=neg))
        for _ in range(STEPS + 4):
            r = pipe.tick()
            if r is not None:
                return r
        raise RuntimeError("no completion")

    # (A) cfg=1 unchanged
    rA = ring(1.0, 1.0)
    refA = reference(1.0, 1.0, zeros_cross)
    gA = maxabs(rA, refA) == 0.0
    print(f"\n  (A) cfg=1.0 == euler baseline      : max|d|={maxabs(rA, refA):.1e}  {'PASS' if gA else 'FAIL'}")

    # (B) CFG parity (cfg=4, apg=1, zeros uncond) vs structure-matched split ref
    rB = ring(4.0, 1.0, neg=None)
    refB = reference(4.0, 1.0, zeros_cross)
    dB = maxabs(rB, refB)
    gB = dB == 0.0
    fusedB = snr(reference_fused(4.0, 1.0, zeros_cross), rB)
    print(f"  (B) cfg=4 apg=1 vs split-ref       : max|d|={dB:.1e}  {'PASS' if gB else 'FAIL'}"
          f"   [info vs B=2-fused: {fusedB:.0f} dB]")

    # (C) APG parity (cfg=4, apg=0.5)
    rC = ring(4.0, 0.5, neg=None)
    refC = reference(4.0, 0.5, zeros_cross)
    dC = maxabs(rC, refC)
    gC = dC == 0.0
    print(f"  (C) cfg=4 apg=0.5 vs split-ref     : max|d|={dC:.1e}  {'PASS' if gC else 'FAIL'}")

    # (D) negative-prompt parity
    rD = ring(4.0, 1.0, neg=neg_cross)
    refD = reference(4.0, 1.0, neg_cross)
    dD = maxabs(rD, refD)
    gD = dD == 0.0
    print(f"  (D) cfg=4 +neg-prompt vs split-ref : max|d|={dD:.1e}  {'PASS' if gD else 'FAIL'}")

    # (E) RCFG-self: finite, distinct from cfg=1 and from full-CFG
    rE = ring(4.0, 1.0, rcfg_mode="self")
    finite = bool(np.isfinite(np.array(rE.astype(mx.float32))).all())
    d_vs_cfg1 = maxabs(rE, rA)
    d_vs_full = maxabs(rE, rB)
    gE = finite and d_vs_cfg1 > 1e-3 and d_vs_full > 1e-3
    print(f"  (E) RCFG-self (0 uncond forwards)  : finite={finite}  "
          f"d(vs cfg1)={d_vs_cfg1:.2f} d(vs full)={d_vs_full:.2f}  {'PASS' if gE else 'FAIL'}")

    gates = [gA, gB, gC, gD, gE]
    print()
    print("=" * 72)
    print(f"  PHASE 2c CFG GATES: {'PASS' if all(gates) else 'FAIL'} ({sum(gates)}/{len(gates)})")
    print("=" * 72)


if __name__ == "__main__":
    main()
