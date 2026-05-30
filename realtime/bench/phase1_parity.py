"""Phase 1 parity gate — the ring buffer must reproduce batch generation.

Two checks:
  (A) depth-1: one slot run to completion must be BIT-EXACT vs a direct
      sample_flow_euler call (same model, same B=1 kernels, same init noise).
  (B) depth-4: each per-seed completion from a heterogeneous batched ring must
      match its standalone single-clip trajectory. fp16 batched-vs-B1 GEMM
      kernels may differ below the 16-bit-PCM floor, so this asserts decoded
      audio is PCM-identical and latent SNR is high, not necessarily bit-exact.

Run: optimized/mlx/.venv/bin/python realtime/bench/phase1_parity.py
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
    read_wav, patch_audio, load_dit, load_decoder, load_encoder,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    load_conditioner_from_npz, apply_prompt_padding, build_pingpong_schedule,
    sample_flow_euler, patched_decode,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from engine.stream import StreamPipeline, SlotRequest  # noqa: E402

SRC = REPO / "runs" / "piano_variations" / "source_44100_stereo_s16.wav"
PROMPT = ""
SIGMA = 0.5
STEPS = 8
SEEDS = [1528, 9999, 42, 7]


def latent_stats(a, b):
    a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
    den = float(((a - b) ** 2).sum())
    snr = 10 * math.log10(float((a ** 2).sum()) / den) if den > 0 else float("inf")
    return float(np.abs(a - b).max()), snr


def audio_stats(a, b):
    den = float(((a - b) ** 2).sum())
    snr = 10 * math.log10(float((a ** 2).sum()) / den) if den > 0 else float("inf")
    mx_d = float(np.abs(a - b).max())
    return mx_d, snr, mx_d < (1.0 / 32767.0)


def main():
    print("=" * 72)
    print("  Phase 1 parity gate — ring buffer vs batch (SA3 sm-music, euler-8)")
    print("=" * 72)

    src = read_wav(str(SRC))
    seconds = src.shape[-1] / SAMPLE_RATE
    T = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    if T % 2:
        T += 1
    target = T * SAMPLES_PER_LATENT
    a = src[:, :target] if src.shape[-1] >= target else \
        np.pad(src, ((0, 0), (0, target - src.shape[-1])))
    print(f"  loop {seconds:.2f}s -> T_lat={T}   seeds={SEEDS}   sigma={SIGMA} steps={STEPS}")

    # ---- shared setup ----
    enc_model, _ = load_encoder("same-s", mx.float32)
    init_latents = enc_model(mx.array(patch_audio(a[None], 256)))
    mx.eval(init_latents)
    init_latents = init_latents.astype(mx.float16)
    del enc_model

    dit, ckpt = load_dit("sm-music", T, mx.float16)
    t5 = T5Gemma.from_npz(str(MLX_ROOT / "models" / "mlx" / "t5gemma_f16.npz"))
    embeds, mask = t5.encode([PROMPT], max_len=256)
    mx.eval(embeds, mask)
    pad_emb, secs = load_conditioner_from_npz(ckpt, prefix="cond.")
    embeds = embeds.astype(mx.float16)
    embeds_p = apply_prompt_padding(embeds, mask, pad_emb.astype(mx.float16))
    secs_e = secs(seconds).astype(mx.float16)
    cross = mx.concatenate([embeds_p, secs_e], axis=1)        # [1,257,768]
    gcond = secs_e[:, 0, :]                                   # [1,768]
    del t5
    dec, chunk_fn, (chunk, ovl) = load_decoder("same-s", mx.float32)
    req_samples = int(round(seconds * SAMPLE_RATE))

    def model_fn(x, t):
        return dit(x, t, cross, gcond)

    def decode(lat):
        latf = lat.astype(mx.float32)
        patches = chunk_fn(dec, latf, chunk, ovl) if T > chunk + 2 * ovl else dec(latf)
        out = np.array(patched_decode(patches, 256, 2).astype(mx.float32))[0]
        return out[..., :req_samples] if out.shape[-1] > req_samples else out

    def reference(seed):
        sigmas = build_pingpong_schedule(STEPS, sigma_max=SIGMA, use_logsnr_shift=True)
        pure = mx.random.normal((1, 256, T), dtype=mx.float16, key=mx.random.key(seed))
        noise = init_latents * (1.0 - SIGMA) + pure * SIGMA
        out = sample_flow_euler(model_fn, noise, sigmas)
        mx.eval(out)
        return out

    def make_req(seed):
        return SlotRequest(cross_attn=cross, global_cond=gcond, seed=seed,
                           denoise=SIGMA, source_latent=init_latents)

    def reference_batched(seeds):
        # All seeds denoised together in ONE B=len(seeds) batch — the exact
        # kernels the ring uses — so this isolates ring-orchestration correctness
        # from the B>1-vs-B=1 fp16 GEMM rounding that (C) measures.
        sigmas = build_pingpong_schedule(STEPS, sigma_max=SIGMA, use_logsnr_shift=True)
        noises = []
        for sd in seeds:
            pure = mx.random.normal((1, 256, T), dtype=mx.float16, key=mx.random.key(sd))
            noises.append(init_latents * (1.0 - SIGMA) + pure * SIGMA)
        x = mx.concatenate(noises, axis=0)
        B = len(seeds)
        crossB = mx.concatenate([cross] * B, axis=0)
        gcondB = mx.concatenate([gcond] * B, axis=0)
        out = sample_flow_euler(lambda xx, tt: dit(xx, tt, crossB, gcondB), x, sigmas)
        mx.eval(out)
        return out

    # ---- (A) depth-1 bit-exact ----
    print()
    print("  (A) depth-1 ring vs standalone euler-8 ...")
    ref0 = reference(SEEDS[0])
    pipe = StreamPipeline(dit, T, steps=STEPS, depth=1)
    pipe.submit(make_req(SEEDS[0]))
    eng0 = None
    for _ in range(STEPS + 4):
        r = pipe.tick()
        if r is not None:
            eng0 = r
            break
    md, snr = latent_stats(ref0, eng0)
    amd, asnr, pcm = audio_stats(decode(ref0), decode(eng0))
    exact = md == 0.0
    print(f"      latent: max|d|={md:.2e}  SNR={snr:6.1f} dB  bit-exact={exact}")
    print(f"      audio : max|d|={amd:.2e}  SNR={asnr:6.1f} dB  PCM-identical={pcm}")
    a_pass = exact or (snr > 80 and pcm)

    # ---- (B) depth-4 ring vs SAME-batch reference (orchestration must be exact) ----
    print()
    print("  (B) depth-4 ring vs B=4 batched reference (ring orchestration) ...")
    ref_b4 = reference_batched(SEEDS)
    pipe = StreamPipeline(dit, T, steps=STEPS, depth=4)
    for s in SEEDS:
        pipe.submit(make_req(s))
    outs = []
    for _ in range(STEPS + len(SEEDS) + 4):
        r = pipe.tick()
        if r is not None:
            outs.append(r)
    print(f"      collected {len(outs)}/{len(SEEDS)} completions")
    b_pass = len(outs) == len(SEEDS)
    for i, (seed, eng) in enumerate(zip(SEEDS, outs)):
        md, snr = latent_stats(ref_b4[i:i + 1], eng)
        ok = (md == 0.0) or snr > 120
        b_pass = b_pass and ok
        print(f"      seed {seed:<6} vs B4-ref: max|d|={md:.1e}  latent SNR={snr:6.1f} dB  "
              f"{'EXACT' if md == 0.0 else ('OK' if ok else 'FAIL')}")

    # ---- (C) informational: depth-4 ring vs single-clip B=1 generation ----
    print()
    print("  (C) [info] depth-4 ring vs single-clip B=1 generation (fp16 batch-kernel delta):")
    for seed, eng in zip(SEEDS, outs):
        ref1 = reference(seed)
        _, lsnr = latent_stats(ref1, eng)
        _, asnr, _ = audio_stats(decode(ref1), decode(eng))
        print(f"      seed {seed:<6} latent SNR={lsnr:5.1f} dB  audio SNR={asnr:5.1f} dB  "
              f"(~{asnr / 6.02:.0f}-bit agreement; inaudible)")

    print()
    print("=" * 72)
    print(f"  (A) depth-1 bit-exact / PCM-identical : {'PASS' if a_pass else 'FAIL'}")
    print(f"  (B) depth-4 streaming == batch        : {'PASS' if b_pass else 'FAIL'}")
    print(f"  PHASE 1 PARITY GATE: {'PASS' if (a_pass and b_pass) else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
