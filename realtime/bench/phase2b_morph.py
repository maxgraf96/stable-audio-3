"""Phase 2b gates — x0-target morph toward a banked variant.

Banks a more-transformed variant of the loop (a denoise=0.8 generation) as the
target B, then morphs a denoise=0.5 generation toward it.

  (A) inactive   : x0_target set but strength None -> bit-exact to plain euler
  (B) morph      : full strength -> output moves toward B (cos->B up vs baseline)
  (C) gradient   : strength ramp 0->1 -> per-segment similarity to B ascends
                   (DEMON Table 9: the morph climbs toward B across the song)
  (D) shared     : shared x0_target_strength == per-slot field (1-tick live alpha)

Run: optimized/mlx/.venv/bin/python realtime/bench/phase2b_morph.py
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
TARGET_SEED = 4242
OUTDIR = REPO / "runs" / "phase2b_morph"


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


def seg_cos(audio, ref, n=4):
    L = min(audio.shape[-1], ref.shape[-1])
    return [cos(mean_mag_spec(to_mono(audio[..., i * L // n:(i + 1) * L // n])),
               mean_mag_spec(to_mono(ref[..., i * L // n:(i + 1) * L // n]))) for i in range(n)]


def main():
    print("=" * 72)
    print("  Phase 2b gates — x0-target morph toward a banked variant")
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
    print(f"  loop {seconds:.2f}s -> T_lat={T}  base sigma={SIGMA} target sigma=0.8")

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

    def run_slot(seed=SEED, denoise=SIGMA, shared=None, **kw):
        pipe = StreamPipeline(dit, T, steps=STEPS, depth=1)
        if shared:
            for k, v in shared.items():
                pipe.set_shared_curve(k, v)
        pipe.submit(SlotRequest(cross_attn=cross, global_cond=gcond, seed=seed,
                                denoise=denoise, source_latent=init_latents, **kw))
        for _ in range(STEPS + 4):
            r = pipe.tick()
            if r is not None:
                return r
        raise RuntimeError("no completion")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    target_B = run_slot(seed=TARGET_SEED, denoise=0.8)      # banked variant
    base = run_slot(seed=SEED, denoise=SIGMA)               # baseline (euler)
    B_aud = decode(target_B); base_aud = decode(base)
    save_wav(str(OUTDIR / "0_base.wav"), base_aud)
    save_wav(str(OUTDIR / "1_target_B.wav"), B_aud)
    specB = mean_mag_spec(to_mono(B_aud))

    def cosB(aud):
        return cos(mean_mag_spec(to_mono(aud)), specB)

    # (A) inactive: target set, no strength -> bit-exact to base
    inactive = run_slot(seed=SEED, x0_target=target_B)
    dA = maxabs(inactive, base)
    gA = dA == 0.0
    print(f"\n  (A) target set, strength None == euler : max|d|={dA:.1e}  {'PASS' if gA else 'FAIL'}")

    # (B) full-strength morph toward B
    morph = run_slot(seed=SEED, x0_target=target_B, x0_target_strength=1.0)
    morph_aud = decode(morph)
    save_wav(str(OUTDIR / "2_morph_full.wav"), morph_aud)
    cb_base, cb_morph = cosB(base_aud), cosB(morph_aud)
    gB = cb_morph > cb_base
    print(f"  (B) full morph moves toward B          : cos->B base={cb_base:.3f} "
          f"-> morph={cb_morph:.3f}  {'PASS' if gB else 'FAIL'}")

    # (C) strength ramp 0->1 -> ascending similarity to B
    ramp = mx.linspace(0.0, 1.0, T)
    morph_ramp = run_slot(seed=SEED, x0_target=target_B, x0_target_strength=ramp)
    mr_aud = decode(morph_ramp)
    save_wav(str(OUTDIR / "3_morph_ramp.wav"), mr_aud)
    segs = seg_cos(mr_aud, B_aud)
    grad = segs[-1] - segs[0]
    gC = grad > 0 and bool(np.isfinite(mr_aud).all())
    print(f"  (C) ramp 0->1 segment cos->B           : "
          f"[{', '.join(f'{c:.3f}' for c in segs)}]  grad={grad:+.3f}  {'PASS' if gC else 'FAIL'}")

    # (D) shared alpha == per-slot alpha
    sh_morph = run_slot(seed=SEED, x0_target=target_B, shared={"x0_target_strength": 1.0})
    dD = maxabs(sh_morph, morph)
    gD = dD == 0.0
    print(f"  (D) shared strength == per-slot field   : max|d|={dD:.1e}  {'PASS' if gD else 'FAIL'}")

    gates = [gA, gB, gC, gD]
    print()
    print("=" * 72)
    print(f"  PHASE 2b MORPH GATES: {'PASS' if all(gates) else 'FAIL'} ({sum(gates)}/{len(gates)})")
    print(f"  demo WAVs in {OUTDIR}  (0_base, 1_target_B, 2_morph_full, 3_morph_ramp)")
    print("=" * 72)


if __name__ == "__main__":
    main()
