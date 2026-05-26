"""In-process MLX inference pipeline for SA3 — sa3_variations.py's runtime.

Loads T5Gemma + DiT + decoder (+ encoder) once and runs every candidate in the
same process. Replaces the previous 1-subprocess-per-candidate path that
re-loaded ~5 GB of weights for each variation. All model code is reused from
optimized/mlx/ — nothing is duplicated here.

Usage:
    from sa3_pipeline import Pipeline, read_wav_44k_stereo_s16
    pipe = Pipeline(dit="medium", decoder="same-l", seconds=8.0)
    init = read_wav_44k_stereo_s16(prepared_wav)
    pipe.generate(prompt="...", seed=42, init_audio_np=init, out_path="out.wav")
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parent
_MLX_ROOT = _REPO / "optimized" / "mlx"
if not _MLX_ROOT.exists():
    raise RuntimeError(f"optimized/mlx not found at {_MLX_ROOT}")
sys.path.insert(0, str(_MLX_ROOT))
sys.path.insert(0, str(_MLX_ROOT / "scripts"))

import mlx.core as mx  # noqa: E402
from models.defs.sa3_pipeline import (  # noqa: E402
    apply_prompt_padding,
    build_pingpong_schedule,
    load_conditioner_from_npz,
    patched_decode,
    sample_flow_pingpong,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from weights import ensure_local  # noqa: E402
from sa3_mlx import (  # noqa: E402
    DECODER_CHOICES,
    DIT_CHOICES,
    ENCODER_CHOICES,
    SAMPLE_RATE,
    SAMPLES_PER_LATENT,
    T5GEMMA_NPZ_REL,
    load_decoder,
    load_dit,
    load_encoder,
    patch_audio,
)
from sa3_mlx import read_wav as _mlx_read_wav  # noqa: E402
from sa3_mlx import save_wav as _mlx_save_wav  # noqa: E402

MIN_SIGMA = 0.01


def read_wav_44k_stereo_s16(path: str) -> np.ndarray:
    """Read a 16-bit PCM WAV at 44.1 kHz into (2, T) float32 in [-1, 1]."""
    return _mlx_read_wav(str(path))


class Pipeline:
    """Resident MLX pipeline. Load once, generate many.

    All candidates in a variations batch share dit/decoder/seconds/dtype; only
    prompt, seed, cfg, apg, noise level, init audio, and inpaint range vary
    per call. cross_attn and init-audio latents are cached so repeats are free.
    """

    def __init__(
        self,
        dit: str,
        decoder: str,
        seconds: float,
        dit_dtype: str = "fp16",
        load_encoder_now: bool = True,
        verbose: bool = True,
    ):
        if dit not in DIT_CHOICES:
            raise ValueError(f"unknown dit {dit!r}; choices: {list(DIT_CHOICES)}")
        if decoder not in DECODER_CHOICES:
            raise ValueError(f"unknown decoder {decoder!r}; choices: {list(DECODER_CHOICES)}")

        self.dit_name = dit
        self.decoder_name = decoder
        self.seconds = float(seconds)
        self.dtype = mx.float32 if dit_dtype == "fp32" else mx.float16
        self.verbose = verbose

        T_lat = max(1, math.ceil(self.seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
        if decoder == "same-s" and T_lat % 2 != 0:
            T_lat += 1
        self.T_lat = T_lat

        def _log(label: str, t0: float) -> None:
            if verbose:
                print(f"[pipeline] {label:<18} {time.time() - t0:.2f}s", file=sys.stderr)

        t0 = time.time()
        self.t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
        _log("T5Gemma load", t0)

        t0 = time.time()
        dit_npz = str(ensure_local(DIT_CHOICES[dit]["ckpt"]))
        self.padding_emb, self.secs_embedder = load_conditioner_from_npz(dit_npz, prefix="cond.")
        self.seconds_embed = self.secs_embedder(self.seconds).astype(self.dtype)
        self.global_cond = self.seconds_embed[:, 0, :]
        mx.eval(self.seconds_embed, self.global_cond)
        _log("Conditioner", t0)

        t0 = time.time()
        self.dit_model, _ = load_dit(dit, T_lat=T_lat, dtype=self.dtype)
        _log("DiT load", t0)

        t0 = time.time()
        self.decoder, self.chunk_fn, (self.chunk, self.ovl) = load_decoder(decoder, mx.float32)
        _log("Decoder load", t0)

        self.encoder = None
        self.pad_mod = None
        if load_encoder_now:
            t0 = time.time()
            self.encoder, self.pad_mod = load_encoder(decoder, mx.float32)
            _log("Encoder load", t0)

        self._cross_attn_cache: dict[str, mx.array] = {}
        self._init_latents_cache: dict[bytes, mx.array] = {}

    def _ensure_encoder(self) -> None:
        if self.encoder is None:
            self.encoder, self.pad_mod = load_encoder(self.decoder_name, mx.float32)

    def _encode_text(self, prompt: str) -> mx.array:
        cached = self._cross_attn_cache.get(prompt)
        if cached is not None:
            return cached
        embeds, mask = self.t5.encode([prompt], max_len=256)
        mx.eval(embeds, mask)
        embeds = embeds.astype(self.dtype)
        padded = apply_prompt_padding(embeds, mask, self.padding_emb.astype(self.dtype))
        cross_attn = mx.concatenate([padded, self.seconds_embed], axis=1)
        mx.eval(cross_attn)
        self._cross_attn_cache[prompt] = cross_attn
        return cross_attn

    def encode_init_audio(self, audio_np: np.ndarray) -> mx.array:
        """Encode (channels, T_audio) float32 in [-1,1] to DiT latents. Cached."""
        self._ensure_encoder()
        target_samples = self.T_lat * SAMPLES_PER_LATENT
        if audio_np.shape[-1] >= target_samples:
            audio_np = audio_np[:, :target_samples]
        elif audio_np.shape[-1] < target_samples:
            pad = target_samples - audio_np.shape[-1]
            audio_np = np.pad(audio_np, ((0, 0), (0, pad)), mode="constant")
        key = audio_np.tobytes()
        cached = self._init_latents_cache.get(key)
        if cached is not None:
            return cached
        patches_np = patch_audio(audio_np[None, ...], patch_size=256)
        assert patches_np.shape[-1] % self.pad_mod == 0, (
            f"T_audio_patches={patches_np.shape[-1]} not divisible by {self.pad_mod}"
        )
        init_latents = self.encoder(mx.array(patches_np))
        mx.eval(init_latents)
        init_latents = init_latents.astype(self.dtype)
        self._init_latents_cache[key] = init_latents
        return init_latents

    def _seconds_to_latent_range(
        self, inpaint_seconds: tuple[float, float]
    ) -> tuple[int, int]:
        s, e = inpaint_seconds
        if not (0 <= s < e <= self.seconds + 1e-6):
            raise ValueError(
                f"inpaint range {s:.3f}-{e:.3f}s outside [0, {self.seconds:.3f}]"
            )
        s0 = max(0, int(round(s * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        s1 = min(self.T_lat, int(round(e * SAMPLE_RATE / SAMPLES_PER_LATENT)))
        if s1 <= s0:
            raise ValueError(
                f"inpaint range {s:.3f}-{e:.3f}s rounds to empty latent span "
                f"(s0={s0}, s1={s1}, T_lat={self.T_lat})"
            )
        return s0, s1

    def generate(
        self,
        prompt: str,
        seed: int,
        steps: int = 8,
        init_noise_level: float = 1.0,
        cfg: float = 1.0,
        apg: float = 1.0,
        negative_prompt: Optional[str] = None,
        init_audio_np: Optional[np.ndarray] = None,
        inpaint_range_seconds: Optional[tuple[float, float]] = None,
        out_path: Optional[str] = None,
    ) -> np.ndarray:
        sigma_max = float(init_noise_level)
        if sigma_max < MIN_SIGMA:
            raise ValueError(
                f"init_noise_level={sigma_max} < {MIN_SIGMA}: rf_denoiser is undefined at t≈0"
            )

        cross_attn = self._encode_text(prompt)
        null_cross_attn = None
        if cfg != 1.0:
            if negative_prompt:
                null_cross_attn = self._encode_text(negative_prompt)
            else:
                null_cross_attn = mx.zeros_like(cross_attn)
            mx.eval(null_cross_attn)

        init_latents = None
        if init_audio_np is not None:
            init_latents = self.encode_init_audio(init_audio_np)

        local_add_cond = None
        paste_back = None
        if inpaint_range_seconds is not None:
            if init_latents is None:
                raise ValueError("inpaint_range_seconds requires init_audio_np")
            s0, s1 = self._seconds_to_latent_range(inpaint_range_seconds)
            mask_np = np.ones((1, 1, self.T_lat), dtype=np.float32)
            mask_np[:, :, s0:s1] = 0.0
            mask = mx.array(mask_np)
            masked_input = init_latents.astype(mx.float32) * mask
            local_add_cond = (
                mx.concatenate([mask, masked_input], axis=1)
                .transpose(0, 2, 1)
                .astype(self.dtype)
            )
            paste_back = (init_latents, mask)

        key = mx.random.key(seed)
        pure_noise = mx.random.normal((1, 256, self.T_lat), dtype=self.dtype, key=key)
        if init_latents is not None and inpaint_range_seconds is None:
            noise = init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
        else:
            noise = pure_noise
        mx.eval(noise)

        dit_model = self.dit_model
        global_cond = self.global_cond

        def model_fn(x, t):
            if cfg == 1.0:
                return dit_model(x, t, cross_attn, global_cond, local_add_cond=local_add_cond)
            x2 = mx.concatenate([x, x], axis=0)
            t2 = mx.concatenate([t, t], axis=0)
            cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
            global2 = mx.concatenate([global_cond, global_cond], axis=0)
            lac2 = (
                None
                if local_add_cond is None
                else mx.concatenate([local_add_cond, local_add_cond], axis=0)
            )
            v_batched = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
            cond_v, uncond_v = mx.split(v_batched, 2, axis=0)
            sigma = t.reshape(-1, 1, 1).astype(mx.float32)
            cond_d = x.astype(mx.float32) - cond_v.astype(mx.float32) * sigma
            uncond_d = x.astype(mx.float32) - uncond_v.astype(mx.float32) * sigma
            diff = cond_d - uncond_d
            if apg <= 0.0:
                cfg_diff = diff
            else:
                norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
                unit = cond_d / mx.maximum(norm, 1e-8)
                parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
                diff_orth = diff - parallel
                cfg_diff = (
                    diff_orth
                    if apg >= 1.0
                    else (apg * diff_orth + (1.0 - apg) * diff)
                )
            cfg_d = cond_d + (cfg - 1.0) * cfg_diff
            cfg_v = (x.astype(mx.float32) - cfg_d) / sigma
            return cfg_v.astype(x.dtype)

        sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max, use_logsnr_shift=True)
        latents = sample_flow_pingpong(
            model_fn, noise, sigmas, seed=seed + 1, paste_back=paste_back
        )
        mx.eval(latents)

        latents_fp32 = latents.astype(mx.float32)
        kernel = self.chunk + 2 * self.ovl
        if self.T_lat > kernel:
            patches = self.chunk_fn(self.decoder, latents_fp32, self.chunk, self.ovl)
        elif self.T_lat % 2 == 0:
            patches = self.decoder(latents_fp32)
        else:
            patches = self.chunk_fn(self.decoder, latents_fp32, 2, 2)
        mx.eval(patches)

        audio = patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)
        audio_np = np.array(audio.astype(mx.float32))[0]
        requested_samples = int(round(self.seconds * SAMPLE_RATE))
        if audio_np.shape[-1] > requested_samples:
            audio_np = audio_np[..., :requested_samples]

        if out_path is not None:
            _mlx_save_wav(str(out_path), audio_np)
        return audio_np
