"""In-process PyTorch inference pipeline for SA3 — the CUDA/CPU runtime.

Mirrors sa3_pipeline_mlx.Pipeline's API exactly, so sa3_variations.py and
sa3_studio.py run unchanged off Apple Silicon. All model code is reused from the
upstream `stable_audio_3` package — nothing is duplicated here.

The MLX backend hand-rolls the sampling loop against the converted npz weights;
this one drives `StableAudioModel.generate()`, which already exposes every knob
the variation harness needs (init_audio + init_noise_level for a2a, inpaint
mask in seconds, cfg, APG, seed, steps) and defaults to the same ping-pong
sampler for the distilled rf_denoiser objective.

Usage:
    from sa3_pipeline_torch import Pipeline, read_wav_44k_stereo_s16
    pipe = Pipeline(dit="medium", decoder="same-l", seconds=8.0)
    init = read_wav_44k_stereo_s16(prepared_wav)
    pipe.generate(prompt="...", seed=42, init_audio_np=init, out_path="out.wav")
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from stable_audio_3 import StableAudioModel

SAMPLE_RATE = 44100
MIN_SIGMA = 0.01


class _NormalisedShift:
    """Applies the model's timestep shift in normalised t, then rescales.

    `build_schedule` warps an already-sigma-scaled ramp: the shift assumes its
    input spans [0,1], so linspace(0.10, 0) comes back as [0.12 .. 0.217] —
    every interior step lands ABOVE the requested sigma, and only t[0] is
    re-anchored. Asking for 0.45 actually peaks at 0.774.

    Normalising first keeps the curve's shape while confining it to
    [0, sigma_max], so the noise level means what it says. At sigma_max=1.0
    this is a no-op, which is why full text-to-audio generation never showed
    the problem — it only ever bit variations.

    Mirrors the `* sigma_max` form used in sa3_pipeline_mlx.py; kept as a shim
    here because the torch schedule is built deep inside sample_diffusion and
    `dist_shift` is the only seam into it.
    """

    def __init__(self, base, sigma_max: float):
        self.base = base
        self.sigma_max = float(sigma_max)

    def shift(self, t, seq_len):
        if self.base is None:
            return t
        if self.sigma_max <= 0.0:
            return self.base.shift(t, seq_len)
        import torch as _torch

        normalised = _torch.clamp(t / self.sigma_max, 0.0, 1.0)
        return self.base.shift(normalised, seq_len) * self.sigma_max

# The variation harness speaks the MLX runner's model names; map them onto the
# torch checkpoint names. The decoder isn't separately selectable here — each
# checkpoint bundles its own SAME autoencoder — so it is validated, not chosen.
DIT_CHOICES = {
    "medium": {"model": "medium", "decoder": "same-l"},
    "sm-music": {"model": "small-music", "decoder": "same-s"},
    "sm-sfx": {"model": "small-sfx", "decoder": "same-s"},
}
DECODER_CHOICES = ("same-s", "same-l")


def read_wav_44k_stereo_s16(path: str) -> np.ndarray:
    """Read a 16-bit PCM WAV at 44.1 kHz into (2, T) float32 in [-1, 1].

    Mono is duplicated to stereo. Matches sa3_mlx.read_wav's contract; callers
    hand us a file ffmpeg already normalised (see convert_to_mlx_wav).
    """
    with wave.open(str(path), "rb") as w:
        nch, sw, sr, nframes = (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getnframes(),
        )
        if sw != 2 or sr != SAMPLE_RATE:
            raise RuntimeError(
                f"{path}: expected 44.1 kHz 16-bit PCM, got {sr} Hz / {sw * 8}-bit. "
                f"Convert with ffmpeg -ar {SAMPLE_RATE} -ac 2 -sample_fmt s16."
            )
        raw = np.frombuffer(w.readframes(nframes), dtype=np.int16)
    raw = raw.astype(np.float32) / 32767.0
    if nch == 1:
        return np.stack([raw, raw], axis=0)
    return np.ascontiguousarray(raw.reshape(-1, nch).T[:2])


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """audio: (channels, T) float32 in [-1, 1]. Writes 16-bit PCM WAV."""
    if not np.isfinite(audio).all():
        n_bad = int((~np.isfinite(audio)).sum())
        raise RuntimeError(
            f"refusing to write WAV — audio contains {n_bad} non-finite samples (NaN/Inf)"
        )
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).T  # (T, channels) interleaved
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def models_root() -> Optional[Path]:
    """Where an installed app keeps its weights, or None on a dev machine.

    Mirrors resolveModelsDir() in the MLX backend's VariationsEngine, including
    the SA3_MODELS_DIR dev override, so both platforms answer "where are the
    models" the same way:

      1. $SA3_MODELS_DIR                              — dev override
      2. %LOCALAPPDATA%\\SA3 Variations\\models         — Windows install
      3. ~/.local/share/SA3 Variations/models         — Linux install
    """
    env = os.environ.get("SA3_MODELS_DIR", "").strip()
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            p = Path(base) / "SA3 Variations" / "models"
            return p if p.is_dir() else None
        return None
    p = Path.home() / ".local" / "share" / "SA3 Variations" / "models"
    return p if p.is_dir() else None


def find_local_model_dir(model_name: str) -> Optional[Path]:
    """The installed directory holding `model_name`, if it's fully present.

    A partial download (interrupted install) must not look like a usable model,
    or the failure surfaces much later as a confusing safetensors error — so
    both the config and the checkpoint have to exist before we claim it.
    """
    root = models_root()
    if root is None:
        return None
    d = root / model_name
    if (d / "model_config.json").is_file() and (d / "model.safetensors").is_file():
        return d
    return None


def load_local_model(model_dir: Path, device: str, model_half: bool) -> StableAudioModel:
    """Load a model from disk, with no network access at any point.

    Reuses upstream's own loader rather than reimplementing it: everything here
    is what StableAudioModel.from_pretrained does, minus the hub round-trip.

    The one rewrite is the text encoder. The shipped config points it at the
    gated repo it was published in:

        {"repo_id": "stabilityai/stable-audio-3-medium",
         "subfolder": "t5gemma-b-b-ul2"}

    which would send transformers to the Hub — and fail without a token. The
    conditioner resolves `load_from = model_path or repo_id or model_name`
    (stable_audio_3/models/conditioners.py), so pointing `model_path` at the
    copy sitting next to the checkpoint makes the load fully local. It's done
    here rather than baked into the published config because the absolute path
    differs on every machine.
    """
    from stable_audio_3.loading_utils import load_diffusion_cond

    with open(model_dir / "model_config.json") as f:
        model_config = json.load(f)

    t5_dir = model_dir / "t5gemma-b-b-ul2"
    for cond in model_config.get("model", {}).get("conditioning", {}).get("configs", []):
        if cond.get("type") == "t5gemma":
            cfg = cond.setdefault("config", {})
            cfg.pop("repo_id", None)
            cfg.pop("subfolder", None)
            cfg["model_path"] = str(t5_dir)

    model = load_diffusion_cond(
        model_config, str(model_dir / "model.safetensors"),
        device=device, model_half=model_half,
    )
    model.use_lora = False
    model.lora_names = []
    return StableAudioModel(model, model_config, device, model_half)


class Pipeline:
    """Resident PyTorch pipeline. Load once, generate many.

    All candidates in a variations batch share dit/decoder/seconds; only prompt,
    seed, cfg, apg, noise level, init audio, and inpaint range vary per call.
    Unlike the MLX backend, `seconds` doesn't fix a latent length at construction
    — it is passed per generate() call — but it is still held on the instance so
    sa3_variations.py's "is the resident pipeline still the right one?" check
    works identically across backends.
    """

    backend = "torch"

    def __init__(
        self,
        dit: str,
        decoder: str,
        seconds: float,
        dit_dtype: str = "fp16",
        load_encoder_now: bool = True,  # noqa: ARG002 - torch loads the full AE with the model
        verbose: bool = True,
        device: Optional[str] = None,
    ):
        if dit not in DIT_CHOICES:
            raise ValueError(f"unknown dit {dit!r}; choices: {list(DIT_CHOICES)}")
        if decoder not in DECODER_CHOICES:
            raise ValueError(f"unknown decoder {decoder!r}; choices: {list(DECODER_CHOICES)}")
        expected_decoder = DIT_CHOICES[dit]["decoder"]
        if decoder != expected_decoder:
            raise ValueError(
                f"dit {dit!r} bundles the {expected_decoder!r} autoencoder, not {decoder!r}; "
                f"the torch backend can't mix them"
            )

        self.dit_name = dit
        self.decoder_name = decoder
        self.seconds = float(seconds)
        self.verbose = verbose

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # fp16 is meaningless on CPU and errors out in parts of the stack.
        self.model_half = dit_dtype != "fp32" and device != "cpu"

        t0 = time.time()
        model_name = DIT_CHOICES[dit]["model"]
        local_dir = find_local_model_dir(model_name)
        if local_dir is not None:
            self.model = load_local_model(local_dir, device, self.model_half)
            source = f"local {local_dir}"
        else:
            self.model = StableAudioModel.from_pretrained(
                model_name, device=device, model_half=self.model_half
            )
            source = "hub"
        if verbose:
            dtype = "fp16" if self.model_half else "fp32"
            print(
                f"[pipeline] {model_name} + {decoder} on {device} ({dtype}) "
                f"loaded in {time.time() - t0:.2f}s [{source}]",
                file=sys.stderr,
            )

    def set_duration(self, seconds: float) -> None:
        """Retarget the resident pipeline at a new loop length.

        Free here: nothing in the torch model is sized to the duration —
        `seconds` only picks the latent length inside each generate() call, and
        StableAudioModel.generate() derives that per call anyway. The MLX
        backend deliberately has no equivalent, because its DiT allocates
        `_local_zeros_1` against T_lat at construction; callers that must
        support both should treat a missing set_duration as "rebuild instead".
        """
        self.seconds = float(seconds)

    def _validate_inpaint_range(self, inpaint_seconds: tuple[float, float]) -> tuple[float, float]:
        s, e = (float(x) for x in inpaint_seconds)
        if not (0 <= s < e <= self.seconds + 1e-6):
            raise ValueError(
                f"inpaint range {s:.3f}-{e:.3f}s outside [0, {self.seconds:.3f}]"
            )
        return s, min(e, self.seconds)

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
        **generate_kwargs,
    ) -> np.ndarray:
        """Extra kwargs pass straight through to StableAudioModel.generate() —
        `duration_padding_sec`, `chunked_decode`, `dist_shift`, `sampler_type`.
        The MLX backend has no equivalent, so the harness never sets them; they
        exist for tuning this backend against its own defaults."""
        sigma_max = float(init_noise_level)
        if sigma_max < MIN_SIGMA:
            raise ValueError(
                f"init_noise_level={sigma_max} < {MIN_SIGMA}: rf_denoiser is undefined at t≈0"
            )

        kwargs: dict = {
            "prompt": prompt,
            "duration": self.seconds,
            "steps": steps,
            "cfg_scale": cfg,
            "apg_scale": apg,
            "seed": seed,
            # Upstream defaults to 6s of headroom, which makes an 8s loop denoise
            # ~14s of latents — 2.9x the work (2.80s -> 0.96s per candidate on a
            # 5070 Ti when dropped). Sizing the latent to the requested duration
            # is also what the MLX backend does, so this keeps the two backends
            # comparable rather than diverging from them. Measured across seeds:
            # same correlation-with-source band, no tail-energy collapse.
            # Override per call if a preset ever wants the headroom back.
            "duration_padding_sec": 0.0,
        }
        # cfg=1.0 disables the unconditional branch entirely, so a negative
        # prompt would be silently ignored — skip it rather than pay for it.
        if cfg != 1.0 and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        init_t = None
        if init_audio_np is not None:
            # Pass a tensor, not the raw array: numpy_audio_to_tensor() treats 2D
            # input as (samples, channels) and would transpose our (2, T).
            init_t = torch.from_numpy(np.ascontiguousarray(init_audio_np, dtype=np.float32))

        if inpaint_range_seconds is not None:
            if init_t is None:
                raise ValueError("inpaint_range_seconds requires init_audio_np")
            s, e = self._validate_inpaint_range(inpaint_range_seconds)
            kwargs["inpaint_audio"] = (SAMPLE_RATE, init_t)
            kwargs["inpaint_mask_start_seconds"] = s
            kwargs["inpaint_mask_end_seconds"] = e
            # sample_diffusion only honours sigma_max when init_data is present,
            # so inpaint noise levels below 1.0 need init_audio passed too. This
            # mixes a (1 - sigma_max) trace of the source into the starting
            # latent, where the MLX backend starts from pure noise and relies on
            # per-step paste-back instead — see README_VARIATIONS "inpaint noise".
            kwargs["init_audio"] = (SAMPLE_RATE, init_t)
            kwargs["init_noise_level"] = sigma_max
        elif init_t is not None:
            kwargs["init_audio"] = (SAMPLE_RATE, init_t)
            kwargs["init_noise_level"] = sigma_max

        # Only variations need the correction — at sigma_max 1.0 the shift is
        # already operating on a unit ramp. Skip it if a caller supplied its own
        # dist_shift, since that's an explicit override.
        if sigma_max < 1.0 and "dist_shift" not in generate_kwargs:
            kwargs["dist_shift"] = _NormalisedShift(
                self.model.model.sampling_dist_shift, sigma_max)

        kwargs.update(generate_kwargs)
        audio = self.model.generate(**kwargs)
        audio_np = audio[0].to(torch.float32).cpu().numpy()

        requested_samples = int(round(self.seconds * SAMPLE_RATE))
        if audio_np.shape[-1] > requested_samples:
            audio_np = audio_np[..., :requested_samples]

        if out_path is not None:
            save_wav(str(out_path), audio_np)
        return audio_np
