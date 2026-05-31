"""GeneratorThread — owns all MLX work for the live morph instrument.

MLX binds compute streams to the creating thread, so EVERY MLX call (model
loads, ticks, decode) happens on this one thread. The audio callback never
touches MLX; it only reads the AudioRing this thread fills.

Loop each iteration:
  1. apply pending control changes (denoise, shared curves, prompt) at a tick
     boundary (so GPU-state mutations serialize with the tick, like DEMON)
  2. keep the ring buffer topped up: submit a fresh morph slot whenever there's
     a free slot (continuous regeneration)
  3. tick() one denoising step across all in-flight slots
  4. on a finished latent: MSE-skip check, else decode -> write_loop into AudioRing

Free-running: no sleep on the hot path; cadence is set by tick+decode time. A
tiny sleep only when fully idle (nothing to do) to avoid a busy-spin.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx

REPO = Path(__file__).resolve().parents[2]
MLX_ROOT = REPO / "optimized" / "mlx"
for p in (REPO / "realtime", MLX_ROOT, MLX_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sa3_mlx import (  # noqa: E402
    patch_audio, load_dit, load_decoder, load_encoder,
    SAMPLE_RATE, SAMPLES_PER_LATENT,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    load_conditioner_from_npz, apply_prompt_padding, patched_decode,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from engine.stream import StreamPipeline, SlotRequest  # noqa: E402

DIT_NAME = "sm-music"
DECODER = "same-s"
CYCLIC_PAD = 8       # latent frames of wrap-around context for seamless looping
                     # (>= SAME-S receptive field; even keeps T+2*pad even)


def cyclic_decode(model, chunk_fn, latents, chunk, ovl, pad=CYCLIC_PAD):
    """Decode a latent as a SEAMLESS LOOP. The plain decoder treats the latent as
    a finite sequence, so frame 0 (no left neighbor) and frame T-1 (no right
    neighbor) don't join when the ring wraps. We pad the latent cyclically — take
    `pad` frames from the END onto the front and `pad` from the START onto the
    back — decode, then trim `pad` frames of audio from each side. Now frame 0 was
    decoded with the true loop-end as context and frame T-1 with the true
    loop-start, so the wrap point is continuous. Returns [B, 512, T*16]."""
    head = latents[:, :, -pad:]
    tail = latents[:, :, :pad]
    padded = mx.concatenate([head, latents, tail], axis=2)      # [B, C, T+2*pad]
    Tp = padded.shape[-1]
    kernel = chunk + 2 * ovl
    patches = chunk_fn(model, padded, chunk, ovl) if Tp > kernel else model(padded)
    return patches[:, :, pad * 16:-(pad * 16)]                  # trim wrap context


def plan_lengths(seconds: float, margin_s: float):
    """Interior-windowing length plan, in latent frames (all even for SAME-S).

    SA3 shapes an intro/outro into every generation (it "opens and closes" a
    piece). To avoid that envelope in the loop, we GENERATE a longer clip and
    play only the even interior: gen = loop + 2*margin. The intro lands in the
    leading margin and the outro in the trailing margin, both cropped away.

    Returns (gen_T, margin_T, loop_T)."""
    def even_ceil(s):
        f = max(2, math.ceil(s * SAMPLE_RATE / SAMPLES_PER_LATENT))
        return f + (f % 2)
    loop_T = even_ceil(seconds)
    margin_T = (max(0, round(margin_s * SAMPLE_RATE / SAMPLES_PER_LATENT)) // 2) * 2
    gen_T = loop_T + 2 * margin_T
    return gen_T, margin_T, loop_T


@dataclass
class GenConfig:
    seconds: float = 8.0
    steps: int = 8
    depth: int = 2
    denoise: float = 0.5
    denoise_slew: float = 0.04   # max change in effective denoise per submission.
                                 # Slider jumps ramp over several completions so the
                                 # morph glides instead of snapping between very
                                 # different loops (denoise is the per-request class;
                                 # this is UI-side smoothing, not a 1-tick override).
    prompt: str = ""
    seed_base: int = 1000
    margin_s: float = 0.0        # interior-windowing margin generated on EACH side
                                 # and cropped away. NOTE: measured ineffective on a
                                 # decaying source (the loud->quiet shape is the
                                 # SOURCE loop's own envelope, faithfully reproduced
                                 # by a2a, not SA3 intro/outro), so default off.
                                 # Loop evenness is handled by loop_norm below.
    loop_xfade_s: float = 0.15   # small tail->head fold for the interior seam
                                 # (interior windowing does the heavy lifting now).
    evolve: bool = False         # False = FIXED seed (coherent morph: your controls
                                 # move the loop, not a new random roll each time).
                                 # True = walk the seed each submission ("browse"
                                 # different variations, the old behavior).
    mse_skip: float = 5e-4       # skip decode when consecutive latents barely move
    cyclic: bool = True          # seamless-loop decode (wrap-around context)
    dtype: object = mx.float16


@dataclass
class _Shared:
    """Thread-shared control state. Single lock; the generator reads a snapshot
    at each tick boundary so a control move never tears a generation."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    denoise: float = 0.5
    sde_denoise_curve: object = None       # scalar / np / mx — passed to set_shared_curve
    velocity_scale: object = None
    evolve: bool = False                   # walk the seed each submission?
    new_prompt: Optional[str] = None       # set -> generator re-encodes, then clears
    running: bool = True
    # telemetry
    ticks: int = 0
    completions: int = 0
    decodes: int = 0
    skips: int = 0
    last_tick_ms: float = 0.0
    last_decode_ms: float = 0.0


class GeneratorThread(threading.Thread):
    def __init__(self, ring, source_audio: np.ndarray, cfg: GenConfig):
        super().__init__(daemon=True, name="sa3-generator")
        self.ring = ring
        self.cfg = cfg
        self.shared = _Shared(denoise=cfg.denoise, evolve=cfg.evolve)
        self._src = source_audio              # [2, T_audio] float32
        self._seed = cfg.seed_base
        self._denoise_eff = cfg.denoise       # slew-limited effective denoise
        self._ready = threading.Event()
        # set on the generator thread in _setup():
        self.T = 0
        self.dit = None
        self.dec = None
        self._chunk = self._ovl = 0
        self._crop_start = 0          # interior-window start (samples)
        self._loop_samples = 0        # interior length played back (samples)
        self._init_latents = None
        self._cross = None
        self._gcond = None
        self._pipe: Optional[StreamPipeline] = None
        self._last_latent = None

    # ---- control API (any thread) ----
    def set_denoise(self, v: float):
        with self.shared.lock:
            self.shared.denoise = float(v)

    def set_shared_curve(self, name: str, value):
        with self.shared.lock:
            setattr(self.shared, name, value)

    def set_prompt(self, text: str):
        with self.shared.lock:
            self.shared.new_prompt = text

    def set_evolve(self, on: bool):
        with self.shared.lock:
            self.shared.evolve = bool(on)

    def stop(self):
        with self.shared.lock:
            self.shared.running = False

    def wait_ready(self, timeout=None) -> bool:
        return self._ready.wait(timeout)

    def stats(self) -> dict:
        with self.shared.lock:
            s = self.shared
            return dict(ticks=s.ticks, completions=s.completions, decodes=s.decodes,
                        skips=s.skips, tick_ms=s.last_tick_ms, decode_ms=s.last_decode_ms)

    # ---- model setup (on this thread) ----
    def _setup(self):
        c = self.cfg
        gen_T, margin_T, loop_T = plan_lengths(c.seconds, c.margin_s)
        self.T = gen_T
        self._crop_start = margin_T * SAMPLES_PER_LATENT
        self._loop_samples = loop_T * SAMPLES_PER_LATENT
        gen_samples = gen_T * SAMPLES_PER_LATENT
        gen_seconds = gen_samples / SAMPLE_RATE

        # Clean loop unit (one source period), then TILE it so the generated
        # interior [crop_start : crop_start+loop] aligns exactly to the source and
        # the margins are a musically-continuous continuation (source is a loop).
        ls = self._loop_samples
        loop_unit = (self._src[:, :ls] if self._src.shape[-1] >= ls
                     else np.pad(self._src, ((0, 0), (0, ls - self._src.shape[-1]))))
        idx = (np.arange(gen_samples) - self._crop_start) % ls
        extended = loop_unit[:, idx]                          # [2, gen_samples]

        enc, _ = load_encoder(DECODER, mx.float32)
        self._init_latents = enc(mx.array(patch_audio(extended[None], 256))).astype(c.dtype)
        mx.eval(self._init_latents)
        del enc

        self.dit, ckpt = load_dit(DIT_NAME, gen_T, c.dtype)
        self._t5 = T5Gemma.from_npz(str(MLX_ROOT / "models" / "mlx" / "t5gemma_f16.npz"))
        self._pad_emb, self._secs = load_conditioner_from_npz(ckpt, prefix="cond.")
        # condition on the TRUE generated duration so SA3 shapes its intro/outro at
        # the gen boundaries (which fall in the cropped margins).
        self._secs_e = self._secs(gen_seconds).astype(c.dtype)
        self._encode_prompt(c.prompt)

        self.dec, self._chunk_fn, (self._chunk, self._ovl) = load_decoder(DECODER, mx.float32)
        self._pipe = StreamPipeline(self.dit, self.T, steps=c.steps, depth=c.depth, dtype=c.dtype)

        # warmup: one full generation so first audible loop isn't a cold-start stall
        self._submit_morph()
        for _ in range(c.steps + c.depth + 2):
            lat = self._pipe.tick()
            if lat is not None:
                self.ring.write_loop(self._decode(lat))
                break
        self._ready.set()

    def _encode_prompt(self, text: str):
        e, m = self._t5.encode([text], max_len=256)
        mx.eval(e, m)
        ep = apply_prompt_padding(e.astype(self.cfg.dtype), m, self._pad_emb.astype(self.cfg.dtype))
        self._cross = mx.concatenate([ep, self._secs_e], axis=1)   # [1,257,768]
        self._gcond = self._secs_e[:, 0, :]                        # [1,768]

    def _submit_morph(self):
        with self.shared.lock:
            denoise_target = self.shared.denoise
            sdc = self.shared.sde_denoise_curve
            vs = self.shared.velocity_scale
            evolve = self.shared.evolve
        # Slew the effective denoise toward the target so a slider jump ramps over
        # several completions (gradual morph) rather than snapping.
        slew = self.cfg.denoise_slew
        delta = denoise_target - self._denoise_eff
        self._denoise_eff += max(-slew, min(slew, delta))
        denoise = round(self._denoise_eff, 4)
        # Fixed seed by default -> every loop is the SAME underlying variation, so
        # your controls morph one coherent loop instead of re-rolling a new random
        # one each time (which sounds like browsing radio stations past denoise~0.5).
        # When held steady, consecutive latents are identical -> MSE-skip engages.
        if evolve:
            self._seed += 1
        seed = self._seed if evolve else self.cfg.seed_base
        self._pipe.submit(SlotRequest(
            cross_attn=self._cross, global_cond=self._gcond, seed=seed,
            denoise=denoise, source_latent=self._init_latents,
            sde_denoise_curve=sdc, velocity_scale=vs))

    # ---- decode + interior crop (skip SA3's intro/outro, keep the even body) ----
    def _decode(self, lat) -> np.ndarray:
        latf = lat.astype(mx.float32)
        if self._crop_start == 0 and self.cfg.cyclic and self.T > 2 * CYCLIC_PAD:
            # margin disabled: fall back to cyclic-wrap decode for a seamless loop
            patches = cyclic_decode(self.dec, self._chunk_fn, latf, self._chunk, self._ovl)
        else:
            kernel = self._chunk + 2 * self._ovl
            patches = (self._chunk_fn(self.dec, latf, self._chunk, self._ovl)
                       if self.T > kernel else self.dec(latf))
        out = np.array(patched_decode(patches, 256, 2).astype(mx.float32))[0]  # [2, gen*4096]
        s = self._crop_start
        return out[:, s:s + self._loop_samples]               # interior body only

    def _apply_shared_curves(self):
        with self.shared.lock:
            sdc = self.shared.sde_denoise_curve
            vs = self.shared.velocity_scale
            new_prompt = self.shared.new_prompt
            self.shared.new_prompt = None
        # Shared-mutable overrides hit all in-flight slots next tick (1-tick onset),
        # so denoise/curve moves morph the in-flight loop instead of only affecting
        # newly-submitted slots — the responsive path from Phase 2.
        self._pipe.set_shared_curve("sde_denoise_curve", sdc)
        self._pipe.set_shared_curve("velocity_scale", vs)
        if new_prompt is not None:
            self._encode_prompt(new_prompt)

    # ---- the loop ----
    def run(self):
        self._setup()
        while True:
            with self.shared.lock:
                if not self.shared.running:
                    break
            self._apply_shared_curves()
            # keep the ring full: top up free slots with fresh morphs
            while (self._pipe.num_active + len(self._pipe._queue)) < self._pipe.depth:
                self._submit_morph()

            t0 = time.perf_counter()
            lat = self._pipe.tick()
            tick_ms = (time.perf_counter() - t0) * 1000.0

            dec_ms = 0.0
            did_decode = False
            if lat is not None:
                skip = False
                if self._last_latent is not None and self.cfg.mse_skip > 0:
                    mse = float(((lat - self._last_latent) ** 2).mean())
                    skip = mse < self.cfg.mse_skip
                self._last_latent = lat
                if not skip:
                    t1 = time.perf_counter()
                    self.ring.write_loop(self._decode(lat))
                    dec_ms = (time.perf_counter() - t1) * 1000.0
                    did_decode = True

            with self.shared.lock:
                s = self.shared
                s.ticks += 1
                s.last_tick_ms = tick_ms
                if lat is not None:
                    s.completions += 1
                    if did_decode:
                        s.decodes += 1
                        s.last_decode_ms = dec_ms
                    else:
                        s.skips += 1
