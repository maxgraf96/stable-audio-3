"""StreamPipeline — a StreamDiffusion-style ring buffer for Stable Audio 3.

A fixed-depth ring of in-flight generations, each at a different denoising step,
advanced together by ONE batched DiT forward per tick(). Adapted from DEMON's
acestep/engine/stream.py to SA3's rectified-flow DiT and [B, 256, T] latents.

Phase 1 (parity-gated core): SlotRequest/_Slot split, submit()/tick(), per-slot
baked schedules + denoise cache, heterogeneous per-row timesteps, deterministic
euler step, set_depth().

Phase 2a: per-frame curves resolved each tick, per-slot OR overridden globally via
set_shared_curve() (1-tick onset): velocity_scale, sde_denoise_curve (per-frame SDE
source-blend, 1=explore / 0=anchor to source).
Phase 2b: x0_target morph (blend x0 toward a banked target, refinement-half gated).
Phase 2c: CFG + APG + RCFG. A slot with cfg!=1 also needs an unconditional velocity.
  - "full"  : uncond forward every step (an extra batch row)
  - "self"  : RCFG — the slot's initial noise stands in as the virtual uncond
              velocity (flow matching: v_uncond ~= noise when x0_uncond ~= 0), zero
              extra forwards
The positive forward batches ALL active slots; the negative forward batches only
the slots that need an uncond pass this step. APG/CFG recombine per slot.

Still to come: inpaint masks; prompt A<->B blend is caller-side (no engine change).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

_MLX_ROOT = Path(__file__).resolve().parents[2] / "optimized" / "mlx"
if str(_MLX_ROOT) not in sys.path:
    sys.path.insert(0, str(_MLX_ROOT))
from models.defs.sa3_pipeline import build_pingpong_schedule  # noqa: E402

from .ode_steps import (step_euler, step_sde_curve, step_euler_x0,  # noqa: E402
                        x0_from_vel, blend_x0_target, normalize_curve, apg_cfg)

IO_CHANNELS = 256
MAX_DEPTH = 8

# Per-frame curve fields settable per-slot (on SlotRequest) or overridden globally
# via set_shared_curve(). Resolved each tick by _eff_curve() (shared wins).
CURVE_NAMES = ("velocity_scale", "sde_denoise_curve", "x0_target_strength")


@dataclass
class SlotRequest:
    """Immutable admission ticket: everything needed to start one generation."""
    cross_attn: mx.array            # [1, 257, 768]  text(256) + seconds(1) tokens
    global_cond: mx.array           # [1, 768]       adaLN global (seconds embed)
    seed: int
    denoise: float = 0.5            # sigma_max: init noise level AND schedule start
    source_latent: Optional[mx.array] = None   # [1, 256, T]; None = pure text2audio
    # Per-frame curves: python scalar, [T] mx.array, or [., 1, T]. None = inactive.
    velocity_scale: Any = None
    sde_denoise_curve: Any = None
    # x0-target morph: blend the x0 estimate toward an independent target latent,
    # gated to the refinement (back) half of the schedule. x0_target is [1,256,T].
    x0_target: Optional[mx.array] = None
    x0_target_strength: Any = None
    # CFG: cfg==1.0 disables guidance (single forward, the rf_denoiser default).
    # cfg!=1.0 needs an unconditional velocity, sourced per rcfg_mode:
    #   "full" -> uncond forward each step using neg_cross_attn (an extra batch row)
    #   "self" -> RCFG, the slot's initial noise as virtual uncond (no extra forward)
    cfg: float = 1.0
    apg: float = 1.0
    rcfg_mode: str = "full"                      # "full" | "self"
    neg_cross_attn: Optional[mx.array] = None    # [1,257,768]; None+full -> zeros

    @property
    def has_cfg(self) -> bool:
        return self.cfg != 1.0

    @property
    def needs_neg_forward(self) -> bool:
        return self.has_cfg and self.rcfg_mode == "full"


@dataclass
class _Slot:
    """Mutable runtime cell: the evolving latent + where it is in its schedule."""
    request: SlotRequest
    xt: mx.array                    # [1, 256, T] current noisy latent
    schedule: mx.array             # [S+1] fp32 sigmas, baked from request.denoise
    rng_key: mx.array              # per-slot key for SDE re-noise (deterministic)
    curves: Dict[str, mx.array]    # pre-normalized per-slot curves [., 1, T]
    initial_noise: Optional[mx.array] = None   # RCFG-self virtual uncond velocity
    step_idx: int = 0

    @property
    def finished(self) -> bool:
        return self.step_idx >= self.schedule.shape[0] - 1


class StreamPipeline:
    def __init__(self, dit, T_lat: int, *, steps: int = 8, depth: int = 4,
                 dtype=mx.float16):
        self.dit = dit
        self.T = int(T_lat)
        self.steps = int(steps)
        self.dtype = dtype
        self._depth = max(1, min(int(depth), MAX_DEPTH))
        self._slots: List[Optional[_Slot]] = [None] * self._depth
        self._queue: List[SlotRequest] = []
        self._sched_cache: Dict[float, mx.array] = {}
        self._shared_curves: Dict[str, mx.array] = {}

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def num_active(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    # ---- shared-mutable per-frame curves (1-tick onset on all in-flight slots) ----
    def set_shared_curve(self, name: str, value) -> None:
        """Override a per-frame curve globally. Effective on the very next tick for
        every in-flight slot (read fresh each step). Pass None to revert to
        per-slot behavior. Precedence: a shared curve beats the slot's own field."""
        if name not in CURVE_NAMES:
            raise ValueError(f"unknown curve {name!r}; expected one of {CURVE_NAMES}")
        if value is None:
            self._shared_curves.pop(name, None)
        else:
            self._shared_curves[name] = normalize_curve(value, dtype=self.dtype)

    def _eff_curve(self, slot: _Slot, name: str) -> Optional[mx.array]:
        v = self._shared_curves.get(name)
        if v is None:
            v = slot.curves.get(name)
        return v

    # ---- schedule cache (keyed by denoise == sigma_max) ----
    def _get_schedule(self, denoise: float) -> mx.array:
        key = round(float(denoise), 6)
        sched = self._sched_cache.get(key)
        if sched is None:
            sched = build_pingpong_schedule(self.steps, sigma_max=key,
                                            use_logsnr_shift=True)
            mx.eval(sched)
            self._sched_cache[key] = sched
        return sched

    # ---- admission ----
    def submit(self, req: SlotRequest) -> None:
        if len(self._queue) >= self._depth:
            self._queue.pop(0)             # drop-oldest backpressure (matches DEMON)
        self._queue.append(req)

    def _init_slot(self, req: SlotRequest) -> _Slot:
        sched = self._get_schedule(req.denoise)
        noise = mx.random.normal((1, IO_CHANNELS, self.T), dtype=self.dtype,
                                 key=mx.random.key(req.seed))
        if req.source_latent is not None:
            s = float(req.denoise)         # a2a init mix: latent*(1-s) + noise*s
            xt = req.source_latent.astype(self.dtype) * (1.0 - s) + noise * s
        else:
            xt = noise
        mx.eval(xt)
        curves = {n: normalize_curve(getattr(req, n), dtype=self.dtype)
                  for n in CURVE_NAMES if getattr(req, n, None) is not None}
        # RCFG-self uses the slot's PURE init noise as the virtual uncond velocity
        # (flow matching: v_uncond ~= noise when the unconditional x0 ~= 0).
        init_noise = noise if (req.has_cfg and req.rcfg_mode == "self") else None
        # rng_key offset from seed so SDE noise is independent of the init noise
        # (which uses key(seed) directly — preserves Phase-1 euler bit-parity).
        return _Slot(request=req, xt=xt, schedule=sched,
                     rng_key=mx.random.key(req.seed + 1), curves=curves,
                     initial_noise=init_noise)

    # ---- the tick ----
    def tick(self) -> Optional[mx.array]:
        """Advance every active slot one denoising step. Returns one finished
        latent [1, 256, T] this tick, or None during warmup / when none finish."""
        finished: Optional[mx.array] = None
        for i, s in enumerate(self._slots):
            if s is not None and s.finished:
                finished = s.xt
                self._slots[i] = None
                break
        for i, s in enumerate(self._slots):
            if s is None and self._queue:
                self._slots[i] = self._init_slot(self._queue.pop(0))
        active = [s for s in self._slots if s is not None and not s.finished]
        if active:
            self._advance(active)
        return finished

    def _advance(self, slots: List[_Slot]) -> None:
        # POSITIVE forward: ONE batched DiT pass across all active slots ...
        xb = mx.concatenate([s.xt for s in slots], axis=0)                  # [B,256,T]
        cb = mx.concatenate([s.request.cross_attn for s in slots], axis=0)  # [B,257,768]
        gb = mx.concatenate([s.request.global_cond for s in slots], axis=0)  # [B,768]
        t_curr = mx.stack([s.schedule[s.step_idx] for s in slots])          # [B] fp32
        t_next = mx.stack([s.schedule[s.step_idx + 1] for s in slots])      # [B] fp32
        v = self.dit(xb, t_curr, cb, gb)                                    # [B,256,T]

        # NEGATIVE forward: batch only the slots that need an uncond pass this step
        # (cfg!=1 and rcfg_mode=="full"). RCFG-self needs no forward.
        neg_idx = [j for j, s in enumerate(slots) if s.request.needs_neg_forward]
        v_neg = None
        if neg_idx:
            def neg_cross(s):
                return s.request.neg_cross_attn if s.request.neg_cross_attn is not None \
                    else mx.zeros_like(s.request.cross_attn)
            xn = mx.concatenate([slots[j].xt for j in neg_idx], axis=0)
            cn = mx.concatenate([neg_cross(slots[j]) for j in neg_idx], axis=0)
            gn = mx.concatenate([slots[j].request.global_cond for j in neg_idx], axis=0)
            tn_b = mx.stack([t_curr[j] for j in neg_idx])
            v_neg = self.dit(xn, tn_b, cn, gn)
        neg_pos = {j: k for k, j in enumerate(neg_idx)}   # slot j -> row in v_neg

        # ... then a cheap per-slot integration loop (heterogeneous guidance/curves).
        half = self.steps // 2          # x0-target gated to the refinement half
        for j, s in enumerate(slots):
            vj = v[j:j + 1]
            tc = t_curr[j:j + 1]
            # --- guidance (CFG + APG) before any curve scaling ---
            if s.request.has_cfg:
                if s.request.rcfg_mode == "self":
                    uncond_v = s.initial_noise
                else:
                    uncond_v = v_neg[neg_pos[j]:neg_pos[j] + 1]
                vj = apg_cfg(s.xt, vj, uncond_v, tc, s.request.cfg, s.request.apg)
            vs = self._eff_curve(s, "velocity_scale")
            if vs is not None:
                vj = vj * vs
            tn = t_next[j:j + 1]
            sdc = self._eff_curve(s, "sde_denoise_curve")
            alpha = self._eff_curve(s, "x0_target_strength")
            use_sde = sdc is not None and s.request.source_latent is not None
            morph = (alpha is not None and s.request.x0_target is not None
                     and s.step_idx >= half)
            if use_sde or morph:
                x0 = x0_from_vel(s.xt, vj, tc)
                if morph:
                    x0 = blend_x0_target(x0, s.request.x0_target.astype(self.dtype), alpha)
                if use_sde:
                    s.rng_key, sub = mx.random.split(s.rng_key)
                    noise = mx.random.normal(s.xt.shape, dtype=self.dtype, key=sub)
                    s.xt = step_sde_curve(s.xt, x0, tn, sdc,
                                          s.request.source_latent.astype(self.dtype), noise)
                else:
                    s.xt = step_euler_x0(s.xt, x0, tc, tn)
            else:
                s.xt = step_euler(s.xt, vj, tc, tn)
            s.step_idx += 1
        mx.eval(*[s.xt for s in slots])

    # ---- hot depth resize (in-flight slots survive) ----
    def set_depth(self, depth: int) -> None:
        depth = max(1, min(int(depth), MAX_DEPTH))
        if depth == self._depth:
            return
        if depth < self._depth:
            self._slots = self._slots[:depth]          # shrink: truncate tail
        else:
            self._slots = self._slots + [None] * (depth - self._depth)  # grow: append
        self._depth = depth
