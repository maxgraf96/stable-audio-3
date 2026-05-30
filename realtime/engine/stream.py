"""StreamPipeline — a StreamDiffusion-style ring buffer for Stable Audio 3.

A fixed-depth ring of in-flight generations, each at a different denoising step,
advanced together by ONE batched DiT forward per tick(). Adapted from DEMON's
acestep/engine/stream.py to SA3's rectified-flow DiT and [B, 256, T] latents.

Phase 1 scope (the parity-gated core):
  - SlotRequest (immutable params) / _Slot (mutable runtime) split
  - submit() -> queue ;  tick() -> harvest finished, admit, advance all active
  - heterogeneous per-row timesteps: each slot draws its own schedule[step_idx]
  - denoise(=sigma_max)-keyed schedule cache
  - deterministic euler step (the engine default)
  - set_depth() hot-resize

Out of scope until Phase 2: per-frame curves, shared-mutable overrides, CFG/RCFG,
x0-target, inpaint masks, multi-condition. Hooks are kept narrow so they slot in.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

# build_pingpong_schedule lives in the MLX defs (shared with the product runner).
_MLX_ROOT = Path(__file__).resolve().parents[2] / "optimized" / "mlx"
if str(_MLX_ROOT) not in sys.path:
    sys.path.insert(0, str(_MLX_ROOT))
from models.defs.sa3_pipeline import build_pingpong_schedule  # noqa: E402

from .ode_steps import step_euler  # noqa: E402

IO_CHANNELS = 256
MAX_DEPTH = 8


@dataclass
class SlotRequest:
    """Immutable admission ticket: everything needed to start one generation.

    Phase 2 will extend this with per-frame curves, cfg/rcfg config, x0_target,
    and a latent mask. For now: conditioning + seed + denoise + optional source.
    """
    cross_attn: mx.array            # [1, 257, 768]  text(256) + seconds(1) tokens
    global_cond: mx.array           # [1, 768]       adaLN global (seconds embed)
    seed: int
    denoise: float = 0.5            # sigma_max: init noise level AND schedule start
    source_latent: Optional[mx.array] = None   # [1, 256, T]; None = pure text2audio


@dataclass
class _Slot:
    """Mutable runtime cell: the evolving latent + where it is in its schedule."""
    request: SlotRequest
    xt: mx.array                    # [1, 256, T] current noisy latent
    schedule: mx.array             # [S+1] fp32 sigmas, baked from request.denoise
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
        self._sched_cache: dict[float, mx.array] = {}

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def num_active(self) -> int:
        return sum(1 for s in self._slots if s is not None)

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
        """Enqueue a request. Drains into an empty slot inside tick()."""
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
        return _Slot(request=req, xt=xt, schedule=sched, step_idx=0)

    # ---- the tick ----
    def tick(self) -> Optional[mx.array]:
        """Advance every active slot one denoising step. Returns one finished
        latent [1, 256, T] this tick, or None during warmup / when none finish."""
        # 1. harvest one finished slot
        finished: Optional[mx.array] = None
        for i, s in enumerate(self._slots):
            if s is not None and s.finished:
                finished = s.xt
                self._slots[i] = None
                break
        # 2. admit queued requests into empty slots
        for i, s in enumerate(self._slots):
            if s is None and self._queue:
                self._slots[i] = self._init_slot(self._queue.pop(0))
        # 3. advance all active (not-yet-finished) slots in one batched forward
        active = [s for s in self._slots if s is not None and not s.finished]
        if active:
            self._advance(active)
        return finished

    def _advance(self, slots: List[_Slot]) -> None:
        xb = mx.concatenate([s.xt for s in slots], axis=0)                  # [B,256,T]
        cb = mx.concatenate([s.request.cross_attn for s in slots], axis=0)  # [B,257,768]
        gb = mx.concatenate([s.request.global_cond for s in slots], axis=0)  # [B,768]
        # heterogeneous per-row timesteps: each row draws its own schedule[step]
        t_curr = mx.stack([s.schedule[s.step_idx] for s in slots])          # [B] fp32
        t_next = mx.stack([s.schedule[s.step_idx + 1] for s in slots])      # [B] fp32
        v = self.dit(xb, t_curr, cb, gb)                                    # [B,256,T]
        xb_new = step_euler(xb, v, t_curr, t_next)
        mx.eval(xb_new)
        for j, s in enumerate(slots):
            s.xt = xb_new[j:j + 1]
            s.step_idx += 1

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
