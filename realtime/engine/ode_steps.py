"""Pure per-step solver primitives for the streaming engine.

Stateless, model-agnostic functions over SA3's [B, 256, T] channels-first
latents. Mirrors DEMON's acestep/engine/ode_steps.py, adapted to SA3's layout
(per-frame curves broadcast as [B, 1, T] here, vs DEMON's [B, T, 1]).

Rectified flow / rf_denoiser convention: t == sigma in [0, 1], t=1 is noise,
the network predicts velocity v, and x0 = xt - t*v ; x_t = (1-t)*x0 + t*noise.

Phase 1 ships the two euler-path primitives. The SDE source-blend curve,
normalize_curve, and the branch-free sentinels arrive in Phase 2.
"""

from __future__ import annotations

import mlx.core as mx


def x0_from_vel(xt: mx.array, vt: mx.array, t: mx.array) -> mx.array:
    """Flow-matching x0 estimate: x0 = xt - t*v.

    xt, vt : [B, 256, T]   ;   t : [B] (per-row timestep) -> broadcast [B,1,1].
    """
    return xt - vt * t.reshape(-1, 1, 1)


def step_euler(xt: mx.array, vt: mx.array, t_curr: mx.array, t_next: mx.array) -> mx.array:
    """Deterministic rectified-flow Euler step: x = x + (t_next - t_curr) * v.

    t_next < t_curr (schedule descends sigma_max -> 0), so x moves toward x0.
    At the final step (t_next=0) this is x - t_curr*v = the x0 estimate.

    Bit-matches models.defs.sa3_pipeline.sample_flow_euler's per-step update:
    the dt is computed in the schedule's fp32, then cast to xt.dtype before the
    multiply, exactly as the standalone sampler does.

    xt, vt        : [B, 256, T]
    t_curr, t_next: [B]  (per-row, gathered from each slot's own schedule)
    """
    dt = (t_next - t_curr).astype(xt.dtype).reshape(-1, 1, 1)
    return xt + dt * vt
