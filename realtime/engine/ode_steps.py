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


def normalize_curve(value, dtype=mx.float16) -> mx.array:
    """Canonicalize a scalar-or-[T] per-frame control into a broadcastable curve.

    SA3 latents are [B, 256, T] (channels-first), so a per-frame curve modulates
    along the LAST axis and broadcasts over channels -> shape [., 1, T] (or the
    [1,1,1] scalar). (DEMON's [B,T,1] equivalent is transposed for our layout.)

      float/int/bool -> [1, 1, 1]
      [T]            -> [1, 1, T]
      [B, T]         -> [B, 1, T]
      [., 1, T]      -> unchanged
    """
    if isinstance(value, (int, float, bool)):
        return mx.full((1, 1, 1), float(value), dtype=dtype)
    v = value if isinstance(value, mx.array) else mx.array(value)
    if v.ndim == 0:
        v = v.reshape(1, 1, 1)
    elif v.ndim == 1:
        v = v.reshape(1, 1, -1)
    elif v.ndim == 2:
        v = v.reshape(v.shape[0], 1, v.shape[1])
    return v.astype(dtype)


def step_sde_curve(xt: mx.array, x0_pred: mx.array, t_next: mx.array,
                   sdc: mx.array, source_latents: mx.array, noise: mx.array) -> mx.array:
    """SDE re-noise step with per-frame source blending (DEMON contribution 3).

    Per frame, blend between re-noising toward the model's x0 prediction (sdc=1,
    standard SDE -> explore/regenerate) and re-noising toward the source latents
    (sdc=0 -> anchor/preserve). This is the per-step source anchor SA3's a2a
    lacks (its init mix happens once, never re-anchors).

      xt, x0_pred, source_latents, noise : [B, 256, T]
      t_next : [B]   (per-row, reshaped to [B,1,1])
      sdc    : [., 1, T]   per-frame blend in [0, 1]

    At the final step (t_next=0): sdc=0 -> exactly source_latents; sdc=1 -> x0_pred.
    """
    tn = t_next.reshape(-1, 1, 1).astype(xt.dtype)
    xt_full = tn * noise + (1.0 - tn) * x0_pred
    xt_source = tn * noise + (1.0 - tn) * source_latents
    return sdc * xt_full + (1.0 - sdc) * xt_source


def blend_x0_target(x0: mx.array, x0_target: mx.array, alpha: mx.array) -> mx.array:
    """Per-frame morph of the x0 estimate toward an independent target latent.
    alpha [., 1, T] in [0, 1]: 0 keeps x0, 1 collapses to the target."""
    return (1.0 - alpha) * x0 + alpha * x0_target


def apg_cfg(xt: mx.array, cond_v: mx.array, uncond_v: mx.array, t: mx.array,
            cfg, apg: float = 1.0) -> mx.array:
    """Classifier-free guidance with Adaptive Projected Guidance, in velocity space.

    Bit-matches the product reference (optimized/mlx/scripts/sa3_mlx.py model_fn):
    work in denoised space d = x - t*v (fp32), project the cond-uncond difference
    orthogonal to cond_d (APG), recombine, convert back to velocity.

      xt, cond_v, uncond_v : [B, 256, T]
      t   : [B]               per-row timestep (sigma); reshaped to [B,1,1]
      cfg : float OR [.,1,T]  guidance scale (scalar matches reference; per-frame
                              generalizes it to a guidance_curve)
      apg : float in [0,1]    1.0 = full orthogonal projection, 0.0 = vanilla CFG

    Reduction axes (-2,-1) = over channels AND frames per batch row, matching the
    reference's sum over (C, T).
    """
    sigma = t.reshape(-1, 1, 1).astype(mx.float32)
    x = xt.astype(mx.float32)
    cond_d = x - cond_v.astype(mx.float32) * sigma
    uncond_d = x - uncond_v.astype(mx.float32) * sigma
    diff = cond_d - uncond_d
    if apg <= 0.0:
        cfg_diff = diff
    else:
        norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
        unit = cond_d / mx.maximum(norm, 1e-8)
        parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
        diff_orth = diff - parallel
        cfg_diff = diff_orth if apg >= 1.0 else (apg * diff_orth + (1.0 - apg) * diff)
    cfg_scale = cfg if not isinstance(cfg, mx.array) else cfg.astype(mx.float32)
    cfg_d = cond_d + (cfg_scale - 1.0) * cfg_diff
    cfg_v = (x - cfg_d) / sigma
    return cfg_v.astype(xt.dtype)


def step_euler_x0(xt: mx.array, x0: mx.array, t_curr: mx.array, t_next: mx.array) -> mx.array:
    """Euler step expressed through the (possibly modified) x0 estimate.

    Algebraically identical to step_euler with v=(xt-x0)/t_curr, written as a
    stable lerp:  x_next = r*xt + (1-r)*x0,  r = t_next/t_curr in [0, 1). Used
    whenever x0 is modified (x0-target morph). NOT bit-exact to the fast
    step_euler path (the x0 round-trip perturbs fp16 at ~0.25 level, per DEMON),
    so plain slots keep using step_euler.
    """
    r = (t_next / t_curr).reshape(-1, 1, 1).astype(xt.dtype)
    return r * xt + (1.0 - r) * x0
