// Pure per-step solver primitives for the C++ streaming engine.
//
// Stateless, model-agnostic functions over SA3's [B, 256, T] channels-first
// latents. Direct port of realtime/engine/ode_steps.py — same mlx::core
// primitives, so bit-exact at fp32 and within fp16 rounding at fp16.
//
// Rectified flow: t == sigma in [0, 1], t=1 is noise, the network predicts
// velocity v, and x0 = xt - t*v ; x_t = (1-t)*x0 + t*noise.
//
// Per-frame curves broadcast as [., 1, T] (channels-first), the transpose of
// DEMON's [B, T, 1].
#pragma once

#include <variant>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

namespace sa3 {
namespace stream {

namespace mx = mlx::core;

// A per-frame control: either a python-scalar-equivalent float, or a [.,1,T]
// (or [T]/[B,T]) tensor. normalize_curve() canonicalizes both to [.,1,T].
using CurveValue = std::variant<float, mx::array>;

// x0 = xt - t*v.  xt, vt: [B,256,T] ; t: [B] -> broadcast [B,1,1].
mx::array x0_from_vel(const mx::array& xt, const mx::array& vt, const mx::array& t);

// Deterministic euler: x = x + (t_next - t_curr) * v.  dt cast to xt.dtype.
mx::array step_euler(const mx::array& xt, const mx::array& vt,
                     const mx::array& t_curr, const mx::array& t_next);

// Canonicalize a scalar-or-tensor curve to a broadcastable [.,1,T] (or [1,1,1]).
mx::array normalize_curve(const CurveValue& value, mx::Dtype dtype);
mx::array normalize_curve(float value, mx::Dtype dtype);
mx::array normalize_curve(const mx::array& value, mx::Dtype dtype);

// SDE re-noise with per-frame source blending. sdc=1 -> standard SDE (explore),
// sdc=0 -> anchor to source_latents. tn = t_next reshaped [B,1,1].
mx::array step_sde_curve(const mx::array& xt, const mx::array& x0_pred,
                         const mx::array& t_next, const mx::array& sdc,
                         const mx::array& source_latents, const mx::array& noise);

// Per-frame morph of x0 toward an independent target: (1-a)*x0 + a*x0_target.
mx::array blend_x0_target(const mx::array& x0, const mx::array& x0_target,
                          const mx::array& alpha);

// CFG + APG in velocity space (fp32 internals). cfg is a scalar OR a [.,1,T]
// guidance curve. Matches sa3_mlx.py model_fn / ode_steps.apg_cfg.
mx::array apg_cfg(const mx::array& xt, const mx::array& cond_v,
                  const mx::array& uncond_v, const mx::array& t,
                  const CurveValue& cfg, float apg = 1.0f);

// Euler via the (possibly modified) x0 estimate: x_next = r*xt + (1-r)*x0,
// r = t_next/t_curr. Used when x0 is modified (x0-target morph).
mx::array step_euler_x0(const mx::array& xt, const mx::array& x0,
                        const mx::array& t_curr, const mx::array& t_next);

}  // namespace stream
}  // namespace sa3
