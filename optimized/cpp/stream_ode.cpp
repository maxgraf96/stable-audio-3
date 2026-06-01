// Per-step solver primitives — implementation. Port of ode_steps.py.
#include "stream_ode.h"

#include <mlx/mlx.h>

namespace sa3 {
namespace stream {

// Reshape a per-row [B] timestep to [B,1,1] for broadcasting against [B,256,T].
static mx::array as_b11(const mx::array& t) {
    return mx::reshape(t, {-1, 1, 1});
}

mx::array x0_from_vel(const mx::array& xt, const mx::array& vt, const mx::array& t) {
    return xt - vt * mx::astype(as_b11(t), xt.dtype());
}

mx::array step_euler(const mx::array& xt, const mx::array& vt,
                     const mx::array& t_curr, const mx::array& t_next) {
    // dt computed in the schedule's fp32, then cast to xt.dtype before multiply
    // (matches sample_flow_euler exactly).
    mx::array dt = mx::astype(as_b11(t_next - t_curr), xt.dtype());
    return xt + dt * vt;
}

mx::array normalize_curve(float value, mx::Dtype dtype) {
    return mx::full({1, 1, 1}, mx::array(value), dtype);
}

mx::array normalize_curve(const mx::array& value, mx::Dtype dtype) {
    mx::array v = value;
    const int nd = static_cast<int>(v.ndim());
    if (nd == 0) {
        v = mx::reshape(v, {1, 1, 1});
    } else if (nd == 1) {
        v = mx::reshape(v, {1, 1, -1});
    } else if (nd == 2) {
        v = mx::reshape(v, {v.shape()[0], 1, v.shape()[1]});
    }  // nd >= 3: assumed already [.,1,T]
    return mx::astype(v, dtype);
}

mx::array normalize_curve(const CurveValue& value, mx::Dtype dtype) {
    if (std::holds_alternative<float>(value)) {
        return normalize_curve(std::get<float>(value), dtype);
    }
    return normalize_curve(std::get<mx::array>(value), dtype);
}

mx::array step_sde_curve(const mx::array& xt, const mx::array& x0_pred,
                         const mx::array& t_next, const mx::array& sdc,
                         const mx::array& source_latents, const mx::array& noise) {
    const mx::Dtype dt = xt.dtype();
    const mx::array one = mx::array(1.0f, dt);
    mx::array tn = mx::astype(as_b11(t_next), dt);
    mx::array xt_full   = tn * noise + (one - tn) * x0_pred;
    mx::array xt_source = tn * noise + (one - tn) * source_latents;
    return sdc * xt_full + (one - sdc) * xt_source;
}

mx::array blend_x0_target(const mx::array& x0, const mx::array& x0_target,
                          const mx::array& alpha) {
    const mx::array one = mx::array(1.0f, x0.dtype());
    return (one - alpha) * x0 + alpha * x0_target;
}

mx::array apg_cfg(const mx::array& xt, const mx::array& cond_v,
                  const mx::array& uncond_v, const mx::array& t,
                  const CurveValue& cfg, float apg) {
    mx::array sigma = mx::astype(as_b11(t), mx::float32);
    mx::array x = mx::astype(xt, mx::float32);
    mx::array cond_d   = x - mx::astype(cond_v,   mx::float32) * sigma;
    mx::array uncond_d = x - mx::astype(uncond_v, mx::float32) * sigma;
    mx::array diff = cond_d - uncond_d;

    mx::array cfg_diff = diff;
    if (apg > 0.0f) {
        mx::array norm = mx::sqrt(mx::sum(cond_d * cond_d,
                                         std::vector<int>{-2, -1}, /*keepdims=*/true));
        mx::array unit = cond_d / mx::maximum(norm, mx::array(1e-8f));
        mx::array parallel =
            mx::sum(diff * unit, std::vector<int>{-2, -1}, true) * unit;
        mx::array diff_orth = diff - parallel;
        cfg_diff = (apg >= 1.0f)
            ? diff_orth
            : (mx::array(apg) * diff_orth + mx::array(1.0f - apg) * diff);
    }

    // cfg scale: scalar or [.,1,T] curve (already at curve dtype) -> fp32.
    mx::array cfg_scale = std::holds_alternative<float>(cfg)
        ? mx::array(std::get<float>(cfg))
        : mx::astype(std::get<mx::array>(cfg), mx::float32);
    mx::array cfg_d = cond_d + (cfg_scale - mx::array(1.0f)) * cfg_diff;
    mx::array cfg_v = (x - cfg_d) / sigma;
    return mx::astype(cfg_v, xt.dtype());
}

mx::array step_euler_x0(const mx::array& xt, const mx::array& x0,
                        const mx::array& t_curr, const mx::array& t_next) {
    mx::array r = mx::astype(as_b11(t_next / t_curr), xt.dtype());
    const mx::array one = mx::array(1.0f, xt.dtype());
    return r * xt + (one - r) * x0;
}

}  // namespace stream
}  // namespace sa3
