// StreamPipeline — implementation. Port of realtime/engine/stream.py.
#include "stream.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <mlx/mlx.h>

#include "sa3_pipeline.h"   // build_pingpong_schedule

namespace sa3 {
namespace stream {

static const char* CURVE_NAMES[] = {
    "velocity_scale", "sde_denoise_curve", "x0_target_strength"};

StreamPipeline::StreamPipeline(DiTForward dit, int T_lat, int steps, int depth,
                               mx::Dtype dtype)
    : dit_(std::move(dit)), T_(T_lat), steps_(steps), dtype_(dtype),
      depth_(std::max(1, std::min(depth, MAX_DEPTH))) {
    slots_.resize(depth_);   // unique_ptr<Slot> default-null == empty cells
}

int StreamPipeline::num_active() const {
    int n = 0;
    for (const auto& s : slots_) if (s) ++n;
    return n;
}

// ── shared-mutable curves ────────────────────────────────────────────
static bool is_curve_name(const std::string& name) {
    for (const char* n : CURVE_NAMES) if (name == n) return true;
    return false;
}

void StreamPipeline::set_shared_curve(const std::string& name,
                                      const CurveValue& value) {
    if (!is_curve_name(name)) {
        throw std::runtime_error("unknown curve '" + name + "'");
    }
    shared_curves_.insert_or_assign(name, normalize_curve(value, dtype_));
}

void StreamPipeline::clear_shared_curve(const std::string& name) {
    shared_curves_.erase(name);
}

std::optional<mx::array> StreamPipeline::eff_curve(const Slot& s,
                                                   const std::string& name) const {
    auto sh = shared_curves_.find(name);
    if (sh != shared_curves_.end()) return sh->second;
    auto it = s.curves.find(name);
    if (it != s.curves.end()) return it->second;
    return std::nullopt;
}

// ── schedule cache (keyed by denoise == sigma_max, rounded to 6 dp) ──
const mx::array& StreamPipeline::get_schedule(float denoise) {
    float key = std::round(denoise * 1e6f) / 1e6f;
    auto it = sched_cache_.find(key);
    if (it != sched_cache_.end()) return it->second;
    mx::array sched = build_pingpong_schedule(steps_, key, /*use_logsnr_shift=*/true);
    mx::eval(sched);
    return sched_cache_.emplace(key, std::move(sched)).first->second;
}

// ── admission ────────────────────────────────────────────────────────
void StreamPipeline::submit(const SlotRequest& req) {
    if (static_cast<int>(queue_.size()) >= depth_) {
        queue_.erase(queue_.begin());   // drop-oldest backpressure
    }
    queue_.push_back(req);
}

Slot StreamPipeline::init_slot(const SlotRequest& req) {
    mx::array sched = get_schedule(req.denoise);
    mx::array noise = mx::random::normal({1, IO_CHANNELS, T_}, dtype_,
                                         mx::random::key(req.seed));
    mx::array xt = noise;
    if (req.source_latent.has_value()) {
        const float s = req.denoise;   // a2a init mix: latent*(1-s) + noise*s
        xt = mx::astype(*req.source_latent, dtype_) * mx::array(1.0f - s, dtype_)
             + noise * mx::array(s, dtype_);
    }
    mx::eval(xt);

    std::map<std::string, mx::array> curves;
    auto add_curve = [&](const char* n, const std::optional<CurveValue>& v) {
        if (v.has_value()) curves.emplace(n, normalize_curve(*v, dtype_));
    };
    add_curve("velocity_scale", req.velocity_scale);
    add_curve("sde_denoise_curve", req.sde_denoise_curve);
    add_curve("x0_target_strength", req.x0_target_strength);

    std::optional<mx::array> init_noise;
    if (req.has_cfg() && req.rcfg_mode == "self") init_noise = noise;

    return Slot{
        /*request=*/ req,
        /*xt=*/ xt,
        /*schedule=*/ sched,
        /*rng_key=*/ mx::random::key(req.seed + 1),
        /*curves=*/ std::move(curves),
        /*initial_noise=*/ init_noise,
        /*step_idx=*/ 0,
    };
}

// ── the tick ─────────────────────────────────────────────────────────
std::optional<mx::array> StreamPipeline::tick() {
    std::optional<mx::array> finished;
    // 1. harvest one finished slot
    for (auto& s : slots_) {
        if (s && s->finished()) {
            finished = s->xt;
            s.reset();
            break;
        }
    }
    // 2. admit queued requests into empty cells
    for (auto& s : slots_) {
        if (!s && !queue_.empty()) {
            s = std::make_unique<Slot>(init_slot(queue_.front()));
            queue_.erase(queue_.begin());
        }
    }
    // 3. advance all active (not-yet-finished) slots in one batched forward
    std::vector<Slot*> active;
    for (auto& s : slots_) {
        if (s && !s->finished()) active.push_back(s.get());
    }
    if (!active.empty()) advance(active);
    return finished;
}

static mx::array neg_cross_for(const Slot& s) {
    if (s.request.neg_cross_attn.has_value()) return *s.request.neg_cross_attn;
    return mx::zeros_like(s.request.cross_attn);
}

void StreamPipeline::advance(std::vector<Slot*>& slots) {
    const int B = static_cast<int>(slots.size());

    // POSITIVE forward: one batched DiT pass across all active slots.
    std::vector<mx::array> xs, cs, gs, tcs, tns;
    xs.reserve(B); cs.reserve(B); gs.reserve(B); tcs.reserve(B); tns.reserve(B);
    for (Slot* s : slots) {
        xs.push_back(s->xt);
        cs.push_back(s->request.cross_attn);
        gs.push_back(s->request.global_cond);
        tcs.push_back(mx::take(s->schedule, s->step_idx, 0));
        tns.push_back(mx::take(s->schedule, s->step_idx + 1, 0));
    }
    mx::array xb = mx::concatenate(xs, 0);
    mx::array cb = mx::concatenate(cs, 0);
    mx::array gb = mx::concatenate(gs, 0);
    mx::array t_curr = mx::stack(tcs, 0);   // [B] fp32
    mx::array t_next = mx::stack(tns, 0);
    mx::array v = dit_(xb, t_curr, cb, gb);

    // NEGATIVE forward: only slots needing an uncond pass (cfg!=1 & full).
    std::vector<int> neg_idx;
    for (int j = 0; j < B; ++j) {
        if (slots[j]->request.needs_neg_forward()) neg_idx.push_back(j);
    }
    std::optional<mx::array> v_neg;
    std::unordered_map<int, int> neg_pos;   // slot j -> row in v_neg
    if (!neg_idx.empty()) {
        std::vector<mx::array> xn, cn, gn, tn;
        for (int k = 0; k < static_cast<int>(neg_idx.size()); ++k) {
            int j = neg_idx[k];
            xn.push_back(slots[j]->xt);
            cn.push_back(neg_cross_for(*slots[j]));
            gn.push_back(slots[j]->request.global_cond);
            tn.push_back(mx::take(t_curr, j, 0));
            neg_pos[j] = k;
        }
        v_neg = dit_(mx::concatenate(xn, 0), mx::stack(tn, 0),
                     mx::concatenate(cn, 0), mx::concatenate(gn, 0));
    }

    const int half = steps_ / 2;   // x0-target gated to the refinement half

    std::vector<mx::array> to_eval;
    to_eval.reserve(B);
    for (int j = 0; j < B; ++j) {
        Slot* s = slots[j];
        mx::array vj = mx::slice(v, {j, 0, 0}, {j + 1, v.shape()[1], v.shape()[2]});
        mx::array tc = mx::slice(t_curr, {j}, {j + 1});

        // guidance (CFG + APG) before any curve scaling
        if (s->request.has_cfg()) {
            mx::array uncond_v = (s->request.rcfg_mode == "self")
                ? *s->initial_noise
                : mx::slice(*v_neg, {neg_pos[j], 0, 0},
                            {neg_pos[j] + 1, v_neg->shape()[1], v_neg->shape()[2]});
            vj = apg_cfg(s->xt, vj, uncond_v, tc,
                         CurveValue{s->request.cfg}, s->request.apg);
        }
        auto vs = eff_curve(*s, "velocity_scale");
        if (vs.has_value()) vj = vj * (*vs);

        mx::array tn = mx::slice(t_next, {j}, {j + 1});
        auto sdc = eff_curve(*s, "sde_denoise_curve");
        auto alpha = eff_curve(*s, "x0_target_strength");
        const bool use_sde = sdc.has_value() && s->request.source_latent.has_value();
        const bool morph = alpha.has_value() && s->request.x0_target.has_value()
            && s->step_idx >= half;

        if (use_sde || morph) {
            mx::array x0 = x0_from_vel(s->xt, vj, tc);
            if (morph) {
                x0 = blend_x0_target(x0, mx::astype(*s->request.x0_target, dtype_), *alpha);
            }
            if (use_sde) {
                auto kp = mx::random::split(s->rng_key);
                s->rng_key = kp.first;
                mx::array noise = mx::random::normal(s->xt.shape(), dtype_, kp.second);
                s->xt = step_sde_curve(s->xt, x0, tn, *sdc,
                                       mx::astype(*s->request.source_latent, dtype_), noise);
            } else {
                s->xt = step_euler_x0(s->xt, x0, tc, tn);
            }
        } else {
            s->xt = step_euler(s->xt, vj, tc, tn);
        }
        s->step_idx += 1;
        to_eval.push_back(s->xt);
    }
    mx::eval(to_eval);
}

// ── hot depth resize (in-flight slots survive) ──────────────────────
void StreamPipeline::set_depth(int depth) {
    depth = std::max(1, std::min(depth, MAX_DEPTH));
    if (depth == depth_) return;
    if (depth < depth_) {
        slots_.resize(depth);   // shrink: truncate tail
    } else {
        slots_.resize(depth);   // grow: appends null cells
    }
    depth_ = depth;
}

}  // namespace stream
}  // namespace sa3
