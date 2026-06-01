// StreamPipeline — StreamDiffusion-style ring buffer for Stable Audio 3 (C++).
//
// A fixed-depth ring of in-flight generations, each at a different denoising
// step, advanced together by ONE batched DiT forward per tick(). Direct port of
// realtime/engine/stream.py — bit-exact (fp32) / fp16-rounding (fp16) vs the
// Python engine, and self-consistent (depth-1 stream == standalone euler,
// depth-D stream == B=D batch).
//
// The DiT is injected as a std::function so the pipeline is model-agnostic
// (dit_sm or medium), matching how the orchestrator dispatches DiTForward.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "stream_ode.h"

namespace sa3 {
namespace stream {

namespace mx = mlx::core;

constexpr int IO_CHANNELS = 256;
constexpr int MAX_DEPTH   = 8;

// DiT forward: (x[B,256,T], t[B], cross[B,257,768], global[B,768]) -> v[B,256,T].
// local_add_cond is unused by the streaming engine (no inpaint), so the binding
// can ignore it; kept absent here to keep the signature tight.
using DiTForward = std::function<mx::array(
    const mx::array& x, const mx::array& t,
    const mx::array& cross, const mx::array& global_cond)>;

// Immutable admission ticket: everything needed to start one generation.
struct SlotRequest {
    mx::array cross_attn;            // [1, 257, 768]
    mx::array global_cond;          // [1, 768]
    uint64_t  seed = 0;
    float     denoise = 0.5f;       // sigma_max: init noise level AND schedule start
    std::optional<mx::array> source_latent;   // [1, 256, T]; absent = text2audio

    // Per-frame curves (absent = inactive). std::optional<CurveValue>.
    std::optional<CurveValue> velocity_scale;
    std::optional<CurveValue> sde_denoise_curve;
    std::optional<CurveValue> x0_target_strength;

    // x0-target morph (refinement-half gated). x0_target is [1,256,T].
    std::optional<mx::array> x0_target;

    // CFG. cfg==1 disables guidance. rcfg_mode: "full" (uncond forward using
    // neg_cross_attn) or "self" (init noise as virtual uncond, no extra forward).
    float       cfg = 1.0f;
    float       apg = 1.0f;
    std::string rcfg_mode = "full";
    std::optional<mx::array> neg_cross_attn;   // [1,257,768]; absent+full -> zeros

    bool has_cfg() const { return cfg != 1.0f; }
    bool needs_neg_forward() const { return has_cfg() && rcfg_mode == "full"; }
};

// Mutable runtime cell: the evolving latent + schedule position.
struct Slot {
    SlotRequest request;
    mx::array xt;                   // [1, 256, T]
    mx::array schedule;             // [S+1] fp32 sigmas
    mx::array rng_key;              // per-slot key for SDE re-noise
    std::map<std::string, mx::array> curves;   // pre-normalized per-slot [.,1,T]
    std::optional<mx::array> initial_noise;     // RCFG-self virtual uncond
    int step_idx = 0;

    bool finished() const { return step_idx >= schedule.shape()[0] - 1; }
};

class StreamPipeline {
public:
    StreamPipeline(DiTForward dit, int T_lat, int steps = 8, int depth = 4,
                   mx::Dtype dtype = mx::float16);

    int depth() const { return depth_; }
    int num_active() const;
    int queue_size() const { return static_cast<int>(queue_.size()); }

    // Shared-mutable per-frame curves (1-tick onset on all in-flight slots).
    // Pass the curve; pass clear_shared_curve(name) to revert to per-slot.
    void set_shared_curve(const std::string& name, const CurveValue& value);
    void clear_shared_curve(const std::string& name);

    void submit(const SlotRequest& req);

    // Advance every active slot one step. Returns a finished latent [1,256,T]
    // this tick, or std::nullopt during warmup / when none finish.
    std::optional<mx::array> tick();

    void set_depth(int depth);

private:
    const mx::array& get_schedule(float denoise);
    Slot init_slot(const SlotRequest& req);
    std::optional<mx::array> eff_curve(const Slot& s, const std::string& name) const;
    void advance(std::vector<Slot*>& slots);

    DiTForward dit_;
    int T_;
    int steps_;
    mx::Dtype dtype_;
    int depth_;
    std::vector<std::unique_ptr<Slot>> slots_;   // nullptr == empty cell
    std::vector<SlotRequest> queue_;
    std::map<float, mx::array> sched_cache_;
    std::map<std::string, mx::array> shared_curves_;
};

}  // namespace stream
}  // namespace sa3
