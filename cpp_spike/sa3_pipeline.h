// C++ port of optimized/mlx/models/defs/sa3_pipeline.py.
//
// All functions return mlx::array and call only mlx::core primitives — the
// same Metal kernels the Python pipeline uses, so output should be bit-exact
// against the Python reference at fp32 (and within fp16 rounding at fp16).
//
// Tested by test_sa3_pipeline.cpp vs verify_sa3_pipeline.py.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mlx/array.h>

namespace sa3 {

namespace mx = mlx::core;

// ──────────────────────────────────────────────────────────────────
// Conditioner: seconds_total → 768-dim embedding
// ──────────────────────────────────────────────────────────────────

mx::array expo_fourier_features(
    const mx::array& t,
    int dim = 256,
    float min_freq = 0.5f,
    float max_freq = 10000.0f);

struct SecondsTotalEmbedder {
    mx::array W;                  // (768, 256)
    mx::array b;                  // (768,)
    float min_val = 0.0f;
    float max_val = 384.0f;
    int fourier_dim = 256;

    mx::array operator()(float seconds) const;
    mx::array operator()(const std::vector<float>& seconds) const;
};

// ──────────────────────────────────────────────────────────────────
// Prompt padding: replace padded positions with the learned embedding
// ──────────────────────────────────────────────────────────────────

mx::array apply_prompt_padding(
    const mx::array& embeds,        // (B, S, D)
    const mx::array& mask,          // (B, S)  — 1 = real, 0 = pad
    const mx::array& padding_embedding);  // (D,)

// ──────────────────────────────────────────────────────────────────
// Schedule + Pingpong sampler (rectified-flow / rf_denoiser)
// ──────────────────────────────────────────────────────────────────

mx::array logsnr_shift(
    const mx::array& t,
    float anchor_logsnr = -6.2f,
    float logsnr_end = 2.0f);

mx::array build_pingpong_schedule(
    int steps,
    float sigma_max = 1.0f,
    bool use_logsnr_shift = true);

using ModelFn = std::function<mx::array(const mx::array& /*x*/, const mx::array& /*t*/)>;
using StepCallback = std::function<void(int /*step*/, int /*total*/)>;

mx::array sample_flow_pingpong(
    const ModelFn& model_fn,
    mx::array x,
    const mx::array& sigmas,
    uint64_t seed = 0,
    std::optional<std::pair<mx::array, mx::array>> paste_back = std::nullopt,
    const StepCallback& on_step = nullptr);

// ──────────────────────────────────────────────────────────────────
// Patched pretransform decode: [B, 512, T*16] → [B, 2, T*4096]
// ──────────────────────────────────────────────────────────────────

mx::array patched_decode(
    const mx::array& patches,
    int patch_size = 256,
    int channels = 2);

// ──────────────────────────────────────────────────────────────────
// Loader: pull conditioner weights from a pre-loaded tensor map
// ──────────────────────────────────────────────────────────────────

struct LoadedConditioner {
    mx::array padding_embedding;
    SecondsTotalEmbedder seconds_embedder;
};

// Read padding_embedding + seconds_total_{weight,bias} from a safetensors
// tensor map (e.g. the DiT .safetensors with its baked-in conditioner under
// the "cond." prefix). Mirrors load_conditioner_from_npz() in the Python.
LoadedConditioner load_conditioner(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix = "");

}  // namespace sa3
