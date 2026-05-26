#include "sa3_pipeline.h"

#include <cmath>
#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {

// ──────────────────────────────────────────────────────────────────
// expo_fourier_features
// ──────────────────────────────────────────────────────────────────

mx::array expo_fourier_features(
    const mx::array& t_in, int dim, float min_freq, float max_freq)
{
    mx::array t = mx::reshape(mx::astype(t_in, mx::float32), {-1, 1});
    int half = dim / 2;
    float denom = static_cast<float>(std::max(half - 1, 1));
    mx::array ramp = mx::arange(0, half, 1, mx::float32) / denom;

    // Python uses fp64 math.log and only casts to fp32 at the mlx scalar-op
    // boundary. Match exactly: compute log + subtraction in fp64, cast once.
    double log_max_d = std::log(static_cast<double>(max_freq));
    double log_min_d = std::log(static_cast<double>(min_freq));
    float scale = static_cast<float>(log_max_d - log_min_d);
    float bias  = static_cast<float>(log_min_d);
    mx::array freqs = mx::exp(ramp * scale + bias);

    // Python chains `t * freqs * 2 * math.pi` left-to-right. Don't pre-compute
    // 2*π — at fp32 the multiplication order can shift the cos/sin arg by a
    // ulp, which is catastrophic for high-freq features (args ~15000 rad).
    mx::array args = t * freqs * 2.0f * static_cast<float>(M_PI);
    return mx::concatenate({mx::cos(args), mx::sin(args)}, -1);
}

// ──────────────────────────────────────────────────────────────────
// SecondsTotalEmbedder
// ──────────────────────────────────────────────────────────────────

mx::array SecondsTotalEmbedder::operator()(float seconds) const {
    return (*this)(std::vector<float>{seconds});
}

mx::array SecondsTotalEmbedder::operator()(const std::vector<float>& seconds) const {
    mx::array s(seconds.begin(), mx::Shape{static_cast<int>(seconds.size())}, mx::float32);
    s = mx::clip(s, mx::array(min_val), mx::array(max_val));
    mx::array norm = (s - min_val) / (max_val - min_val);
    mx::array ff = expo_fourier_features(norm, fourier_dim);
    mx::array out = mx::matmul(ff, mx::transpose(W)) + b;          // (B, 768)
    return mx::expand_dims(out, 1);                                 // (B, 1, 768)
}

// ──────────────────────────────────────────────────────────────────
// apply_prompt_padding
// ──────────────────────────────────────────────────────────────────

mx::array apply_prompt_padding(
    const mx::array& embeds, const mx::array& mask, const mx::array& padding_embedding)
{
    mx::array m = mx::expand_dims(mx::astype(mask, embeds.dtype()), -1);   // (B, S, 1)
    mx::array pe = mx::reshape(mx::astype(padding_embedding, embeds.dtype()),
                               {1, 1, -1});                                  // (1, 1, D)
    return embeds * m + pe * (1.0f - m);
}

// ──────────────────────────────────────────────────────────────────
// logsnr_shift
// ──────────────────────────────────────────────────────────────────

mx::array logsnr_shift(const mx::array& t, float anchor_logsnr, float logsnr_end) {
    mx::array t32 = mx::astype(t, mx::float32);
    mx::array logsnr = logsnr_end - t32 * (logsnr_end - anchor_logsnr);
    mx::array t_out = mx::sigmoid(-logsnr);
    t_out = mx::where(t32 <= 0.0f, mx::zeros_like(t_out), t_out);
    t_out = mx::where(t32 >= 1.0f, mx::ones_like(t_out), t_out);
    return t_out;
}

// ──────────────────────────────────────────────────────────────────
// build_pingpong_schedule
// ──────────────────────────────────────────────────────────────────

mx::array build_pingpong_schedule(int steps, float sigma_max, bool use_logsnr_shift_) {
    mx::array t = mx::linspace(sigma_max, 0.0f, steps + 1, mx::float32);
    if (use_logsnr_shift_) {
        t = logsnr_shift(t);
        // Re-anchor start to sigma_max: t = concat([sigma_max], t[1:])
        mx::array head = mx::array({sigma_max}, mx::float32);
        mx::array tail = mx::slice(t, mx::Shape{1}, mx::Shape{steps + 1});
        t = mx::concatenate({head, tail}, 0);
    }
    return t;
}

// ──────────────────────────────────────────────────────────────────
// sample_flow_pingpong
// ──────────────────────────────────────────────────────────────────

mx::array sample_flow_pingpong(
    const ModelFn& model_fn,
    mx::array x,
    const mx::array& sigmas,
    uint64_t seed,
    std::optional<std::pair<mx::array, mx::array>> paste_back,
    const StepCallback& on_step)
{
    // Pre-extract sigma values to host once (avoid in-loop eval/item calls).
    mx::eval(sigmas);
    const float* sigma_data = sigmas.data<float>();
    int num_steps = sigmas.shape()[0] - 1;

    mx::array key = mx::random::key(seed);

    for (int i = 0; i < num_steps; ++i) {
        float t_curr_f = sigma_data[i];
        float t_next_f = sigma_data[i + 1];

        // t_tensor: Python writes `t_curr * mx.ones(..., x.dtype)` without an
        // astype — t_curr (fp32) * ones (fp16) promotes to fp32. Mirror by
        // passing the raw float (C++ `float * fp16_array` also promotes to fp32).
        mx::array t_tensor = t_curr_f * mx::ones({x.shape()[0]}, x.dtype());
        mx::array v = model_fn(x, t_tensor);

        // denoised: Python uses `t_curr.astype(x.dtype) * v` — t_curr ROUNDED
        // to x.dtype before multiplying. C++ `float * fp32_v` skips that
        // fp16 rounding step. At fp16 this compounds across steps; at fp32
        // it's a no-op.
        mx::array denoised = x - mx::array(t_curr_f, x.dtype()) * v;

        if (i < num_steps - 1 && t_next_f > 0.0f) {
            auto split = mx::random::split(key);
            key = split.first;
            mx::array noise = mx::random::normal(x.shape(), x.dtype(), split.second);
            // Python: `(1.0 - t_next).astype(x.dtype) * denoised + t_next.astype(x.dtype) * noise`
            // — both factors rounded to x.dtype before the multiply.
            mx::array one_minus_t = mx::array(1.0f - t_next_f, x.dtype());
            mx::array t_next_a    = mx::array(t_next_f,         x.dtype());
            x = one_minus_t * denoised + t_next_a * noise;
        } else {
            x = denoised;
        }
        mx::eval(x);
        if (on_step) on_step(i + 1, num_steps);
    }

    if (paste_back.has_value()) {
        const auto& [init_latents, mask] = *paste_back;
        mx::array m = mx::astype(mask, x.dtype());
        x = mx::astype(init_latents, x.dtype()) * m + x * (1.0f - m);
        mx::eval(x);
    }
    return x;
}

// ──────────────────────────────────────────────────────────────────
// patched_decode
// ──────────────────────────────────────────────────────────────────

mx::array patched_decode(const mx::array& patches, int patch_size, int channels) {
    const auto& s = patches.shape();
    if (s.size() != 3) {
        throw std::runtime_error("patched_decode: expected 3D input");
    }
    int B = s[0], CH = s[1], L = s[2];
    if (CH != channels * patch_size) {
        throw std::runtime_error("patched_decode: CH mismatch");
    }
    mx::array x = mx::reshape(patches, {B, channels, patch_size, L});  // b c h l
    x = mx::transpose(x, {0, 1, 3, 2});                                  // b c l h
    return mx::reshape(x, {B, channels, L * patch_size});                // b c (l*h)
}

// ──────────────────────────────────────────────────────────────────
// load_conditioner
// ──────────────────────────────────────────────────────────────────

LoadedConditioner load_conditioner(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix)
{
    auto get = [&](const std::string& suffix) -> mx::array {
        const std::string key = prefix + suffix;
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            throw std::runtime_error("load_conditioner: missing key '" + key + "'");
        }
        return mx::astype(it->second, mx::float32);
    };

    LoadedConditioner out{
        /*padding_embedding=*/get("padding_embedding"),
        /*seconds_embedder=*/SecondsTotalEmbedder{
            /*W=*/get("seconds_total_weight"),
            /*b=*/get("seconds_total_bias"),
            /*min_val=*/0.0f,
            /*max_val=*/384.0f,
            /*fourier_dim=*/256,
        },
    };
    return out;
}

}  // namespace sa3
