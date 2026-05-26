// Exercise every function in sa3_pipeline.h with deterministic inputs and
// print labeled outputs at full fp32 precision. Cross-check vs
// cpp_spike/verify_sa3_pipeline.py — outputs must match bit-exactly.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "sa3_pipeline.h"

namespace mx = mlx::core;

// Print first N values of a 1D-flattened array at fp32 precision.
static void dump(const std::string& label, const mx::array& a, int n = -1) {
    mx::array f = mx::astype(a, mx::float32);
    mx::eval(f);
    std::cout << "[" << label << "] shape=(";
    for (size_t i = 0; i < f.shape().size(); ++i) {
        std::cout << f.shape()[i];
        if (i + 1 < f.shape().size()) std::cout << ", ";
    }
    std::cout << ")\n";
    int total = static_cast<int>(f.size());
    int limit = (n < 0 || n > total) ? total : n;
    const float* d = f.data<float>();
    std::cout.precision(10);
    for (int i = 0; i < limit; ++i) {
        std::cout << "  [" << std::setw(3) << i << "] " << d[i] << "\n";
    }
}

int main() {
    // ── T1: build_pingpong_schedule(steps=8, sigma_max=1.0) ────────────
    {
        mx::array s = sa3::build_pingpong_schedule(8, 1.0f, /*use_logsnr_shift=*/true);
        dump("T1 schedule(8, 1.0)", s);
    }

    // ── T2: build_pingpong_schedule(steps=4, sigma_max=0.52) ───────────
    {
        mx::array s = sa3::build_pingpong_schedule(4, 0.52f, /*use_logsnr_shift=*/true);
        dump("T2 schedule(4, 0.52)", s);
    }

    // ── T3: logsnr_shift over t = arange(0, 11)/10 ────────────────────
    {
        mx::array t = mx::arange(0, 11, 1, mx::float32) / 10.0f;
        mx::array warped = sa3::logsnr_shift(t);
        dump("T3 logsnr_shift", warped);
    }

    // ── T4: expo_fourier_features(arange(0, 4)/4, dim=8) ──────────────
    {
        mx::array t = mx::arange(0, 4, 1, mx::float32) / 4.0f;
        mx::array ff = sa3::expo_fourier_features(t, 8);
        dump("T4 fourier_features", ff);
    }

    // ── T5: apply_prompt_padding — synthetic 1x4x3 embeds ──────────────
    {
        // embeds[0, s, :] = s+1.0   so values are 1,1,1, 2,2,2, 3,3,3, 4,4,4
        mx::array embeds = mx::arange(1, 5, 1, mx::float32);
        embeds = mx::expand_dims(mx::expand_dims(embeds, 0), -1);   // (1, 4, 1)
        embeds = mx::broadcast_to(embeds, {1, 4, 3});                // (1, 4, 3)
        // mask: real, real, pad, pad
        mx::array mask = mx::array({1, 1, 0, 0}, mx::Shape{1, 4}, mx::int32);
        mx::array pad = mx::array({9.0f, 8.0f, 7.0f});
        mx::array y = sa3::apply_prompt_padding(embeds, mask, pad);
        dump("T5 apply_prompt_padding", y);
    }

    // ── T6: patched_decode — synthetic (1, 4, 3) with patch_size=2, channels=2 → (1, 2, 6)
    {
        mx::array p = mx::arange(0, 12, 1, mx::float32);
        p = mx::reshape(p, {1, 4, 3});
        mx::array y = sa3::patched_decode(p, /*patch_size=*/2, /*channels=*/2);
        dump("T6 patched_decode", y);
    }

    // ── T7: sample_flow_pingpong with a constant-velocity model_fn ─────
    // model_fn returns 0.5 * x (independent of t), so each step's denoised
    // value is x - t_curr * 0.5x = x * (1 - 0.5 * t_curr).  Tests sampler
    // mechanics + noise renoising bit-exactly.
    {
        sa3::ModelFn fn = [](const mx::array& x, const mx::array&) {
            return 0.5f * x;
        };
        mx::array sigmas = sa3::build_pingpong_schedule(4, 1.0f, true);
        mx::array x0 = mx::ones({1, 4}, mx::float32);
        mx::array y = sa3::sample_flow_pingpong(fn, x0, sigmas, /*seed=*/42);
        dump("T7 sample_flow_pingpong", y);
    }

    // ── T8: SecondsTotalEmbedder with deterministic synthetic weights ──
    // Use W shaped (768, 256) but fill with tiny deterministic pattern.
    // Build directly via fp32 vector → mx::array(it, shape, dtype).
    {
        const int OUT = 768, IN = 256;
        // Construct test data in fp64 then cast to fp32 — matches Python's
        // `(i % N) * 0.001` which is int * fp64 = fp64, then cast on mx.array
        // construction. Pure fp32 multiplication would give different last-bit
        // rounding (e.g. fp32(2)*fp32(0.001) ≠ fp32(fp64(2)*fp64(0.001))).
        std::vector<float> W_data(OUT * IN);
        for (int i = 0; i < OUT * IN; ++i) {
            W_data[i] = static_cast<float>(static_cast<double>(i % 13) * 0.001);
        }
        std::vector<float> b_data(OUT);
        for (int i = 0; i < OUT; ++i) {
            b_data[i] = static_cast<float>(static_cast<double>(i % 7) * 0.01);
        }
        sa3::SecondsTotalEmbedder emb{
            mx::array(W_data.begin(), mx::Shape{OUT, IN}, mx::float32),
            mx::array(b_data.begin(), mx::Shape{OUT}, mx::float32),
        };
        mx::array out = emb(30.0f);             // (1, 1, 768)
        dump("T8 seconds_embedder(30.0)", out, /*n=*/12);
    }

    return 0;
}
