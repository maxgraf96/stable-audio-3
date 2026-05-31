// Bit-exactness test for the C++ sm-music DiT port.
//
// Loads the sm-music DiT from safetensors, runs a forward pass on deterministic
// inputs, and dumps shape + first 16 values + stats. Compare against
// verify_dit_sm.py (same deterministic inputs) for bit-exactness.
//
// Usage:
//   ./test_dit_sm <dit_sm_safetensors_path> [T_lat]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "dit_sm.h"
#include "samel_common.h"  // to_row_contiguous

namespace mx = mlx::core;

static void dump(const std::string& name, const mx::array& a_in) {
    mx::array a = mx::astype(a_in, mx::float32);
    a = sa3::samel::to_row_contiguous(a);
    mx::eval(a);
    const float* d = a.data<float>();
    size_t n = a.size();
    std::printf("%s shape=[", name.c_str());
    for (size_t i = 0; i < a.ndim(); ++i)
        std::printf("%d%s", a.shape()[i], i + 1 < a.ndim() ? ", " : "");
    std::printf("]\n");
    std::printf("  first16:");
    for (size_t i = 0; i < std::min<size_t>(16, n); ++i)
        std::printf(" %.6f", d[i]);
    std::printf("\n");
    double sum = 0, sq = 0, amax = 0;
    for (size_t i = 0; i < n; ++i) {
        double v = d[i];
        sum += v; sq += v * v; amax = std::max(amax, std::abs(v));
    }
    double mean = sum / n;
    double std = std::sqrt(sq / n - mean * mean);
    std::printf("  mean=%.6f std=%.6f absmax=%.6f n=%zu\n", mean, std, amax, n);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <dit_sm_safetensors> [T_lat]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int T_lat = argc > 2 ? std::stoi(argv[2]) : 16;
    mx::Dtype dt = (argc > 3 && std::string(argv[3]) == "fp16") ? mx::float16 : mx::float32;

    auto loaded = mx::load_safetensors(path);
    auto& tensors = loaded.first;

    auto dit = sa3::dit_sm::load_dit(tensors, T_lat, dt);

    // Deterministic inputs: arange/total, matching verify_dit_sm.py
    auto make = [&](const std::vector<int>& shape) {
        int total = 1; for (int s : shape) total *= s;
        mx::array a = mx::arange(0, total, 1, mx::float32) / static_cast<float>(total);
        mx::Shape sh(shape.begin(), shape.end());
        return mx::reshape(a, sh);
    };

    mx::array x     = mx::astype(make({1, sa3::dit_sm::IO_CHANNELS, T_lat}), dt);
    mx::array t     = mx::astype(mx::array({0.5f}), dt);
    mx::array cross = mx::astype(make({1, 257, sa3::dit_sm::COND_TOKEN_DIM}), dt);
    mx::array gcond = mx::astype(make({1, sa3::dit_sm::GLOBAL_COND_DIM}), dt);

    auto v = dit(x, t, cross, gcond);
    dump("v", v);
    return 0;
}
