// Bit-exactness test for the C++ SAME-S encoder. Compare vs verify_sames_encoder.py.
//   ./test_sames_encoder <same_s_encoder_safetensors> [T_lat]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "sames_encoder.h"
#include "samel_common.h"  // to_row_contiguous

namespace mx = mlx::core;

static void dump(const std::string& name, const mx::array& a_in) {
    mx::array a = sa3::samel::to_row_contiguous(mx::astype(a_in, mx::float32));
    mx::eval(a);
    const float* d = a.data<float>();
    size_t n = a.size();
    std::printf("%s shape=[", name.c_str());
    for (size_t i = 0; i < a.ndim(); ++i)
        std::printf("%d%s", a.shape()[i], i + 1 < a.ndim() ? ", " : "");
    std::printf("]\n  first16:");
    for (size_t i = 0; i < std::min<size_t>(16, n); ++i) std::printf(" %.6f", d[i]);
    std::printf("\n");
    double sum = 0, sq = 0, amax = 0;
    for (size_t i = 0; i < n; ++i) { double v = d[i]; sum += v; sq += v * v; amax = std::max(amax, std::abs(v)); }
    double mean = sum / n, std = std::sqrt(sq / n - mean * mean);
    std::printf("  mean=%.6f std=%.6f absmax=%.6f n=%zu\n", mean, std, amax, n);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <safetensors> [T_lat]\n", argv[0]); return 1; }
    int T_lat = argc > 2 ? std::stoi(argv[2]) : 16;
    int T_aud = T_lat * 16;
    auto loaded = mx::load_safetensors(argv[1]);
    auto enc = sa3::sames::load_sames_encoder(loaded.first, mx::float32);

    int total = 512 * T_aud;
    mx::array patches = mx::reshape(
        mx::arange(0, total, 1, mx::float32) / static_cast<float>(total),
        mx::Shape{1, 512, T_aud});
    auto out = enc(patches);
    dump("enc", out);
    return 0;
}
