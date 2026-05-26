// Run the C++ SAME-L encoder against real weights and a deterministic input.
// Print the first N values of the output flat for diff against verify_samel_encoder.py.

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mlx/mlx.h>
#include "samel_encoder.h"

namespace mx = mlx::core;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0]
                  << " <path/to/same_l_encoder_f32.safetensors>\n";
        return 2;
    }

    auto loaded = mx::load_safetensors(argv[1]);
    auto& tensors = loaded.first;

    auto enc = sa3::samel::load_samel_encoder(tensors, mx::float32);

    // Deterministic input: (1, 512, 128), values in [0, ~0.262] generated as
    // arange / total_size. T_aud=128, so T_lat = 128/16 = 8.
    const int B = 1, C = 512, T_aud = 128;
    const int total = B * C * T_aud;
    mx::array x = mx::arange(0, total, 1, mx::float32) / static_cast<float>(total);
    x = mx::reshape(x, {B, C, T_aud});

    mx::array y = enc(x);
    mx::eval(y);

    // Print shape + first 16 values
    std::cout << "y shape=(";
    for (size_t i = 0; i < y.shape().size(); ++i) {
        std::cout << y.shape()[i];
        if (i + 1 < y.shape().size()) std::cout << ", ";
    }
    std::cout << ")\n";

    const float* d = y.data<float>();
    std::cout.precision(10);
    int limit = std::min<int>(static_cast<int>(y.size()), 16);
    for (int i = 0; i < limit; ++i) {
        std::cout << "  y[" << std::setw(3) << i << "] = " << d[i] << "\n";
    }

    // Also print summary stats (mean, std, abs.max) for a quick sanity check
    mx::array y_mean = mx::mean(y);
    mx::array y_std  = mx::sqrt(mx::mean((y - y_mean) * (y - y_mean)));
    mx::array y_abs_max = mx::max(mx::abs(y));
    mx::eval(y_mean, y_std, y_abs_max);
    std::cout << "  mean    = " << y_mean.item<float>() << "\n";
    std::cout << "  std     = " << y_std.item<float>() << "\n";
    std::cout << "  abs.max = " << y_abs_max.item<float>() << "\n";
    return 0;
}
