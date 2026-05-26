// Run the C++ DiT-medium against real weights with deterministic inputs.
// Compare bit-exactly with verify_dit.py (Python reference).
//
// Test runs at fp32 (matches the Python __main__) and at small T_lat=16 to
// keep the forward fast.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "dit.h"
#include "samel_common.h"   // for to_row_contiguous

namespace mx = mlx::core;

static void dump(const std::string& label, const mx::array& a, int n = 16) {
    mx::array a_fp32 = mx::astype(a, mx::float32);
    mx::array mean   = mx::mean(a_fp32);
    mx::array std_   = mx::sqrt(mx::mean((a_fp32 - mean) * (a_fp32 - mean)));
    mx::array absmax = mx::max(mx::abs(a_fp32));
    mx::eval(mean, std_, absmax);

    mx::array f = sa3::samel::to_row_contiguous(a_fp32);
    mx::eval(f);
    std::cout << "[" << label << "] shape=(";
    for (size_t i = 0; i < f.shape().size(); ++i) {
        std::cout << f.shape()[i];
        if (i + 1 < f.shape().size()) std::cout << ", ";
    }
    std::cout << ")\n";
    const float* d = f.data<float>();
    int limit = std::min<int>(static_cast<int>(f.size()), n);
    std::cout.precision(10);
    for (int i = 0; i < limit; ++i) {
        std::cout << "  " << label << "[" << std::setw(3) << i << "] = " << d[i] << "\n";
    }
    std::cout << "  " << label << " mean    = " << mean.item<float>() << "\n";
    std::cout << "  " << label << " std     = " << std_.item<float>() << "\n";
    std::cout << "  " << label << " abs.max = " << absmax.item<float>() << "\n";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0]
                  << " <path/to/dit_medium_f16.safetensors>\n";
        return 2;
    }

    auto loaded = mx::load_safetensors(argv[1]);
    auto& tensors = loaded.first;

    constexpr int T_lat = 16;
    auto dit_model = sa3::dit::load_dit(tensors, T_lat, mx::float32);

    // Deterministic inputs (all fp32, all derived from arange/total to be
    // reproducible across Python and C++).
    auto make = [&](const std::vector<int>& shape) {
        int total = 1;
        for (int d : shape) total *= d;
        mx::array a = mx::arange(0, total, 1, mx::float32) / static_cast<float>(total);
        mx::Shape sh(shape.begin(), shape.end());
        return mx::reshape(a, sh);
    };
    mx::array x     = make({1, sa3::dit::IO_CHANNELS, T_lat});
    mx::array tval  = mx::array({0.5f});                                  // (1,)
    mx::array cross = make({1, 257, sa3::dit::COND_TOKEN_DIM});
    mx::array gcond = make({1, sa3::dit::GLOBAL_COND_DIM});

    mx::array out = dit_model(x, tval, cross, gcond);
    mx::eval(out);
    dump("DiT", out);
    return 0;
}
