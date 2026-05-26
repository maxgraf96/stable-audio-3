// Run the C++ SAME-L decoder against real weights on two paths:
//   1. Single-shot decode of a small input (T_lat <= kernel)
//   2. Chunked decode of a longer input (T_lat > kernel) — exercises
//      decode_chunked's slice/concat orchestration
// Both must be bit-exact against verify_samel_decoder.py.

#include <iomanip>
#include <iostream>
#include <string>

#include <mlx/mlx.h>
#include "samel_common.h"
#include "samel_decoder.h"

namespace mx = mlx::core;

static void dump(const std::string& label, const mx::array& a, int n = 16) {
    mx::array a_fp32 = mx::astype(a, mx::float32);

    // Stats must be computed on the original-layout array (potentially a
    // non-contiguous view), not on the materialized copy — Python's verify
    // does the same, and the summation order is bit-sensitive.
    mx::array mean   = mx::mean(a_fp32);
    mx::array std_   = mx::sqrt(mx::mean((a_fp32 - mean) * (a_fp32 - mean)));
    mx::array absmax = mx::max(mx::abs(a_fp32));
    mx::eval(mean, std_, absmax);

    // For per-value printing, materialize into row-contig so `.data<float>()`
    // reads in logical row-major order (see note in samel_common.h).
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
                  << " <path/to/same_l_decoder_f32.safetensors>\n";
        return 2;
    }

    auto loaded = mx::load_safetensors(argv[1]);
    auto& tensors = loaded.first;
    auto dec = sa3::samel::load_samel_decoder(tensors, mx::float32);

    // ── Test 1: single-shot decode ──
    // Input (1, 256, 16) — small enough that decode_chunked falls through to single-shot.
    {
        const int B = 1, C = 256, T_lat = 16;
        const int total = B * C * T_lat;
        mx::array x = mx::arange(0, total, 1, mx::float32) / static_cast<float>(total);
        x = mx::reshape(x, {B, C, T_lat});
        mx::array y = dec(x);
        mx::eval(y);
        dump("T1 single", y);
    }

    // ── Test 2: chunked decode ──
    // Input (1, 256, 64), chunk_size=8 + overlap=8 → kernel=24, T=64 > 24,
    // so the loop runs (1 first + interior + last).
    {
        const int B = 1, C = 256, T_lat = 64;
        const int total = B * C * T_lat;
        // Different seed-like offset so the chunked test isn't a prefix of T1.
        mx::array x = mx::arange(0, total, 1, mx::float32) / static_cast<float>(total);
        x = mx::reshape(x, {B, C, T_lat});
        mx::array y = sa3::samel::decode_chunked(dec, x, /*chunk_size=*/8, /*overlap=*/8);
        mx::eval(y);
        dump("T2 chunked", y);
    }

    return 0;
}
