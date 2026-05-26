// Run the C++ T5Gemma encoder against real weights and deterministic token IDs.
// Print first 16 values + summary stats; cross-check against verify_t5gemma.py.
//
// Inputs avoid the SentencePiece tokenizer entirely — we feed token IDs directly
// so the bit-exact comparison is purely an encoder-forward test.

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "samel_common.h"   // to_row_contiguous helper
#include "t5gemma.h"

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
                  << " <path/to/t5gemma_f16.safetensors>\n";
        return 2;
    }

    auto loaded = mx::load_safetensors(argv[1]);
    auto& tensors = loaded.first;
    auto enc = sa3::t5g::load_encoder(tensors, mx::float16);

    // Deterministic input: 30 token IDs starting from 1 (avoid pad=0).
    const int B = 1, S = 30;
    std::vector<int32_t> ids_data(B * S);
    for (int i = 0; i < B * S; ++i) ids_data[i] = i + 1;
    mx::array ids(ids_data.begin(), mx::Shape{B, S}, mx::int32);

    std::vector<int32_t> mask_data(B * S, 1);
    mx::array mask(mask_data.begin(), mx::Shape{B, S}, mx::int32);

    mx::array out = enc(ids, mask);
    mx::eval(out);
    dump("T5G", out);
    return 0;
}
