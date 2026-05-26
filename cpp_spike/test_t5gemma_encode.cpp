// End-to-end T5Gemma test: feed strings, run the SentencePiece tokenizer
// plus the encoder forward pass, print token IDs + embeddings.
// Cross-checked bit-exactly by verify_t5gemma_encode.py.

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "samel_common.h"
#include "t5gemma.h"

namespace mx = mlx::core;

static void print_ids(const std::string& label, const mx::array& ids) {
    mx::array f = sa3::samel::to_row_contiguous(ids);
    mx::eval(f);
    const auto& s = f.shape();
    const int B = s[0], S = s[1];
    const int32_t* d = f.data<int32_t>();
    std::cout << "[" << label << "] shape=(" << B << ", " << S << ")\n";
    for (int i = 0; i < B; ++i) {
        std::cout << "  row " << i << ": [";
        for (int j = 0; j < S; ++j) {
            std::cout << d[i * S + j];
            if (j + 1 < S) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

static void dump_embeds(const std::string& label, const mx::array& a, int n = 16) {
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
    sa3::t5g::T5Gemma t5 = sa3::t5g::load_t5gemma(tensors, mx::float16);

    // Three real prompts + one empty (exercises the empty-row guard).
    std::vector<std::string> prompts = {
        "A beautiful piano arpeggio grows into a grand cinematic climax",
        "lofi house loop",
        "Amen break 174 BPM",
        "",
    };
    const int max_len = 20;

    // First: print the tokenized IDs + mask so the Python ref can cross-check
    // the tokenizer side independently of the encoder.
    auto tk = t5.tokenizer.tokenize(prompts, max_len);
    print_ids("ids",  tk.first);
    print_ids("mask", tk.second);

    // Then: full encode (tokenize + encoder forward with empty-row guard).
    auto enc = t5.encode(prompts, max_len);
    dump_embeds("T5G", enc.first);
    return 0;
}
