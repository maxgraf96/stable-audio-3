// Sanity check for sa3::sames — load encoder + decoder safetensors, run a
// forward pass with deterministic inputs, dump shape + stats. Used to
// verify loader keys, shapes, and chunked-attention math against Python.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include "samel_common.h"   // to_row_contiguous helper
#include "sames_encoder.h"
#include "sames_decoder.h"

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
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << " <same_s_encoder_f32.safetensors>"
                  << " <same_s_decoder_f32.safetensors>\n";
        return 2;
    }

    auto load = [](const char* path) {
        auto loaded = mx::load_safetensors(path);
        return std::move(loaded.first);
    };
    auto enc_w = load(argv[1]);
    auto dec_w = load(argv[2]);

    auto encoder = sa3::sames::load_sames_encoder(enc_w, mx::float32);
    auto decoder = sa3::sames::load_sames_decoder(dec_w, mx::float32);

    // Deterministic inputs.
    // T_aud must be a multiple of 32 → 32 is the smallest valid choice
    // (gives T_lat = 2, which is the minimum even latent length).
    constexpr int B = 1;
    constexpr int T_aud = 32;

    auto make = [&](const std::vector<int>& shape) {
        int total = 1;
        for (int d : shape) total *= d;
        mx::array a = mx::arange(0, total, 1, mx::float32) / static_cast<float>(total);
        mx::Shape sh(shape.begin(), shape.end());
        return mx::reshape(a, sh);
    };
    mx::array audio = make({B, sa3::sames::IN_CHANNELS, T_aud});

    mx::array latents = encoder(audio);
    mx::eval(latents);
    dump("SAMES_enc", latents);

    mx::array recon = decoder(latents);
    mx::eval(recon);
    dump("SAMES_dec", recon);
    return 0;
}
