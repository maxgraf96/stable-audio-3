// SAME-S encoder (audio → latents). Mirror of same_s_encoder.py.
//
// Input:  [B, 512, T_lat*16]  audio patches (T_lat even; T_aud % 32 == 0)
// Output: [B, 256, T_lat]     latents
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "sames_common.h"

namespace sa3 {
namespace sames {

struct SAMESEncoder {
    mx::array mapping_w;       // (DIM, IN_CHANNELS)
    mx::array mapping_b;       // (DIM,)
    mx::array new_tokens;      // (1, 1, DIM)
    std::vector<TransformerBlock> blocks;  // NUM_BLOCKS = 6
    mx::array project_out_w;   // (LATENT_DIM, DIM)
    mx::array project_out_b;   // (LATENT_DIM,)
    mx::array scaling_factor;  // (1,)
    mx::array bias;            // (LATENT_DIM,)
    mx::array running_std;     // (1,)

    mx::array operator()(const mx::array& patches) const;
};

SAMESEncoder load_sames_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float32);

}  // namespace sames
}  // namespace sa3
