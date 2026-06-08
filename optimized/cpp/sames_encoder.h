// SAME-S encoder (~440 MB fp32 / 220 MB fp16). Inverse of SAME-S decoder.
//
// Input:  [B, 512, T_aud]   audio patches (channels-first, T_aud % 32 == 0)
// Output: [B, 256, T_lat]   latents in softnorm space (T_lat = T_aud / 16)
//
// Ported from optimized/mlx/models/defs/same_s_encoder.py.
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
    mx::array mapping_w;       // (DIM, IN_CHANNELS) — Linear (Conv1d k=1 flattened)
    mx::array mapping_b;       // (DIM,)
    mx::array new_tokens;      // (1, 1, DIM)
    std::vector<TransformerBlock> blocks;  // NUM_BLOCKS = 6
    mx::array project_out_w;   // (LATENT_DIM, DIM)
    mx::array project_out_b;   // (LATENT_DIM,)
    mx::array scaling_factor;  // (1, LATENT_DIM, 1)
    mx::array bias;            // (1, LATENT_DIM, 1)
    mx::array running_std;     // (1,)

    mx::array operator()(const mx::array& audio_patches) const;
};

SAMESEncoder load_sames_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float32);

}  // namespace sames
}  // namespace sa3
