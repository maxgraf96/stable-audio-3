// SAME-L encoder (1.6 GB fp32 / 850 MB fp16).
//
// Input:  [B, 512, T_audio_patches]   channels-first audio patches
// Output: [B, 256, T_lat]              latents in softnorm space
//
// Ported from optimized/mlx/models/defs/same_l_encoder.py.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "samel_common.h"

namespace sa3 {
namespace samel {

struct SAMELEncoder {
    mx::array mapping_w;       // (DIM, IN_CHANNELS)
    mx::array mapping_b;       // (DIM,)
    mx::array new_tokens;      // (1, 1, DIM)
    std::vector<TransformerBlock> blocks;  // NUM_BLOCKS = 12, all use_sin=false
    mx::array project_out_w;   // (LATENT_DIM, DIM)
    mx::array project_out_b;   // (LATENT_DIM,)
    mx::array scaling_factor;  // (1, LATENT_DIM, 1)
    mx::array bias;            // (1, LATENT_DIM, 1)
    mx::array running_std;     // (1,)

    mx::array operator()(
        const mx::array& audio_patches,
        bool full_attention = false) const;
};

// Load encoder from the {tensors, metadata} map returned by load_safetensors.
SAMELEncoder load_samel_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float32);

}  // namespace samel
}  // namespace sa3
