// SAME-S decoder (~440 MB fp32 / 220 MB fp16). Inverse of SAME-S encoder.
//
// Input:  [B, 256, T_lat]      latents in softnorm space (T_lat must be even)
// Output: [B, 512, T_lat*16]   audio patches (channels-first)
//
// Ported from optimized/mlx/models/defs/same_s_decoder.py.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "sames_common.h"

namespace sa3 {
namespace sames {

struct SAMESDecoder {
    mx::array running_std;     // (1,)
    mx::array project_in_w;    // (DIM, LATENT_DIM)
    mx::array project_in_b;    // (DIM,)
    mx::array new_tokens;      // (1, 1, DIM)
    std::vector<TransformerBlock> blocks;  // NUM_BLOCKS = 6
    mx::array mapping_w;       // (OUT_CHANNELS, 3, DIM)  Conv1d k=3 MLX layout
    mx::array mapping_b;       // (OUT_CHANNELS,)

    mx::array operator()(const mx::array& latents) const;
};

SAMESDecoder load_sames_decoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float32);

// Uniform-kernel chunked decode for SAME-S. Mirrors
// same_s_decoder.py:decode_chunked: every model call sees a window of
// `kernel_size = chunk_size + 2*overlap` REAL latents — never zero-padded.
// `kernel_size` must be even (SAME-S internal alignment requires
// internal_T = T*17 divisible by 34, i.e. T even).
mx::array decode_chunked(
    const SAMESDecoder& model,
    const mx::array& latents,
    int chunk_size,
    int overlap);

}  // namespace sames
}  // namespace sa3
