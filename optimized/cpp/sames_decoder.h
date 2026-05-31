// SAME-S decoder (~50M params). Mirror of same_s_decoder.py.
//
// Input:  [B, 256, T_lat]      latents in softnorm space (T_lat even)
// Output: [B, 512, T_lat*16]   audio patches (post patched-pretransform)
//
// Differs from SAME-L decoder: chunk-midpoint-shift sequence pass (not a single
// all-blocks pass) and a real Conv1d(k=3, pad=1) output mapping (not a Linear).
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
    mx::array mapping_w;       // (OUT_CHANNELS, 3, DIM) — Conv1d k=3, MLX layout
    mx::array mapping_b;       // (OUT_CHANNELS,)

    mx::array operator()(const mx::array& latents) const;
};

SAMESDecoder load_sames_decoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float32);

// Uniform-kernel chunked decode (SAME-S). kernel = chunk + 2*overlap must be
// EVEN (SAME-S internal T_lat*17 must align to 34). Falls back to single-shot
// if T <= kernel (requires even T then).
mx::array decode_chunked(
    const SAMESDecoder& model,
    const mx::array& latents,
    int chunk_size,
    int overlap);

}  // namespace sames
}  // namespace sa3
