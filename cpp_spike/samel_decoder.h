// SAME-L decoder (1.6 GB fp32 / 850 MB fp16). 426M params.
//
// Input:  [B, 256, T_lat]      latents in softnorm space
// Output: [B, 512, T_lat*16]   audio patches (post patched-pretransform)
//
// Ported from optimized/mlx/models/defs/same_l_decoder.py.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "samel_common.h"

namespace sa3 {
namespace samel {

struct SAMELDecoder {
    mx::array running_std;     // (1,)
    mx::array project_in_w;    // (DIM, LATENT_DIM)
    mx::array project_in_b;    // (DIM,)
    mx::array new_tokens;      // (1, 1, DIM)
    std::vector<TransformerBlock> blocks;  // NUM_BLOCKS = 12
                                           // blocks[i].ff.use_sin = (i >= SIN_START_BLOCK)
    mx::array mapping_w;       // (OUT_CHANNELS, DIM)  reshaped from (OUT_CHANNELS, DIM, 1)
    mx::array mapping_b;       // (OUT_CHANNELS,)

    mx::array operator()(
        const mx::array& latents,
        bool full_attention = false) const;
};

// Load decoder from the {tensors, metadata} map returned by load_safetensors.
// Reshapes mapping.weight from (OUT_CHANNELS, DIM, 1) to (OUT_CHANNELS, DIM)
// — stored as Conv1d in the source npz, used as Linear here.
SAMELDecoder load_samel_decoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float32);

// Uniform-kernel chunked decode. Every model call sees kernel = chunk + 2*overlap
// REAL latents — never zero-padded. Falls back to single-shot if T <= kernel.
// Empirical receptive field is ~8 latents per side; overlap=8 → ≥80 dB, overlap=12
// → bit-exact vs un-chunked at natural sizes.
mx::array decode_chunked(
    const SAMELDecoder& model,
    const mx::array& latents,
    int chunk_size,
    int overlap);

}  // namespace samel
}  // namespace sa3
