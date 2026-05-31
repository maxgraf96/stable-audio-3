// SAME-S encoder implementation.
#include "sames_encoder.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace sames {

mx::array SAMESEncoder::operator()(const mx::array& patches) const {
    const int B = patches.shape()[0];
    const int T_aud = patches.shape()[2];
    const int T_lat = T_aud / STRIDE;

    // [B, IN, T_aud] → [B, T_aud, IN] → project to DIM (Linear mapping)
    mx::array x = mx::matmul(mx::transpose(patches, {0, 2, 1}),
                             mx::transpose(mapping_w)) + mapping_b;

    // group every STRIDE audio frames, APPEND new_token at the END of each
    // sub-chunk (encoder convention; the decoder puts the latent slot first).
    x = mx::reshape(x, {B, T_lat, STRIDE, DIM});
    mx::array nt = mx::broadcast_to(
        mx::reshape(new_tokens, {1, 1, 1, DIM}),
        {B, T_lat, 1, DIM});
    x = mx::concatenate({x, nt}, 2);                         // [B, T_lat, 17, DIM]
    x = mx::reshape(x, {B, T_lat * SUB_CHUNK_SIZE, DIM});

    // Two-half chunk-midpoint-shift over the 6 blocks
    x = chunk_midpoint_shift(blocks, x, B);

    // keep only the new_token output = the LAST position of each sub-chunk
    x = mx::reshape(x, {B * T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::slice(x, {0, SUB_CHUNK_SIZE - 1, 0}, {B * T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::reshape(x, {B, T_lat, DIM});

    // project to latent, transpose to channels-first, THEN softnorm bottleneck
    // (order matters: scaling_factor/bias/running_std are (1, LATENT_DIM, 1)).
    mx::array x_out = mx::matmul(x, mx::transpose(project_out_w)) + project_out_b;  // [B, T_lat, 256]
    x_out = mx::transpose(x_out, {0, 2, 1});                 // [B, LATENT_DIM, T_lat]
    x_out = x_out * scaling_factor + bias;
    x_out = x_out / running_std;
    return x_out;
}

// ── Loader ───────────────────────────────────────────────────────────
static const mx::array& at(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& key)
{
    auto it = tensors.find(key);
    if (it == tensors.end())
        throw std::runtime_error("missing tensor '" + key + "'");
    return it->second;
}

SAMESEncoder load_sames_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    auto get = [&](const std::string& k) { return mx::astype(at(tensors, k), dtype); };

    SAMESEncoder enc{
        /*mapping_w=*/ get("mapping.weight"),
        /*mapping_b=*/ get("mapping.bias"),
        /*new_tokens=*/ get("new_tokens"),
        /*blocks=*/ {},
        /*project_out_w=*/ get("project_out.weight"),
        /*project_out_b=*/ get("project_out.bias"),
        /*scaling_factor=*/ get("scaling_factor"),
        /*bias=*/ get("bias"),
        /*running_std=*/ get("running_std"),
    };

    // mapping.weight: kernel-1 conv stored as (DIM, IN, 1) → squeeze to (DIM, IN).
    if (enc.mapping_w.ndim() == 3) {
        enc.mapping_w = mx::reshape(enc.mapping_w, {DIM, IN_CHANNELS});
    }

    enc.blocks.reserve(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        enc.blocks.push_back(
            load_block(tensors, "blocks." + std::to_string(i) + ".", dtype));
    }
    return enc;
}

}  // namespace sames
}  // namespace sa3
