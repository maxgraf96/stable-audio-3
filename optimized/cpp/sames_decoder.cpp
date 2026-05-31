// SAME-S decoder implementation.
#include "sames_decoder.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace sames {

// ── Decoder forward ──────────────────────────────────────────────────
mx::array SAMESDecoder::operator()(const mx::array& latents) const {
    const int B = latents.shape()[0];
    const int T_lat = latents.shape()[2];

    // Bottleneck softnorm decode (scalar multiply)
    mx::array x = latents * running_std;

    // [B, 256, T_lat] → [B, T_lat, 256] → project to DIM
    x = mx::matmul(mx::transpose(x, {0, 2, 1}), mx::transpose(project_in_w))
        + project_in_b;

    // new_tokens broadcast: latent slot first, then SIN_PER_POS(16) new tokens.
    mx::array x_e = mx::expand_dims(x, 2);                    // [B, T_lat, 1, DIM]
    mx::array nt = mx::broadcast_to(
        mx::reshape(new_tokens, {1, 1, 1, DIM}),
        {B, T_lat, SIN_PER_POS, DIM});
    x = mx::concatenate({x_e, nt}, 2);                       // [B, T_lat, 17, DIM]
    x = mx::reshape(x, {B, T_lat * SUB_CHUNK_SIZE, DIM});

    // Two-half chunk-midpoint-shift over the 6 blocks
    x = chunk_midpoint_shift(blocks, x, B);                  // [B, internal_T, DIM]

    // Drop the latent slot (position 0 of each sub-chunk), keep 16 audio frames
    x = mx::reshape(x, {B * T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::slice(x, {0, 1, 0}, {B * T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::reshape(x, {B, T_lat * SIN_PER_POS, DIM});       // [B, T_lat*16, DIM]

    // mapping: Conv1d(DIM → OUT_CHANNELS, k=3, pad=1) over the time axis.
    // input [B, T, DIM] channels-last, weight [OUT, 3, DIM]
    mx::array out = mx::conv1d(x, mapping_w, /*stride=*/1, /*padding=*/1) + mapping_b;
    return mx::transpose(out, {0, 2, 1});                    // [B, OUT_CHANNELS, T_lat*16]
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

SAMESDecoder load_sames_decoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    auto get = [&](const std::string& k) { return mx::astype(at(tensors, k), dtype); };

    SAMESDecoder dec{
        /*running_std=*/ get("running_std"),
        /*project_in_w=*/ get("project_in.weight"),
        /*project_in_b=*/ get("project_in.bias"),
        /*new_tokens=*/ get("new_tokens"),
        /*blocks=*/ {},
        /*mapping_w=*/ get("mapping.weight"),
        /*mapping_b=*/ get("mapping.bias"),
    };

    // mapping.weight: stored PyTorch Conv1d layout [OUT, DIM, 3] → MLX [OUT, 3, DIM].
    if (dec.mapping_w.ndim() == 3 &&
        dec.mapping_w.shape()[1] == DIM && dec.mapping_w.shape()[2] == 3) {
        dec.mapping_w = mx::transpose(dec.mapping_w, {0, 2, 1});
    }

    dec.blocks.reserve(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        dec.blocks.push_back(
            load_block(tensors, "blocks." + std::to_string(i) + ".", dtype));
    }
    return dec;
}

// ── Uniform-kernel chunked decode ───────────────────────────────────
mx::array decode_chunked(
    const SAMESDecoder& model,
    const mx::array& latents,
    int chunk_size,
    int overlap)
{
    const int T = latents.shape()[2];
    const int kernel = chunk_size + 2 * overlap;
    if (kernel % 2 != 0)
        throw std::runtime_error("SAME-S needs even kernel (chunk + 2*overlap)");
    if (T <= kernel) {
        if (T % 2 != 0)
            throw std::runtime_error("SAME-S un-chunked requires even T");
        return model(latents);
    }

    const int B = latents.shape()[0];
    std::vector<mx::array> pieces;
    mx::array first_out = model(mx::slice(latents, {0, 0, 0}, {B, LATENT_DIM, kernel}));
    int valid_first = chunk_size + overlap;
    pieces.push_back(mx::slice(first_out, {0, 0, 0},
        {B, OUT_CHANNELS, valid_first * STRIDE}));
    int i = valid_first;

    while (i + chunk_size + overlap <= T) {
        mx::array out = model(mx::slice(latents, {0, 0, i - overlap},
            {B, LATENT_DIM, i + chunk_size + overlap}));
        pieces.push_back(mx::slice(out, {0, 0, overlap * STRIDE},
            {B, OUT_CHANNELS, (overlap + chunk_size) * STRIDE}));
        i += chunk_size;
    }

    int remaining = T - i;
    if (remaining > 0) {
        mx::array last_out = model(mx::slice(latents, {0, 0, T - kernel}, {B, LATENT_DIM, T}));
        pieces.push_back(mx::slice(last_out, {0, 0, last_out.shape()[2] - remaining * STRIDE},
            {B, OUT_CHANNELS, last_out.shape()[2]}));
    }
    return mx::concatenate(pieces, -1);
}

}  // namespace sames
}  // namespace sa3
