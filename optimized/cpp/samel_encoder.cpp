#include "samel_encoder.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace samel {

mx::array SAMELEncoder::operator()(
    const mx::array& audio_patches,
    bool full_attention) const
{
    const auto& sh = audio_patches.shape();
    const int B = sh[0], C = sh[1], T_aud = sh[2];
    if (C != IN_CHANNELS) {
        throw std::runtime_error("expected " + std::to_string(IN_CHANNELS) +
                                 " channels, got " + std::to_string(C));
    }
    if (T_aud % STRIDE != 0) {
        throw std::runtime_error("audio_patch length " + std::to_string(T_aud) +
                                 " must be a multiple of " + std::to_string(STRIDE));
    }
    const int T_lat = T_aud / STRIDE;

    // 1. mapping: (B, IN_CHANNELS, T_aud) → (B, T_aud, DIM)
    mx::array x = mx::matmul(mx::transpose(audio_patches, {0, 2, 1}),
                              mx::transpose(mapping_w)) + mapping_b;

    // 2. Group every STRIDE positions, append 1 broadcast new_token at the end
    x = mx::reshape(x, {B * T_lat, STRIDE, DIM});
    mx::array nt = mx::broadcast_to(new_tokens, {B * T_lat, 1, DIM});
    x = mx::concatenate({x, nt}, 1);                                 // (B*T_lat, 17, DIM)
    x = mx::reshape(x, {B, T_lat * SUB_CHUNK_SIZE, DIM});

    // 3. 12 transformer blocks with SWA mask
    std::optional<mx::array> mask;
    if (!full_attention) {
        mask = mx::astype(swa_mask(), x.dtype());
    }
    for (const auto& blk : blocks) {
        x = blk(x, mask, full_attention);
    }

    // 4. Take the LAST position of every 17-sub-chunk (the new_token's output)
    x = mx::reshape(x, {B, T_lat, SUB_CHUNK_SIZE, DIM});
    // slice [:, :, -1, :] → take index SUB_CHUNK_SIZE-1 on axis 2
    x = mx::slice(x,
                  /*start=*/mx::Shape{0, 0, SUB_CHUNK_SIZE - 1, 0},
                  /*stop=*/mx::Shape{B, T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::reshape(x, {B, T_lat, DIM});

    // 5. project_out 1536 → 256, transpose to (B, LATENT_DIM, T_lat)
    x = mx::matmul(x, mx::transpose(project_out_w)) + project_out_b;
    x = mx::transpose(x, {0, 2, 1});

    // 6. Softnorm bottleneck (encoder direction): x * scaling + bias, / running_std
    x = x * scaling_factor + bias;
    x = x / running_std;
    return x;
}

// ── Loader ───────────────────────────────────────────────────────────
static const mx::array& at(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& key)
{
    auto it = tensors.find(key);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor '" + key + "'");
    }
    return it->second;
}

SAMELEncoder load_samel_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    SAMELEncoder e{
        /*mapping_w=*/      mx::astype(at(tensors, "mapping.weight"), dtype),
        /*mapping_b=*/      mx::astype(at(tensors, "mapping.bias"),   dtype),
        /*new_tokens=*/     mx::astype(at(tensors, "new_tokens"),     dtype),
        /*blocks=*/         {},
        /*project_out_w=*/  mx::astype(at(tensors, "project_out.weight"), dtype),
        /*project_out_b=*/  mx::astype(at(tensors, "project_out.bias"),   dtype),
        /*scaling_factor=*/ mx::astype(at(tensors, "scaling_factor"),     dtype),
        /*bias=*/           mx::astype(at(tensors, "bias"),               dtype),
        /*running_std=*/    mx::astype(at(tensors, "running_std"),        dtype),
    };
    e.blocks.reserve(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        e.blocks.push_back(load_transformer_block(tensors, p, dtype, /*use_sin=*/false));
    }
    return e;
}

}  // namespace samel
}  // namespace sa3
