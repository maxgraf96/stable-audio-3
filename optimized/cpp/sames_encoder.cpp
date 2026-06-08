#include "sames_encoder.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace sames {

mx::array SAMESEncoder::operator()(const mx::array& audio_patches) const {
    const auto& sh = audio_patches.shape();
    const int B     = sh[0];
    const int C     = sh[1];
    const int T_aud = sh[2];
    if (C != IN_CHANNELS) {
        throw std::runtime_error(
            "SAME-S encoder expected " + std::to_string(IN_CHANNELS) +
            " channels, got " + std::to_string(C));
    }
    if (T_aud % CHUNK_SIZE_LAT != 0) {
        throw std::runtime_error(
            "SAME-S encoder requires audio length multiple of " +
            std::to_string(CHUNK_SIZE_LAT) + " (got " + std::to_string(T_aud) + ")");
    }

    const int T_lat = T_aud / STRIDE;

    // 1. mapping (Linear, 512 → 768): (B, IN, T_aud) → (B, T_aud, IN) → (B, T_aud, DIM)
    mx::array x = mx::matmul(mx::transpose(audio_patches, {0, 2, 1}),
                             mx::transpose(mapping_w)) + mapping_b;

    // 2. Group every STRIDE positions; append one new_token at the END of each group.
    x = mx::reshape(x, {B * T_lat, STRIDE, DIM});
    mx::array nt = mx::broadcast_to(new_tokens, {B * T_lat, 1, DIM});
    x = mx::concatenate({x, nt}, 1);                          // (B*T_lat, 17, DIM)
    x = mx::reshape(x, {B, T_lat * SUB_CHUNK_SIZE, DIM});     // (B, internal_T, DIM)

    const int internal_T = T_lat * SUB_CHUNK_SIZE;

    // 3. First half (blocks 0..2): chunk to (B*nc1, 34, DIM).
    const int nc1 = internal_T / EFFECTIVE_CHUNK;
    x = mx::reshape(x, {B * nc1, EFFECTIVE_CHUNK, DIM});
    for (int i = 0; i < NUM_BLOCKS / 2; ++i) {
        x = blocks[i](x);
    }
    x = mx::reshape(x, {B, internal_T, DIM});

    // 4. Second half (blocks 3..5): pad with first/last SHIFT internal tokens, re-chunk.
    mx::array left  = mx::slice(x,
                                mx::Shape{0, 0, 0},
                                mx::Shape{B, SHIFT, DIM});
    mx::array right = mx::slice(x,
                                mx::Shape{0, internal_T - SHIFT, 0},
                                mx::Shape{B, internal_T, DIM});
    x = mx::concatenate({left, x, right}, 1);                 // (B, internal_T + 2*SHIFT, DIM)
    const int nc2 = (internal_T + EFFECTIVE_CHUNK) / EFFECTIVE_CHUNK;
    x = mx::reshape(x, {B * nc2, EFFECTIVE_CHUNK, DIM});
    for (int i = NUM_BLOCKS / 2; i < NUM_BLOCKS; ++i) {
        x = blocks[i](x);
    }
    x = mx::reshape(x, {B, internal_T + EFFECTIVE_CHUNK, DIM});
    // Drop the SHIFT padding on both sides — back to (B, internal_T, DIM).
    x = mx::slice(x,
                  mx::Shape{0, SHIFT, 0},
                  mx::Shape{B, SHIFT + internal_T, DIM});

    // 5. Take the LAST position of every 17-sub-chunk (the new_token's output is the latent).
    x = mx::reshape(x, {B * T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::slice(x,
                  mx::Shape{0, SUB_CHUNK_SIZE - 1, 0},
                  mx::Shape{B * T_lat, SUB_CHUNK_SIZE, DIM});  // (B*T_lat, 1, DIM)
    x = mx::reshape(x, {B, T_lat, DIM});

    // 6. project_out (Linear, 768 → 256), then channels-first.
    x = mx::matmul(x, mx::transpose(project_out_w)) + project_out_b;  // (B, T_lat, LATENT_DIM)
    x = mx::transpose(x, {0, 2, 1});                                  // (B, LATENT_DIM, T_lat)

    // 7. Softnorm bottleneck: x = x * scaling_factor + bias; then / running_std.
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

SAMESEncoder load_sames_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    // mapping may be stored either as Linear (DIM, IN_CHANNELS) or as a Conv1d
    // k=1 (DIM, 1, IN_CHANNELS / DIM, IN_CHANNELS, 1). Normalise to Linear.
    mx::array mapping_w = mx::astype(at(tensors, "mapping.weight"), dtype);
    if (mapping_w.ndim() == 3) {
        const auto& s = mapping_w.shape();
        // Either (DIM, 1, IN) [MLX] or (DIM, IN, 1) [PyTorch]; both flatten to (DIM, IN).
        mapping_w = mx::reshape(mapping_w, {s[0], s[1] * s[2]});
    }

    SAMESEncoder e{
        /*mapping_w=*/      mapping_w,
        /*mapping_b=*/      mx::astype(at(tensors, "mapping.bias"),       dtype),
        /*new_tokens=*/     mx::astype(at(tensors, "new_tokens"),         dtype),
        /*blocks=*/         {},
        /*project_out_w=*/  mx::astype(at(tensors, "project_out.weight"), dtype),
        /*project_out_b=*/  mx::astype(at(tensors, "project_out.bias"),   dtype),
        /*scaling_factor=*/ mx::astype(at(tensors, "scaling_factor"),     dtype),
        /*bias=*/           mx::astype(at(tensors, "bias"),               dtype),
        /*running_std=*/    mx::astype(at(tensors, "running_std"),        dtype),
    };
    e.blocks.reserve(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        e.blocks.push_back(
            load_transformer_block(tensors, "blocks." + std::to_string(i) + ".", dtype));
    }
    return e;
}

}  // namespace sames
}  // namespace sa3
