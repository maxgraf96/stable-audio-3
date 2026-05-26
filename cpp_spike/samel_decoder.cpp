#include "samel_decoder.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace samel {

mx::array SAMELDecoder::operator()(
    const mx::array& latents,
    bool full_attention) const
{
    const auto& sh = latents.shape();
    const int B = sh[0], T_lat = sh[2];

    // 1. Softnorm bottleneck decode (scalar multiply only — no scaling/bias on decoder)
    mx::array x = latents * running_std;

    // 2. project_in: (B, LATENT_DIM, T_lat) → (B, T_lat, LATENT_DIM) → (B, T_lat, DIM)
    x = mx::matmul(mx::transpose(x, {0, 2, 1}),
                   mx::transpose(project_in_w)) + project_in_b;

    // 3. Build 17-position groups: [latent, new_token x16]
    // x_e shape: (B, T_lat, 1, DIM); nt shape: (1, 1, 1, DIM) broadcast to (B, T_lat, 16, DIM)
    mx::array x_e = mx::expand_dims(x, 2);
    mx::array nt  = mx::broadcast_to(
        mx::expand_dims(new_tokens, 0),
        {B, T_lat, SIN_PER_POS, DIM});
    x = mx::concatenate({x_e, nt}, 2);                          // (B, T_lat, 17, DIM)
    x = mx::reshape(x, {B, T_lat * SUB_CHUNK_SIZE, DIM});

    // 4. 12 transformer blocks (decoder uses sin gate on blocks 5..11)
    std::optional<mx::array> mask;
    if (!full_attention) {
        mask = mx::astype(swa_mask(), x.dtype());
    }
    for (const auto& blk : blocks) {
        x = blk(x, mask, full_attention);
    }

    // 5. Drop original latent slot at index 0 of each 17-group, keep last 16
    x = mx::reshape(x, {B, T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::slice(x,
                  /*start=*/mx::Shape{0, 0, 1, 0},
                  /*stop=*/mx::Shape{B, T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::reshape(x, {B, T_lat * SIN_PER_POS, DIM});

    // 6. mapping (Linear after Conv1d reshape): (B, T_lat*16, DIM) → (B, T_lat*16, OUT_CHANNELS),
    //    then transpose to channels-first. Same op order as Python so the matmul
    //    accumulation is bit-exact. The trailing transpose is a non-contiguous
    //    VIEW — callers reading data<float>() must materialize first
    //    (samel::to_row_contiguous helper, or any elementwise op).
    x = mx::matmul(x, mx::transpose(mapping_w)) + mapping_b;
    return mx::transpose(x, {0, 2, 1});
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

SAMELDecoder load_samel_decoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    // mapping.weight is stored as Conv1d (OUT_CHANNELS, DIM, 1) — reshape to (OUT_CHANNELS, DIM).
    mx::array mapping_w = mx::astype(at(tensors, "mapping.weight"), dtype);
    if (mapping_w.ndim() == 3) {
        const auto& s = mapping_w.shape();
        if (s[2] != 1) {
            throw std::runtime_error("expected mapping.weight last dim == 1, got " +
                                     std::to_string(s[2]));
        }
        mapping_w = mx::reshape(mapping_w, {s[0], s[1]});
    }

    SAMELDecoder d{
        /*running_std=*/  mx::astype(at(tensors, "running_std"),       dtype),
        /*project_in_w=*/ mx::astype(at(tensors, "project_in.weight"), dtype),
        /*project_in_b=*/ mx::astype(at(tensors, "project_in.bias"),   dtype),
        /*new_tokens=*/   mx::astype(at(tensors, "new_tokens"),        dtype),
        /*blocks=*/       {},
        /*mapping_w=*/    mapping_w,
        /*mapping_b=*/    mx::astype(at(tensors, "mapping.bias"),      dtype),
    };
    d.blocks.reserve(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        const bool use_sin = (i >= SIN_START_BLOCK);
        d.blocks.push_back(load_transformer_block(tensors, p, dtype, use_sin));
    }
    return d;
}

// ── decode_chunked ───────────────────────────────────────────────────
// Slice helper for the last axis (channel-first time index).
static mx::array slice_last(const mx::array& a, int start, int stop) {
    const auto& shape = a.shape();
    const int ndim = static_cast<int>(shape.size());
    mx::Shape s_start(ndim, 0);
    mx::Shape s_stop(shape.begin(), shape.end());
    s_start[ndim - 1] = start;
    s_stop[ndim - 1]  = stop;
    return mx::slice(a, s_start, s_stop);
}

mx::array decode_chunked(
    const SAMELDecoder& model,
    const mx::array& latents,
    int chunk_size,
    int overlap)
{
    const auto& sh = latents.shape();
    const int T = sh[2];
    const int kernel = chunk_size + 2 * overlap;
    if (T <= kernel) {
        return model(latents);
    }

    std::vector<mx::array> pieces;

    // 1) First decode covers output positions [0, chunk+overlap)
    mx::array first_out = model(slice_last(latents, 0, kernel));
    const int valid_first = chunk_size + overlap;
    pieces.push_back(slice_last(first_out, 0, valid_first * STRIDE));
    int i = valid_first;

    // 2) Interior: stride by chunk_size with bilateral overlap context
    while (i + chunk_size + overlap <= T) {
        mx::array out = model(slice_last(latents, i - overlap, i + chunk_size + overlap));
        pieces.push_back(slice_last(out, overlap * STRIDE, (overlap + chunk_size) * STRIDE));
        i += chunk_size;
    }

    // 3) Last decode covers remaining (T - i) output positions
    const int remaining = T - i;
    if (remaining > 0) {
        mx::array last_out = model(slice_last(latents, T - kernel, T));
        const int total_out = last_out.shape().back();
        pieces.push_back(slice_last(last_out, total_out - remaining * STRIDE, total_out));
    }

    return mx::concatenate(pieces, -1);
}

}  // namespace samel
}  // namespace sa3
