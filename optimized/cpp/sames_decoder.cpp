#include "sames_decoder.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace sames {

mx::array SAMESDecoder::operator()(const mx::array& latents) const {
    const auto& sh = latents.shape();
    const int B     = sh[0];
    const int T_lat = sh[2];

    // 1. Softnorm bottleneck decode (scalar multiply — no scaling/bias on decoder).
    mx::array x = latents * running_std;

    // 2. project_in: (B, LATENT_DIM, T_lat) → (B, T_lat, LATENT_DIM) → (B, T_lat, DIM).
    x = mx::matmul(mx::transpose(x, {0, 2, 1}),
                   mx::transpose(project_in_w)) + project_in_b;

    // 3. Build 17-position sub-chunks: [latent, new_token x 16].
    mx::array x_e = mx::expand_dims(x, 2);                              // (B, T_lat, 1, DIM)
    mx::array nt  = mx::broadcast_to(
        mx::expand_dims(new_tokens, 0),                                 // (1, 1, 1, DIM)
        {B, T_lat, SIN_PER_POS, DIM});
    x = mx::concatenate({x_e, nt}, 2);                                  // (B, T_lat, 17, DIM)
    x = mx::reshape(x, {B, T_lat * SUB_CHUNK_SIZE, DIM});               // (B, internal_T, DIM)

    const int internal_T = T_lat * SUB_CHUNK_SIZE;

    // 4. First half (blocks 0..2): chunks of 34.
    const int nc1 = internal_T / EFFECTIVE_CHUNK;
    x = mx::reshape(x, {B * nc1, EFFECTIVE_CHUNK, DIM});
    for (int i = 0; i < NUM_BLOCKS / 2; ++i) {
        x = blocks[i](x);
    }
    x = mx::reshape(x, {B, internal_T, DIM});

    // 5. Second half (blocks 3..5): pad with first/last 17 internal tokens, re-chunk.
    mx::array left  = mx::slice(x,
                                mx::Shape{0, 0, 0},
                                mx::Shape{B, SHIFT, DIM});
    mx::array right = mx::slice(x,
                                mx::Shape{0, internal_T - SHIFT, 0},
                                mx::Shape{B, internal_T, DIM});
    x = mx::concatenate({left, x, right}, 1);                           // (B, internal_T + 2*SHIFT, DIM)
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

    // 6. Drop the original latent slot at index 0 of each 17-group, keep last 16.
    x = mx::reshape(x, {B, T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::slice(x,
                  mx::Shape{0, 0, 1, 0},
                  mx::Shape{B, T_lat, SUB_CHUNK_SIZE, DIM});
    x = mx::reshape(x, {B, T_lat * SIN_PER_POS, DIM});                  // (B, T_lat*16, DIM)

    // 7. Mapping (Conv1d k=3 pad=1): (B, T_lat*16, DIM) → (B, T_lat*16, OUT_CHANNELS).
    //    mx::conv1d takes [B, T, C_in] input + [C_out, k, C_in] weight when the
    //    canonical MLX nn.Conv1d layout is used.
    mx::array h = mx::conv1d(x, mapping_w, /*stride=*/1, /*padding=*/1) + mapping_b;
    // 8. Channels-first output for the audio side.
    return mx::transpose(h, {0, 2, 1});                                 // (B, OUT_CHANNELS, T_lat*16)
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

SAMESDecoder load_sames_decoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    // mapping.weight may be either MLX layout (OUT, 3, DIM) or PyTorch layout
    // (OUT, DIM, 3); normalise to MLX layout (OUT, 3, DIM).
    mx::array mapping_w = mx::astype(at(tensors, "mapping.weight"), dtype);
    if (mapping_w.ndim() != 3) {
        throw std::runtime_error("mapping.weight expected 3-D Conv1d weight");
    }
    {
        const auto& s = mapping_w.shape();
        if (s[1] == DIM && s[2] == 3) {
            // PyTorch (OUT, IN, K) → MLX (OUT, K, IN)
            mapping_w = mx::transpose(mapping_w, {0, 2, 1});
        } else if (!(s[1] == 3 && s[2] == DIM)) {
            throw std::runtime_error(
                "mapping.weight has unexpected shape; got ("
                + std::to_string(s[0]) + ","
                + std::to_string(s[1]) + ","
                + std::to_string(s[2]) + ")");
        }
    }

    SAMESDecoder d{
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
        d.blocks.push_back(
            load_transformer_block(tensors, "blocks." + std::to_string(i) + ".", dtype));
    }
    return d;
}

// ── decode_chunked ───────────────────────────────────────────────────
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
    const SAMESDecoder& model,
    const mx::array& latents,
    int chunk_size,
    int overlap)
{
    const auto& sh = latents.shape();
    const int T = sh[2];
    const int kernel = chunk_size + 2 * overlap;
    if (kernel % 2 != 0) {
        throw std::runtime_error(
            "SAME-S decode_chunked requires even kernel (chunk + 2*overlap); got "
            + std::to_string(kernel));
    }
    if (T <= kernel) {
        // Un-chunked fallback. SAME-S un-chunked still requires even T so the
        // internal_T = T*17 aligns to 34.
        if (T % 2 != 0) {
            throw std::runtime_error(
                "SAME-S un-chunked decode requires even T (= " + std::to_string(T) +
                "); pick a smaller chunk/overlap so kernel < T");
        }
        return model(latents);
    }

    std::vector<mx::array> pieces;
    pieces.reserve(static_cast<size_t>((T + chunk_size - 1) / chunk_size + 2));

    // 1) First decode covers output positions [0, chunk+overlap).
    mx::array first_out = model(slice_last(latents, 0, kernel));
    const int valid_first = chunk_size + overlap;
    pieces.push_back(slice_last(first_out, 0, valid_first * STRIDE));
    int i = valid_first;

    // 2) Interior: stride by chunk_size with bilateral overlap.
    while (i + chunk_size + overlap <= T) {
        mx::array out = model(slice_last(latents, i - overlap, i + chunk_size + overlap));
        pieces.push_back(slice_last(out, overlap * STRIDE, (overlap + chunk_size) * STRIDE));
        i += chunk_size;
    }

    // 3) Last decode covers the remaining (T - i) output positions.
    const int remaining = T - i;
    if (remaining > 0) {
        mx::array last_out = model(slice_last(latents, T - kernel, T));
        const int total_out = last_out.shape().back();
        pieces.push_back(slice_last(last_out, total_out - remaining * STRIDE, total_out));
    }

    return mx::concatenate(pieces, -1);
}

}  // namespace sames
}  // namespace sa3
