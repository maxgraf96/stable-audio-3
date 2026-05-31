// SAME-S shared building blocks implementation.
#include "sames_common.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace sames {

// ── DyT ──────────────────────────────────────────────────────────────
mx::array DyT::operator()(const mx::array& x) const {
    return gamma * mx::tanh(alpha * x) + beta;
}

// ── DifferentialAttention (full, no SWA) ─────────────────────────────
mx::array DifferentialAttention::operator()(const mx::array& x) const {
    const auto& s = x.shape();
    const int B = s[0], T = s[1];
    constexpr int H = NUM_HEADS, D = HEAD_DIM;

    mx::array qkv = mx::matmul(x, mx::transpose(to_qkv));
    auto parts = mx::split(qkv, 5, -1);
    auto to_heads = [&](const mx::array& t) {
        return mx::transpose(mx::reshape(t, {B, T, H, D}), {0, 2, 1, 3});
    };
    mx::array q      = to_heads(parts[0]);
    mx::array k      = to_heads(parts[1]);
    mx::array v      = to_heads(parts[2]);
    mx::array q_diff = to_heads(parts[3]);
    mx::array k_diff = to_heads(parts[4]);

    q      = q_norm(q);
    k      = k_norm(k);
    q_diff = q_norm(q_diff);
    k_diff = k_norm(k_diff);

    auto rope = [&](const mx::array& t) {
        return mx::fast::rope(t, ROPE_DIMS, /*traditional=*/false,
                              /*base=*/10000.0f, /*scale=*/1.0f, /*offset=*/0);
    };
    q      = rope(q);
    k      = rope(k);
    q_diff = rope(q_diff);
    k_diff = rope(k_diff);

    mx::array out_main = mx::fast::scaled_dot_product_attention(q,      k,      v, ATTN_SCALE);
    mx::array out_diff = mx::fast::scaled_dot_product_attention(q_diff, k_diff, v, ATTN_SCALE);
    mx::array out = out_main - out_diff;

    out = mx::transpose(out, {0, 2, 1, 3});
    out = mx::reshape(out, {B, T, DIM});
    return mx::matmul(out, mx::transpose(to_out));
}

// ── FeedForward (SiLU GLU) ───────────────────────────────────────────
mx::array FeedForward::operator()(const mx::array& x) const {
    mx::array h = mx::matmul(x, mx::transpose(glu_w)) + glu_b;
    auto parts = mx::split(h, 2, -1);
    mx::array value = parts[0];
    mx::array gate  = parts[1];
    // value * silu(gate)
    mx::array gated = value * (gate * mx::sigmoid(gate));
    return mx::matmul(gated, mx::transpose(out_w)) + out_b;
}

// ── TransformerBlock ─────────────────────────────────────────────────
mx::array TransformerBlock::operator()(const mx::array& x) const {
    mx::array h = x + attn(pre_norm(x));
    return h + ff(ff_norm(h));
}

// ── chunk_midpoint_shift ─────────────────────────────────────────────
mx::array chunk_midpoint_shift(
    const std::vector<TransformerBlock>& blocks,
    const mx::array& x_in,
    int B)
{
    const int internal_T = x_in.shape()[1];

    // First half (blocks 0..2): chunks of EFFECTIVE_CHUNK
    const int nc1 = internal_T / EFFECTIVE_CHUNK;
    mx::array x = mx::reshape(x_in, {B * nc1, EFFECTIVE_CHUNK, DIM});
    x = blocks[0](x);
    x = blocks[1](x);
    x = blocks[2](x);
    x = mx::reshape(x, {B, internal_T, DIM});

    // Shift by SHIFT on both ends → second half (blocks 3..5)
    mx::array left  = mx::slice(x, {0, 0, 0}, {B, SHIFT, DIM});
    mx::array right = mx::slice(x, {0, internal_T - SHIFT, 0}, {B, internal_T, DIM});
    x = mx::concatenate({left, x, right}, 1);                 // [B, internal_T + 2*SHIFT, DIM]
    const int padded_T = internal_T + EFFECTIVE_CHUNK;        // 2*SHIFT == EFFECTIVE_CHUNK
    const int nc2 = padded_T / EFFECTIVE_CHUNK;
    x = mx::reshape(x, {B * nc2, EFFECTIVE_CHUNK, DIM});
    x = blocks[3](x);
    x = blocks[4](x);
    x = blocks[5](x);
    x = mx::reshape(x, {B, padded_T, DIM});
    // Trim SHIFT back off both ends
    x = mx::slice(x, {0, SHIFT, 0}, {B, SHIFT + internal_T, DIM});
    return x;
}

// ── Loaders ──────────────────────────────────────────────────────────
static const mx::array& at(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& key)
{
    auto it = tensors.find(key);
    if (it == tensors.end())
        throw std::runtime_error("missing tensor '" + key + "'");
    return it->second;
}

DyT load_dyt(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype)
{
    auto get = [&](const std::string& k) { return mx::astype(at(tensors, k), dtype); };
    return DyT{get(prefix + ".alpha"), get(prefix + ".gamma"), get(prefix + ".beta")};
}

TransformerBlock load_block(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& p,
    mx::Dtype dtype)
{
    auto get = [&](const std::string& k) { return mx::astype(at(tensors, k), dtype); };
    return TransformerBlock{
        /*pre_norm=*/ load_dyt(tensors, p + "pre_norm", dtype),
        /*attn=*/ DifferentialAttention{
            get(p + "attn.to_qkv.weight"),
            get(p + "attn.to_out.weight"),
            load_dyt(tensors, p + "attn.q_norm", dtype),
            load_dyt(tensors, p + "attn.k_norm", dtype),
        },
        /*ff_norm=*/ load_dyt(tensors, p + "ff_norm", dtype),
        /*ff=*/ FeedForward{
            get(p + "ff.glu_proj.weight"),
            get(p + "ff.glu_proj.bias"),
            get(p + "ff.proj_out.weight"),
            get(p + "ff.proj_out.bias"),
        },
    };
}

}  // namespace sames
}  // namespace sa3
