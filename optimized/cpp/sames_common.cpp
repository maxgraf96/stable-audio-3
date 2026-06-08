#include "sames_common.h"

#include <cmath>
#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace sames {

// ── DyT ──────────────────────────────────────────────────────────────
mx::array DyT::operator()(const mx::array& x) const {
    return gamma * mx::tanh(alpha * x) + beta;
}

// ── GLU_FF (SwiGLU, SiLU only) ───────────────────────────────────────
mx::array GLU_FF::operator()(const mx::array& x) const {
    mx::array h = mx::matmul(x, mx::transpose(glu_proj_w)) + glu_proj_b;
    auto parts = mx::split(h, 2, -1);
    const mx::array& value = parts[0];
    const mx::array& gate  = parts[1];
    // SiLU: gate * sigmoid(gate)
    mx::array activated = value * (gate * mx::sigmoid(gate));
    return mx::matmul(activated, mx::transpose(proj_out_w)) + proj_out_b;
}

// ── DifferentialAttention ────────────────────────────────────────────
// 5-way QKV split; two SDPAs over (q,k,v) and (q_diff,k_diff,v), subtracted.
// No mask, no sliding window — straight self-attention within the chunk.
mx::array DifferentialAttention::operator()(const mx::array& x) const {
    const auto& s = x.shape();
    const int B = s[0], T = s[1];
    constexpr int H = NUM_HEADS, D = HEAD_DIM;

    mx::array qkv = mx::matmul(x, mx::transpose(to_qkv));  // (B, T, 5*DIM)
    auto parts = mx::split(qkv, 5, -1);
    auto to_heads = [&](const mx::array& t) {
        return mx::transpose(mx::reshape(t, {B, T, H, D}), {0, 2, 1, 3});
    };
    mx::array q      = to_heads(parts[0]);
    mx::array k      = to_heads(parts[1]);
    mx::array v      = to_heads(parts[2]);
    mx::array q_diff = to_heads(parts[3]);
    mx::array k_diff = to_heads(parts[4]);

    // DyT normalisation on Q/K (NOT RMSNorm — SAME-S uses DyT throughout).
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

// ── TransformerBlock ─────────────────────────────────────────────────
mx::array TransformerBlock::operator()(const mx::array& x) const {
    mx::array h = x + attn(pre_norm(x));
    return h + ff(ff_norm(h));
}

// ── Loaders ──────────────────────────────────────────────────────────
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

DyT load_dyt(
    const std::unordered_map<std::string, mx::array>& t,
    const std::string& prefix,
    mx::Dtype dt)
{
    return DyT{
        mx::astype(at(t, prefix + "alpha"), dt),
        mx::astype(at(t, prefix + "gamma"), dt),
        mx::astype(at(t, prefix + "beta"),  dt),
    };
}

GLU_FF load_glu_ff(
    const std::unordered_map<std::string, mx::array>& t,
    const std::string& prefix,
    mx::Dtype dt)
{
    return GLU_FF{
        mx::astype(at(t, prefix + "glu_proj.weight"), dt),
        mx::astype(at(t, prefix + "glu_proj.bias"),   dt),
        mx::astype(at(t, prefix + "proj_out.weight"), dt),
        mx::astype(at(t, prefix + "proj_out.bias"),   dt),
    };
}

DifferentialAttention load_differential_attention(
    const std::unordered_map<std::string, mx::array>& t,
    const std::string& prefix,
    mx::Dtype dt)
{
    return DifferentialAttention{
        mx::astype(at(t, prefix + "to_qkv.weight"), dt),
        mx::astype(at(t, prefix + "to_out.weight"), dt),
        load_dyt(t, prefix + "q_norm.", dt),
        load_dyt(t, prefix + "k_norm.", dt),
    };
}

TransformerBlock load_transformer_block(
    const std::unordered_map<std::string, mx::array>& t,
    const std::string& prefix,
    mx::Dtype dt)
{
    return TransformerBlock{
        /*pre_norm=*/ load_dyt(t, prefix + "pre_norm.", dt),
        /*attn=*/     load_differential_attention(t, prefix + "attn.", dt),
        /*ff_norm=*/  load_dyt(t, prefix + "ff_norm.", dt),
        /*ff=*/       load_glu_ff(t, prefix + "ff.", dt),
    };
}

}  // namespace sames
}  // namespace sa3
