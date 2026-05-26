#include "samel_common.h"

#include <cmath>
#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace samel {

// ── DyT ──────────────────────────────────────────────────────────────
mx::array DyT::operator()(const mx::array& x) const {
    return gamma * mx::tanh(alpha * x) + beta;
}

// ── FeedForward ──────────────────────────────────────────────────────
mx::array FeedForward::operator()(const mx::array& x) const {
    // x: (..., DIM)
    mx::array h = mx::matmul(x, mx::transpose(glu_proj_w)) + glu_proj_b;
    auto parts = mx::split(h, 2, -1);
    mx::array value = parts[0];
    mx::array gate  = parts[1];
    // silu(gate) = gate * sigmoid(gate). mx::array has no default constructor,
    // so initialize via ternary at declaration.
    mx::array activated = use_sin
        ? value * mx::sin(gate * static_cast<float>(M_PI))
        : value * (gate * mx::sigmoid(gate));
    return mx::matmul(activated, mx::transpose(proj_out_w)) + proj_out_b;
}

// ── to_row_contiguous ────────────────────────────────────────────────
// `array + 0` gets elided by MLX. Forcing a copy via flatten→reshape
// guarantees a fresh row-contiguous buffer regardless of input stride layout.
mx::array to_row_contiguous(const mx::array& a) {
    mx::Shape shape{a.shape().begin(), a.shape().end()};
    return mx::reshape(mx::flatten(a), shape);
}

// ── SWA mask ─────────────────────────────────────────────────────────
const mx::array& swa_mask() {
    static const mx::array cached = []() {
        // q index in (BLOCK_SIZE, 1), kv index in (1, 3*BLOCK_SIZE)
        mx::array q  = mx::expand_dims(mx::arange(0, BLOCK_SIZE, 1, mx::int32), -1);
        mx::array kv = mx::expand_dims(mx::arange(0, 3 * BLOCK_SIZE, 1, mx::int32), 0);
        mx::array valid = (kv >= q) & (kv <= (q + 2 * BLOCK_SIZE));
        mx::array m = mx::where(valid,
                                mx::array(0.0f, mx::float32),
                                mx::array(-1e9f, mx::float32));
        mx::eval(m);
        return m;
    }();
    return cached;
}

// ── DifferentialSWA ──────────────────────────────────────────────────
// Internal: _diff_sdpa branch (full attention; T <= BLOCK_SIZE).
static mx::array diff_sdpa(
    const mx::array& q1, const mx::array& k1, const mx::array& v,
    const mx::array& q2, const mx::array& k2,
    const std::optional<mx::array>& mask)
{
    mx::array Q = mx::concatenate({q1, q2}, 1);
    mx::array K = mx::concatenate({k1, k2}, 1);
    mx::array V = mx::concatenate({v, v}, 1);
    mx::array out = mask.has_value()
        ? mx::fast::scaled_dot_product_attention(Q, K, V, ATTN_SCALE, "", *mask)
        : mx::fast::scaled_dot_product_attention(Q, K, V, ATTN_SCALE);
    auto parts = mx::split(out, 2, 1);
    return parts[0] - parts[1];
}

// Internal: _swa branch (sliding window via strided KV view).
static mx::array swa_branch(
    const mx::array& q1, const mx::array& k1, const mx::array& v,
    const mx::array& q2, const mx::array& k2,
    const mx::array& mask)
{
    const auto& sh = q1.shape();
    const int B = sh[0], H = sh[1], T = sh[2], D = sh[3];
    const int G = T / BLOCK_SIZE;
    const int W = 3 * BLOCK_SIZE;  // 51

    // Pad K/V by one BLOCK on each side of the sequence axis (axis 2).
    const std::vector<std::pair<int, int>> pad_w = {{0,0}, {0,0}, {BLOCK_SIZE, BLOCK_SIZE}, {0,0}};
    mx::array k1p = mx::pad(k1, pad_w);
    mx::array k2p = mx::pad(k2, pad_w);
    mx::array vp  = mx::pad(v,  pad_w);

    const int Tp = T + 2 * BLOCK_SIZE;
    const mx::Strides win_strides = {
        static_cast<int64_t>(H) * Tp * D,
        static_cast<int64_t>(Tp) * D,
        static_cast<int64_t>(BLOCK_SIZE) * D,
        static_cast<int64_t>(D),
        static_cast<int64_t>(1),
    };
    const mx::Shape win_shape = {B, H, G, W, D};

    mx::array k1w = mx::as_strided(k1p, win_shape, win_strides, 0);
    mx::array k2w = mx::as_strided(k2p, win_shape, win_strides, 0);
    mx::array vw  = mx::as_strided(vp,  win_shape, win_strides, 0);

    mx::array q1g = mx::reshape(q1, {B, H, G, BLOCK_SIZE, D});
    mx::array q2g = mx::reshape(q2, {B, H, G, BLOCK_SIZE, D});

    // Boundary mask: zero out attention to padded KV positions at sequence edges.
    mx::array g_idx = mx::expand_dims(mx::arange(0, G, 1, mx::int32), -1);   // (G, 1)
    mx::array w_idx = mx::expand_dims(mx::arange(0, W, 1, mx::int32), 0);    // (1, W)
    mx::array padded_pos = g_idx * BLOCK_SIZE + w_idx;                       // (G, W)
    mx::array boundary = mx::where(
        (padded_pos >= BLOCK_SIZE) & (padded_pos < (T + BLOCK_SIZE)),
        mx::array(0.0f, q1.dtype()),
        mx::array(-1e9f, q1.dtype()));
    boundary = mx::expand_dims(boundary, 1);                                 // (G, 1, W)

    mx::array combined = mask + boundary;                                    // (G, 17, 51)
    combined = mx::broadcast_to(
        mx::expand_dims(combined, 0), {B, G, BLOCK_SIZE, W});               // (B, G, 17, 51)
    combined = mx::reshape(combined, {B * G, 1, BLOCK_SIZE, W});             // (B*G, 1, 17, 51)

    q1g = mx::reshape(mx::transpose(q1g, {0, 2, 1, 3, 4}), {B * G, H, BLOCK_SIZE, D});
    q2g = mx::reshape(mx::transpose(q2g, {0, 2, 1, 3, 4}), {B * G, H, BLOCK_SIZE, D});
    k1w = mx::reshape(mx::transpose(k1w, {0, 2, 1, 3, 4}), {B * G, H, W, D});
    k2w = mx::reshape(mx::transpose(k2w, {0, 2, 1, 3, 4}), {B * G, H, W, D});
    vw  = mx::reshape(mx::transpose(vw,  {0, 2, 1, 3, 4}), {B * G, H, W, D});

    mx::array Q = mx::concatenate({q1g, q2g}, 1);
    mx::array K = mx::concatenate({k1w, k2w}, 1);
    mx::array V = mx::concatenate({vw,  vw},  1);

    mx::array out = mx::fast::scaled_dot_product_attention(
        Q, K, V, ATTN_SCALE, /*mask_mode=*/"", combined);
    auto parts = mx::split(out, 2, 1);
    mx::array diff = parts[0] - parts[1];

    // (B*G, H, 17, D) → (B, H, T, D)
    diff = mx::reshape(diff, {B, G, H, BLOCK_SIZE, D});
    diff = mx::transpose(diff, {0, 2, 1, 3, 4});
    return mx::reshape(diff, {B, H, T, D});
}

mx::array DifferentialSWA::operator()(
    const mx::array& x,
    const std::optional<mx::array>& mask,
    bool full_attention) const
{
    const auto& sh = x.shape();
    const int B = sh[0], T = sh[1];
    constexpr int H = NUM_HEADS, D = HEAD_DIM;

    // qkv: (B, T, 5*DIM) — no bias on to_qkv
    mx::array qkv = mx::matmul(x, mx::transpose(to_qkv));
    auto parts = mx::split(qkv, 5, -1);
    mx::array q1 = parts[0];
    mx::array k1 = parts[1];
    mx::array v  = parts[2];
    mx::array q2 = parts[3];
    mx::array k2 = parts[4];

    auto to_heads = [&](const mx::array& t) {
        return mx::transpose(mx::reshape(t, {B, T, H, D}), {0, 2, 1, 3});
    };
    q1 = to_heads(q1); k1 = to_heads(k1); v = to_heads(v);
    q2 = to_heads(q2); k2 = to_heads(k2);

    q1 = q_norm(q1); k1 = k_norm(k1);
    q2 = q_norm(q2); k2 = k_norm(k2);

    auto apply_rope = [&](const mx::array& t) {
        return mx::fast::rope(t, ROPE_DIMS, /*traditional=*/false,
                              /*base=*/10000.0f, /*scale=*/1.0f, /*offset=*/0);
    };
    q1 = apply_rope(q1); k1 = apply_rope(k1);
    q2 = apply_rope(q2); k2 = apply_rope(k2);

    mx::array out = [&]() {
        if (full_attention || T <= BLOCK_SIZE) {
            return diff_sdpa(q1, k1, v, q2, k2, mask);
        }
        if (!mask.has_value()) {
            throw std::runtime_error("DifferentialSWA::swa requires a mask");
        }
        return swa_branch(q1, k1, v, q2, k2, *mask);
    }();

    // (B, H, T, D) → (B, T, DIM)
    out = mx::transpose(out, {0, 2, 1, 3});
    out = mx::reshape(out, {B, T, DIM});
    return mx::matmul(out, mx::transpose(to_out));
}

// ── Loaders ──────────────────────────────────────────────────────────
static const mx::array& get_tensor(
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
        mx::astype(get_tensor(t, prefix + "alpha"), dt),
        mx::astype(get_tensor(t, prefix + "gamma"), dt),
        mx::astype(get_tensor(t, prefix + "beta"),  dt),
    };
}

FeedForward load_feedforward(
    const std::unordered_map<std::string, mx::array>& t,
    const std::string& prefix,
    mx::Dtype dt,
    bool use_sin)
{
    return FeedForward{
        mx::astype(get_tensor(t, prefix + "glu_proj.weight"), dt),
        mx::astype(get_tensor(t, prefix + "glu_proj.bias"),   dt),
        mx::astype(get_tensor(t, prefix + "proj_out.weight"), dt),
        mx::astype(get_tensor(t, prefix + "proj_out.bias"),   dt),
        use_sin,
    };
}

DifferentialSWA load_differential_swa(
    const std::unordered_map<std::string, mx::array>& t,
    const std::string& prefix,
    mx::Dtype dt)
{
    return DifferentialSWA{
        mx::astype(get_tensor(t, prefix + "to_qkv.weight"), dt),
        mx::astype(get_tensor(t, prefix + "to_out.weight"), dt),
        load_dyt(t, prefix + "q_norm.", dt),
        load_dyt(t, prefix + "k_norm.", dt),
    };
}

// ── TransformerBlock ─────────────────────────────────────────────────
mx::array TransformerBlock::operator()(
    const mx::array& x,
    const std::optional<mx::array>& mask,
    bool full_attention) const
{
    mx::array h = x + attn(pre_norm(x), mask, full_attention);
    return h + ff(ff_norm(h));
}

TransformerBlock load_transformer_block(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype,
    bool use_sin)
{
    return TransformerBlock{
        /*pre_norm=*/ load_dyt(tensors, prefix + "pre_norm.", dtype),
        /*attn=*/     load_differential_swa(tensors, prefix + "attn.", dtype),
        /*ff_norm=*/  load_dyt(tensors, prefix + "ff_norm.", dtype),
        /*ff=*/       load_feedforward(tensors, prefix + "ff.", dtype, use_sin),
    };
}

}  // namespace samel
}  // namespace sa3
