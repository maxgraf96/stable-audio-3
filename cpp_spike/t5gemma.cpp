#include "t5gemma.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>
#include <sentencepiece_processor.h>

namespace sa3 {
namespace t5g {

// ── rms_norm (Gemma-style: scale by 1 + weight, fp32 internals) ──────
mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    const mx::Dtype dt = x.dtype();
    mx::array x32 = mx::astype(x, mx::float32);
    mx::array var = mx::mean(x32 * x32, /*axes=*/std::vector<int>{-1}, /*keepdims=*/true);
    mx::array n = x32 * mx::rsqrt(var + eps);
    return mx::astype(n * (1.0f + mx::astype(weight, mx::float32)), dt);
}

// ── rotate_half: concat([-x_right_half, x_left_half], -1) ────────────
mx::array rotate_half(const mx::array& x) {
    const auto& s = x.shape();
    const int ndim = static_cast<int>(s.size());
    const int half = s.back() / 2;
    mx::Shape full_stop(s.begin(), s.end());

    mx::Shape start_r(ndim, 0);
    start_r[ndim - 1] = half;
    mx::array right = mx::slice(x, start_r, full_stop);     // x[..., half:]

    mx::Shape start_l(ndim, 0);
    mx::Shape stop_l(s.begin(), s.end());
    stop_l[ndim - 1] = half;
    mx::array left = mx::slice(x, start_l, stop_l);          // x[..., :half]

    return mx::concatenate({-right, left}, -1);
}

// ── rope_cos_sin: precompute cos/sin from per-pair inv_freq ──────────
std::pair<mx::array, mx::array> rope_cos_sin(int seq_len, const mx::array& inv_freq) {
    mx::array pos = mx::arange(0, seq_len, 1, mx::float32);  // (S,)
    mx::array freqs = mx::outer(pos, inv_freq);              // (S, HEAD_DIM/2)
    mx::array emb = mx::concatenate({freqs, freqs}, -1);     // (S, HEAD_DIM)
    return {mx::cos(emb), mx::sin(emb)};
}

// ── apply_rope (in-place mutation of q, k) ───────────────────────────
void apply_rope(mx::array& q, mx::array& k, const mx::array& cos, const mx::array& sin) {
    // Broadcast (S, HEAD_DIM) → (1, 1, S, HEAD_DIM)
    mx::array cos4 = mx::expand_dims(mx::expand_dims(cos, 0), 0);
    mx::array sin4 = mx::expand_dims(mx::expand_dims(sin, 0), 0);
    mx::array cos_q = mx::astype(cos4, q.dtype());
    mx::array sin_q = mx::astype(sin4, q.dtype());
    mx::array cos_k = mx::astype(cos4, k.dtype());
    mx::array sin_k = mx::astype(sin4, k.dtype());
    q = q * cos_q + rotate_half(q) * sin_q;
    k = k * cos_k + rotate_half(k) * sin_k;
}

// ── SelfAttention ────────────────────────────────────────────────────
mx::array SelfAttention::operator()(
    const mx::array& x,
    const mx::array& cos,
    const mx::array& sin,
    const std::optional<mx::array>& add_mask) const
{
    const auto& s = x.shape();
    const int B = s[0], S = s[1];
    constexpr int H = NUM_ATTENTION_HEADS, D = HEAD_DIM;

    auto to_heads = [&](const mx::array& proj) {
        return mx::transpose(mx::reshape(proj, {B, S, H, D}), {0, 2, 1, 3});
    };
    mx::array q = to_heads(mx::matmul(x, mx::transpose(q_proj_w)));
    mx::array k = to_heads(mx::matmul(x, mx::transpose(k_proj_w)));
    mx::array v = to_heads(mx::matmul(x, mx::transpose(v_proj_w)));

    apply_rope(q, k, cos, sin);

    // QK^T scaled, softcap, mask, softmax in fp32, attend, project out.
    // Scalars wrapped in the array's dtype to prevent C++ fp32 promotion of fp16 ops.
    const mx::Dtype qdt = q.dtype();
    mx::array scale_a = mx::array(ATTN_SCALE,              qdt);
    mx::array cap_a   = mx::array(ATTN_LOGIT_SOFTCAPPING,  qdt);
    mx::array qk = mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})) * scale_a;
    qk = mx::tanh(qk / cap_a) * cap_a;
    if (add_mask.has_value()) {
        qk = qk + *add_mask;
    }
    mx::array p = mx::astype(
        mx::softmax(mx::astype(qk, mx::float32), /*axes=*/std::vector<int>{-1}),
        v.dtype());
    mx::array out = mx::matmul(p, v);                              // (B, H, S, D)
    out = mx::transpose(out, {0, 2, 1, 3});                         // (B, S, H, D)
    out = mx::reshape(out, {B, S, H * D});                          // (B, S, HIDDEN)
    return mx::matmul(out, mx::transpose(o_proj_w));
}

// ── MLP (GeGLU with tanh-approx GELU on the gate path) ───────────────
static mx::array gelu_approx(const mx::array& x) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Compute constants in fp64 then cast — match Python's math.sqrt(2/math.pi).
    static const float SQRT_2_OVER_PI = static_cast<float>(std::sqrt(2.0 / M_PI));
    // Every scalar is wrapped in mx::array(value, x.dtype()) — in C++, a raw
    // `float * fp16_array` op promotes the result to fp32 because the scalar's
    // type is fp32; Python's fp64 scalars get cast down to fp16 by MLX's
    // Python bindings, but C++ template overloads don't apply the same coercion.
    const mx::Dtype dt = x.dtype();
    // Match Python's `x ** 3` which calls mx.power, not chained multiplication —
    // different op, different fp16 rounding path.
    mx::array x3 = mx::power(x, mx::array(3, mx::int32));
    mx::array c    = mx::array(0.044715f,       dt);
    mx::array sp   = mx::array(SQRT_2_OVER_PI,  dt);
    mx::array half = mx::array(0.5f,            dt);
    mx::array one  = mx::array(1.0f,            dt);
    mx::array inner = sp * (x + c * x3);
    return half * x * (one + mx::tanh(inner));
}

mx::array MLP::operator()(const mx::array& x) const {
    mx::array gate = mx::matmul(x, mx::transpose(gate_proj_w));   // (B, S, INTERMEDIATE)
    mx::array up   = mx::matmul(x, mx::transpose(up_proj_w));     // (B, S, INTERMEDIATE)
    mx::array hidden = gelu_approx(gate) * up;
    return mx::matmul(hidden, mx::transpose(down_proj_w));         // (B, S, HIDDEN)
}

// ── EncoderLayer ─────────────────────────────────────────────────────
mx::array EncoderLayer::operator()(
    const mx::array& x_in,
    const mx::array& cos,
    const mx::array& sin,
    const std::optional<mx::array>& add_mask) const
{
    mx::array h = rms_norm(x_in, pre_self_attn_norm);
    h = self_attn(h, cos, sin, add_mask);
    h = rms_norm(h, post_self_attn_norm);
    mx::array x = x_in + h;

    h = rms_norm(x, pre_feedforward_norm);
    h = mlp(h);
    h = rms_norm(h, post_feedforward_norm);
    return x + h;
}

// ── Encoder forward ──────────────────────────────────────────────────
mx::array Encoder::operator()(
    const mx::array& input_ids,
    const std::optional<mx::array>& attention_mask) const
{
    // Embedding lookup + scale by sqrt(HIDDEN_SIZE)
    mx::array x = mx::take(embed_tokens_w, input_ids, /*axis=*/0);
    x = mx::astype(x, mx::float16);
    // Match Python: scalar (fp64 sqrt) cast to x.dtype before multiply.
    float normalizer = static_cast<float>(std::sqrt(static_cast<double>(HIDDEN_SIZE)));
    x = x * mx::array(normalizer, x.dtype());

    const int S = x.shape()[1];
    auto cs = rope_cos_sin(S, rope_inv_freq);
    mx::array cos = cs.first;
    mx::array sin = cs.second;

    std::optional<mx::array> add_mask;
    if (attention_mask.has_value()) {
        mx::array keep = mx::astype(*attention_mask, mx::float32);
        // (B, S) → (B, 1, 1, S) of -1e9 on padded positions, 0 on real
        mx::array m = (1.0f - keep) * -1e9f;
        m = mx::expand_dims(mx::expand_dims(m, 1), 1);   // (B, 1, 1, S)
        add_mask = mx::astype(m, x.dtype());
    }

    for (const auto& layer : layers) {
        x = layer(x, cos, sin, add_mask);
    }
    return rms_norm(x, norm_w);
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

Encoder load_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    Encoder e{
        /*embed_tokens_w=*/ mx::astype(at(tensors, "embed_tokens.weight"), dtype),
        /*layers=*/         {},
        /*norm_w=*/         mx::astype(at(tensors, "norm.weight"), dtype),
        // rope_inv_freq is always fp32 (Python casts on load).
        /*rope_inv_freq=*/  mx::astype(at(tensors, "rope_inv_freq"), mx::float32),
    };
    e.layers.reserve(NUM_HIDDEN_LAYERS);
    for (int i = 0; i < NUM_HIDDEN_LAYERS; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        e.layers.push_back(EncoderLayer{
            /*pre_self_attn_norm=*/    mx::astype(at(tensors, p + "pre_self_attn_layernorm.weight"),    dtype),
            /*post_self_attn_norm=*/   mx::astype(at(tensors, p + "post_self_attn_layernorm.weight"),   dtype),
            /*pre_feedforward_norm=*/  mx::astype(at(tensors, p + "pre_feedforward_layernorm.weight"),  dtype),
            /*post_feedforward_norm=*/ mx::astype(at(tensors, p + "post_feedforward_layernorm.weight"), dtype),
            /*self_attn=*/ SelfAttention{
                mx::astype(at(tensors, p + "self_attn.q_proj.weight"), dtype),
                mx::astype(at(tensors, p + "self_attn.k_proj.weight"), dtype),
                mx::astype(at(tensors, p + "self_attn.v_proj.weight"), dtype),
                mx::astype(at(tensors, p + "self_attn.o_proj.weight"), dtype),
            },
            /*mlp=*/ MLP{
                mx::astype(at(tensors, p + "mlp.gate_proj.weight"), dtype),
                mx::astype(at(tensors, p + "mlp.up_proj.weight"),   dtype),
                mx::astype(at(tensors, p + "mlp.down_proj.weight"), dtype),
            },
        });
    }
    return e;
}

// ── Tokenizer (pimpl wrapping SentencePieceProcessor) ───────────────
struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
};

Tokenizer::Tokenizer() : impl(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void Tokenizer::load(std::string_view serialized_proto) {
    auto status = impl->sp.LoadFromSerializedProto(serialized_proto);
    if (!status.ok()) {
        throw std::runtime_error(
            "SentencePiece LoadFromSerializedProto failed: " + status.ToString());
    }
}

std::pair<mx::array, mx::array> Tokenizer::tokenize(
    const std::vector<std::string>& prompts, int max_len) const
{
    const int B = static_cast<int>(prompts.size());
    std::vector<int32_t> ids_buf(B * max_len, PAD_TOKEN_ID);
    std::vector<int32_t> mask_buf(B * max_len, 0);
    for (int i = 0; i < B; ++i) {
        std::vector<int> tok = impl->sp.EncodeAsIds(prompts[i]);
        const int n = std::min<int>(static_cast<int>(tok.size()), max_len);
        for (int j = 0; j < n; ++j) {
            ids_buf[i * max_len + j]  = static_cast<int32_t>(tok[j]);
            mask_buf[i * max_len + j] = 1;
        }
    }
    mx::array ids(ids_buf.data(),  mx::Shape{B, max_len}, mx::int32);
    mx::array mask(mask_buf.data(), mx::Shape{B, max_len}, mx::int32);
    return {ids, mask};
}

// ── T5Gemma::encode (tokenize → forward with empty-row guard) ───────
std::pair<mx::array, mx::array> T5Gemma::encode(
    const std::vector<std::string>& prompts, int max_len) const
{
    auto tk = tokenizer.tokenize(prompts, max_len);
    mx::array ids  = tk.first;
    mx::array mask = tk.second;

    // Empty-row guard: an all-zero mask row makes softmax see all -inf → NaN.
    // Set position 0 to 1 for the forward pass, but RETURN the original mask
    // so downstream code still treats those positions as padding.
    mx::eval(mask);
    const int B = ids.shape()[0];
    const int32_t* mask_data = mask.data<int32_t>();
    std::vector<int32_t> fixed;
    bool any_empty = false;
    for (int i = 0; i < B; ++i) {
        bool empty = true;
        for (int j = 0; j < max_len; ++j) {
            if (mask_data[i * max_len + j]) { empty = false; break; }
        }
        if (empty) { any_empty = true; break; }
    }
    if (any_empty) {
        fixed.assign(mask_data, mask_data + B * max_len);
        for (int i = 0; i < B; ++i) {
            bool empty = true;
            for (int j = 0; j < max_len; ++j) {
                if (fixed[i * max_len + j]) { empty = false; break; }
            }
            if (empty) fixed[i * max_len + 0] = 1;
        }
    }
    mx::array out = any_empty
        ? encoder(ids, mx::array(fixed.data(), mx::Shape{B, max_len}, mx::int32))
        : encoder(ids, mask);
    return {out, mask};
}

// ── load_t5gemma (encoder + tokenizer from one safetensors map) ─────
T5Gemma load_t5gemma(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype)
{
    T5Gemma t{ load_encoder(tensors, dtype), Tokenizer{} };

    // Extract SentencePiece serialized proto from the uint8 "TOKENIZER_MODEL"
    // array. The tensor is loaded from safetensors as a 1-D uint8 buffer which
    // is row-contiguous, so .data<uint8_t>() gives the raw bytes directly.
    auto it = tensors.find("TOKENIZER_MODEL");
    if (it == tensors.end()) {
        throw std::runtime_error("missing 'TOKENIZER_MODEL' in safetensors");
    }
    mx::array tok_arr = it->second;
    mx::eval(tok_arr);
    const uint8_t* bytes = tok_arr.data<uint8_t>();
    const size_t n = tok_arr.size();
    t.tokenizer.load(std::string_view(reinterpret_cast<const char*>(bytes), n));
    return t;
}

}  // namespace t5g
}  // namespace sa3
