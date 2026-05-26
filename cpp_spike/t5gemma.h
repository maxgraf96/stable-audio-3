// T5Gemma encoder (google/t5gemma-b-b-ul2) — Gemma2-style architecture.
//
// 12 layers · dim 768 · 12 heads · head_dim 64 · GeGLU(2048) · RMSNorm(1+w)
// RoPE (θ=10000, half-half layout) · attn logit softcap=50
//
// Input:  (B, S) int32 token ids + (B, S) int32 attention mask (1=real, 0=pad)
// Output: (B, S, 768) fp16
//
// Ported from optimized/mlx/models/defs/t5gemma_mlx.py. The encoder forward
// path is exercised directly with deterministic token ids by test_t5gemma.cpp
// — SentencePiece tokenization is decoupled and not required for the
// bit-exact verification.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

namespace sa3 {
namespace t5g {

namespace mx = mlx::core;

// ── Config (hardcoded — matches META in t5gemma_f16.npz exactly) ──────
constexpr int   HIDDEN_SIZE              = 768;
constexpr int   NUM_HIDDEN_LAYERS        = 12;
constexpr int   NUM_ATTENTION_HEADS      = 12;
constexpr int   NUM_KEY_VALUE_HEADS      = 12;
constexpr int   HEAD_DIM                 = 64;
constexpr int   INTERMEDIATE_SIZE        = 2048;
constexpr int   VOCAB_SIZE               = 256000;
constexpr int   MAX_POSITION_EMBEDDINGS  = 8192;
constexpr float ROPE_THETA               = 10000.0f;
constexpr float RMS_NORM_EPS             = 1e-6f;
constexpr float ATTN_LOGIT_SOFTCAPPING   = 50.0f;
constexpr int   QUERY_PRE_ATTN_SCALAR    = 64;     // sqrt(64)^-1 = 0.125, exact fp32
constexpr int   PAD_TOKEN_ID             = 0;

constexpr float ATTN_SCALE = 0.125f;               // = QUERY_PRE_ATTN_SCALAR ** -0.5

// ── Helpers ──────────────────────────────────────────────────────────
// Gemma-style RMSNorm: normalize in fp32, scale by (1 + weight), cast back.
mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps = RMS_NORM_EPS);

// Half-half rotation: concat([-x_right_half, x_left_half], -1).
mx::array rotate_half(const mx::array& x);

// Precompute cos/sin of shape (S, HEAD_DIM) from per-pair inv_freq (HEAD_DIM/2,).
std::pair<mx::array, mx::array> rope_cos_sin(int seq_len, const mx::array& inv_freq);

// Apply RoPE in-place (returns rotated q, k).  cos/sin shape (S, HEAD_DIM).
void apply_rope(mx::array& q, mx::array& k, const mx::array& cos, const mx::array& sin);

// ── Model blocks ─────────────────────────────────────────────────────
struct SelfAttention {
    mx::array q_proj_w;  // (HIDDEN, HIDDEN)
    mx::array k_proj_w;  // (HIDDEN, HIDDEN)
    mx::array v_proj_w;  // (HIDDEN, HIDDEN)
    mx::array o_proj_w;  // (HIDDEN, HIDDEN)

    mx::array operator()(
        const mx::array& x,
        const mx::array& cos,
        const mx::array& sin,
        const std::optional<mx::array>& add_mask) const;
};

struct MLP {
    mx::array gate_proj_w;  // (INTERMEDIATE, HIDDEN)
    mx::array up_proj_w;    // (INTERMEDIATE, HIDDEN)
    mx::array down_proj_w;  // (HIDDEN, INTERMEDIATE)

    mx::array operator()(const mx::array& x) const;
};

struct EncoderLayer {
    mx::array pre_self_attn_norm;     // (HIDDEN,)
    mx::array post_self_attn_norm;    // (HIDDEN,)
    mx::array pre_feedforward_norm;   // (HIDDEN,)
    mx::array post_feedforward_norm;  // (HIDDEN,)
    SelfAttention self_attn;
    MLP mlp;

    mx::array operator()(
        const mx::array& x,
        const mx::array& cos,
        const mx::array& sin,
        const std::optional<mx::array>& add_mask) const;
};

struct Encoder {
    mx::array embed_tokens_w;  // (VOCAB, HIDDEN)
    std::vector<EncoderLayer> layers;  // NUM_HIDDEN_LAYERS
    mx::array norm_w;          // (HIDDEN,) final RMSNorm weight
    mx::array rope_inv_freq;   // (HEAD_DIM/2,) fp32

    // input_ids: (B, S) int32; attention_mask: (B, S) int32 (1=real, 0=pad)
    // Returns (B, S, HIDDEN) fp16.
    mx::array operator()(
        const mx::array& input_ids,
        const std::optional<mx::array>& attention_mask) const;
};

// ── Loader ───────────────────────────────────────────────────────────
// Encoder weights load as fp16 (Python default).  Output dtype follows that.
Encoder load_encoder(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float16);

// ── Tokenizer ────────────────────────────────────────────────────────
// Thin wrapper around SentencePieceProcessor.  Pimpl'd so this header doesn't
// drag the sentencepiece include into every translation unit that uses T5Gemma.
struct Tokenizer {
    struct Impl;
    std::unique_ptr<Impl> impl;

    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Load tokenizer from a SentencePiece serialized proto blob.
    void load(std::string_view serialized_proto);

    // Tokenize a list of strings into (ids, mask) arrays:
    //   ids:  (B, max_len) int32, right-padded with PAD_TOKEN_ID
    //   mask: (B, max_len) int32, 1 = real token, 0 = pad
    // Strings longer than max_len are truncated.
    std::pair<mx::array, mx::array> tokenize(
        const std::vector<std::string>& prompts,
        int max_len = 256) const;
};

// ── T5Gemma — encoder + tokenizer, full string-in → embeds-out API ───
struct T5Gemma {
    Encoder encoder;
    Tokenizer tokenizer;

    // Tokenize + encode in one call.
    // Returns:
    //   embeds: (B, max_len, HIDDEN) fp16 — last_hidden_state
    //   mask:   (B, max_len) int32      — same mask as tokenize(); consumers
    //                                     should consult it because pad
    //                                     positions still hold numeric junk.
    //
    // Internally restores Python's empty-prompt guard: rows with an all-zero
    // mask would NaN through softmax, so for the forward pass we set position
    // 0 to 1.  The returned mask still has 0 there so downstream code stays
    // correct.
    std::pair<mx::array, mx::array> encode(
        const std::vector<std::string>& prompts,
        int max_len = 256) const;
};

// Construct a complete T5Gemma from a safetensors-loaded tensor map, including
// the SentencePiece tokenizer extracted from the "TOKENIZER_MODEL" uint8 array.
T5Gemma load_t5gemma(
    const std::unordered_map<std::string, mx::array>& tensors,
    mx::Dtype dtype = mx::float16);

}  // namespace t5g
}  // namespace sa3
