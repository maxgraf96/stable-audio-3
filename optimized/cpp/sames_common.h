// Shared SAME-S components (encoder + decoder use the same blocks).
//
// Ported from optimized/mlx/models/defs/same_s_decoder.py:
//   - DyT                    (Dynamic Tanh normalization: gamma * tanh(alpha*x) + beta)
//   - GLU_FF                 (SwiGLU feedforward, FF_INNER=2304, SiLU only — no sin gate)
//   - DifferentialAttention  (5-way QKV split + two SDPAs subtracted, RoPE, DyT norm.
//                             Unlike SAME-L there is NO sliding window — the math is
//                             plain differential attention over a 34-token chunk.)
//   - TransformerBlock       (pre_norm → attn → ff_norm → ff with residual)
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

namespace sa3 {
namespace sames {

namespace mx = mlx::core;

// ── Constants (SAME-S from sa3-sm-music / sa3-sm-sfx) ────────────────
constexpr int LATENT_DIM       = 256;
constexpr int DIM              = 768;
constexpr int NUM_HEADS        = 12;
constexpr int HEAD_DIM         = 64;
constexpr int ROPE_DIMS        = 32;
constexpr int NUM_BLOCKS       = 6;
constexpr int FF_INNER         = 2304;       // ff_mult = 3
constexpr int OUT_CHANNELS     = 512;
constexpr int STRIDE           = 16;
constexpr int SUB_CHUNK_SIZE   = STRIDE + 1; // 17 (1 latent + 16 new-token positions)
constexpr int CHUNK_SIZE_LAT   = 32;         // latent-side chunk size for half-shift
constexpr int EFFECTIVE_CHUNK  = 34;         // CHUNK_SIZE_LAT + CHUNK_SIZE_LAT / STRIDE
constexpr int SHIFT            = 17;         // EFFECTIVE_CHUNK / 2 — second-half shift
constexpr int SIN_PER_POS      = SUB_CHUNK_SIZE - 1; // 16
constexpr int IN_CHANNELS      = 512;        // encoder input channels

// HEAD_DIM ** -0.5 = 64 ** -0.5 = 0.125, exact in fp32
constexpr float ATTN_SCALE = 0.125f;

// ── DyT: gamma * tanh(alpha * x) + beta ──────────────────────────────
// Same math as samel::DyT but kept self-contained in the sames namespace
// to avoid a cross-namespace dependency between the two autoencoders.
struct DyT {
    mx::array alpha;   // (1,)
    mx::array gamma;   // (dim,)
    mx::array beta;    // (dim,)

    mx::array operator()(const mx::array& x) const;
};

// ── GLU_FF: SwiGLU feedforward (SiLU only, no sin variant) ───────────
struct GLU_FF {
    mx::array glu_proj_w;   // (FF_INNER * 2, DIM)
    mx::array glu_proj_b;   // (FF_INNER * 2,)
    mx::array proj_out_w;   // (DIM, FF_INNER)
    mx::array proj_out_b;   // (DIM,)

    mx::array operator()(const mx::array& x) const;
};

// ── DifferentialAttention: 5-way QKV + two SDPAs subtracted ──────────
// Operates on (B', T, DIM) where T is the chunk length (34 in practice).
// No sliding window, no mask — full self-attention within the chunk.
struct DifferentialAttention {
    mx::array to_qkv;   // (5*DIM, DIM)
    mx::array to_out;   // (DIM, DIM)
    DyT q_norm;         // alpha (1,), gamma (HEAD_DIM,), beta (HEAD_DIM,)
    DyT k_norm;

    mx::array operator()(const mx::array& x) const;
};

// Single transformer block: residual + attn + residual + ff.
struct TransformerBlock {
    DyT pre_norm;
    DifferentialAttention attn;
    DyT ff_norm;
    GLU_FF ff;

    mx::array operator()(const mx::array& x) const;
};

// ── Loaders ──────────────────────────────────────────────────────────
DyT load_dyt(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

GLU_FF load_glu_ff(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

DifferentialAttention load_differential_attention(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

TransformerBlock load_transformer_block(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

}  // namespace sames
}  // namespace sa3
