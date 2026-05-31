// Shared SAME-S building blocks (differential attention WITHOUT sliding window,
// DyT norm, SiLU GLU FF, transformer block, chunk-midpoint-shift helper). Used
// by both the SAME-S encoder and decoder.
//
// Differs from SAME-L (samel_common): smaller dims (DIM=768, HEADS=12,
// BLOCKS=6, FF=2304), NO sliding-window attention (always full), NO sinusoidal
// FF gate (always SiLU), and a two-half "chunk_midpoint_shift" sequence pass
// instead of a single all-blocks pass.
//
// Ported from optimized/mlx/models/defs/same_s_{encoder,decoder}.py.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

namespace sa3 {
namespace sames {

namespace mx = mlx::core;

// ── Constants (SAME-S) ───────────────────────────────────────────────
constexpr int   LATENT_DIM      = 256;
constexpr int   DIM             = 768;
constexpr int   NUM_HEADS       = 12;
constexpr int   HEAD_DIM        = 64;
constexpr int   ROPE_DIMS       = 32;
constexpr int   NUM_BLOCKS      = 6;
constexpr int   FF_INNER        = 2304;  // ff_mult=3
constexpr int   OUT_CHANNELS    = 512;
constexpr int   IN_CHANNELS     = 512;
constexpr int   STRIDE          = 16;
constexpr int   SUB_CHUNK_SIZE  = 17;    // STRIDE + 1
constexpr int   SIN_PER_POS     = 16;    // SUB_CHUNK_SIZE - 1
constexpr int   CHUNK_SIZE_LAT  = 32;
constexpr int   EFFECTIVE_CHUNK = 34;    // CHUNK_SIZE_LAT + CHUNK_SIZE_LAT/STRIDE
constexpr int   SHIFT           = 17;    // EFFECTIVE_CHUNK / 2
constexpr float ATTN_SCALE      = 0.125f;  // 1/sqrt(HEAD_DIM=64)

// ── DyT: gamma * tanh(alpha * x) + beta ──────────────────────────────
struct DyT {
    mx::array alpha;  // (1,)
    mx::array gamma;  // (dim,)
    mx::array beta;   // (dim,)
    mx::array operator()(const mx::array& x) const;
};

// ── Differential attention (5×QKV split, two SDPAs, subtract). Full
//    attention — SAME-S has no sliding window. ──────────────────────────
struct DifferentialAttention {
    mx::array to_qkv;   // (5*DIM, DIM)
    mx::array to_out;   // (DIM, DIM)
    DyT q_norm;
    DyT k_norm;
    mx::array operator()(const mx::array& x) const;
};

// ── GLU feedforward (SiLU gate only — no sinusoidal in SAME-S) ────────
struct FeedForward {
    mx::array glu_w;    // (2*FF_INNER, DIM)
    mx::array glu_b;    // (2*FF_INNER,)
    mx::array out_w;    // (DIM, FF_INNER)
    mx::array out_b;    // (DIM,)
    mx::array operator()(const mx::array& x) const;
};

// ── Transformer block (DyT norms, diff attention, GLU FF) ────────────
struct TransformerBlock {
    DyT pre_norm;
    DifferentialAttention attn;
    DyT ff_norm;
    FeedForward ff;
    mx::array operator()(const mx::array& x) const;
};

// ── chunk_midpoint_shift: the SAME-S two-half sequence pass ───────────
// Input/output x: [B, internal_T, DIM] where internal_T = T_lat * SUB_CHUNK_SIZE
// (T_lat even, so internal_T % EFFECTIVE_CHUNK == 0). blocks must have 6 entries.
//   1. blocks[0..2] over chunks of EFFECTIVE_CHUNK(34)
//   2. shift SHIFT(17) on both ends, blocks[3..5] over the padded chunks
//   3. trim SHIFT back off both ends
mx::array chunk_midpoint_shift(
    const std::vector<TransformerBlock>& blocks,
    const mx::array& x,
    int B);

// ── DyT loader helper ─────────────────────────────────────────────────
DyT load_dyt(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

// Build a TransformerBlock from a "blocks.{i}." key prefix.
TransformerBlock load_block(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

}  // namespace sames
}  // namespace sa3
