// Shared SAME-L components (encoder + decoder use the same blocks).
//
// Ported from optimized/mlx/models/defs/same_l_decoder.py:
//   - DyT          (Dynamic Tanh normalization: gamma * tanh(alpha*x) + beta)
//   - FeedForward  (GLU FF, optional sin(πx) gate for decoder blocks 5..11)
//   - DifferentialSWA  (differential attention with sliding window)
//   - swa_mask()   (cached 17x51 mask)
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

namespace sa3 {
namespace samel {

namespace mx = mlx::core;

// ── Constants (SAME-L from sa3-medium) ───────────────────────────────
constexpr int LATENT_DIM      = 256;
constexpr int DIM             = 1536;
constexpr int NUM_HEADS       = 24;
constexpr int HEAD_DIM        = 64;
constexpr int ROPE_DIMS       = 32;
constexpr int NUM_BLOCKS      = 12;
constexpr int FF_INNER        = 4608;       // ff_mult = 3
constexpr int SIN_START_BLOCK = 5;          // decoder uses sin on blocks 5..11
constexpr int OUT_CHANNELS    = 512;
constexpr int STRIDE          = 16;
constexpr int SUB_CHUNK_SIZE  = STRIDE + 1; // 17 (1 latent + 16 new-token positions)
constexpr int BLOCK_SIZE      = SUB_CHUNK_SIZE;
constexpr int SIN_PER_POS     = SUB_CHUNK_SIZE - 1; // 16
constexpr int IN_CHANNELS     = 512;        // encoder input channels

// HEAD_DIM ** -0.5 = 64 ** -0.5 = 0.125, exact in fp32
constexpr float ATTN_SCALE = 0.125f;

// ── DyT: gamma * tanh(alpha * x) + beta ──────────────────────────────
struct DyT {
    mx::array alpha;   // (1,)
    mx::array gamma;   // (dim,)
    mx::array beta;    // (dim,)

    mx::array operator()(const mx::array& x) const;
};

// ── FeedForward: GLU, optional sin(πx) gate ──────────────────────────
struct FeedForward {
    mx::array glu_proj_w;   // (FF_INNER * 2, DIM)
    mx::array glu_proj_b;   // (FF_INNER * 2,)
    mx::array proj_out_w;   // (DIM, FF_INNER)
    mx::array proj_out_b;   // (DIM,)
    bool use_sin = false;

    mx::array operator()(const mx::array& x) const;
};

// ── DifferentialSWA: differential attention with sliding window ──────
struct DifferentialSWA {
    mx::array to_qkv;   // (5*DIM, DIM)
    mx::array to_out;   // (DIM, DIM)
    DyT q_norm;         // alpha (1,), gamma (HEAD_DIM,), beta (HEAD_DIM,)
    DyT k_norm;

    mx::array operator()(
        const mx::array& x,
        const std::optional<mx::array>& mask,
        bool full_attention) const;
};

// Cached static SWA mask of shape (17, 51), float32. Each query token in a
// 17-block attends to 51 KV tokens (3 consecutive groups).
const mx::array& swa_mask();

// Force an array to be row-contiguous in memory so `.data<float>()` reads
// values in logical row-major order. Required after view-only ops like
// `transpose`/`slice`/`as_strided` if you intend to dereference the buffer
// directly — MLX's `.data<T>()` is a raw pointer to the backing storage and
// doesn't respect strides.
mx::array to_row_contiguous(const mx::array& a);

// Shared block used by both encoder and decoder. The only thing that varies
// between the two is `ff.use_sin` (false for all encoder blocks; true for
// decoder blocks with index >= SIN_START_BLOCK).
struct TransformerBlock {
    DyT pre_norm;
    DifferentialSWA attn;
    DyT ff_norm;
    FeedForward ff;

    mx::array operator()(
        const mx::array& x,
        const std::optional<mx::array>& mask,
        bool full_attention) const;
};

TransformerBlock load_transformer_block(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,   // e.g. "blocks.3."
    mx::Dtype dtype,
    bool use_sin);

// ── Loaders from safetensors tensor map ─────────────────────────────
DyT load_dyt(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

FeedForward load_feedforward(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype,
    bool use_sin);

DifferentialSWA load_differential_swa(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& prefix,
    mx::Dtype dtype);

}  // namespace samel
}  // namespace sa3
