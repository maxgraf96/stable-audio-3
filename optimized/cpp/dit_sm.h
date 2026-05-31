// sa3-sm-music DiT (~440M params, 877 MB fp16) — STANDARD-attention diffusion
// transformer. Same block structure as the medium DiT (dit.h) but:
//   - Standard self-attention  (3×QKV split, single SDPA, no diff subtract)
//   - Standard cross-attention (1× to_q + 2× to_kv, single SDPA, no RoPE)
//   - Smaller dims: EMBED_DIM=1024, DEPTH=20, NUM_HEADS=16, FF_INNER=4096
//
// Forward (identical signature to dit::DiT):
//   x:                 (B, IO_CHANNELS, T_lat)  noisy latents
//   t:                 (B,)                     scalar timestep per batch
//   cross_attn_cond:   (B, 257, COND_TOKEN_DIM) text+seconds conditioning
//   global_cond:       (B, GLOBAL_COND_DIM)     pooled conditioning
//   local_add_cond:    (B, T_lat, 257) | none   inpaint mask + masked input
//   → v:               (B, IO_CHANNELS, T_lat)  predicted velocity
//
// Ported from optimized/mlx/models/defs/dit_mlx.py.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

namespace sa3 {
namespace dit_sm {

namespace mx = mlx::core;

// ── Constants (sa3-sm-music) ─────────────────────────────────────────
constexpr int   IO_CHANNELS         = 256;
constexpr int   EMBED_DIM           = 1024;
constexpr int   DEPTH               = 20;
constexpr int   NUM_HEADS           = 16;
constexpr int   HEAD_DIM            = 64;
constexpr int   ROPE_DIMS           = 32;
constexpr int   COND_TOKEN_DIM      = 768;
constexpr int   GLOBAL_COND_DIM     = 768;
constexpr int   LOCAL_ADD_COND_DIM  = 257;
constexpr int   NUM_MEMORY_TOKENS   = 64;
constexpr int   FF_INNER            = 4096;
constexpr int   TIMESTEP_FEAT_DIM   = 256;

constexpr float NORM_EPS    = 1e-5f;
constexpr float QK_NORM_EPS = 1e-6f;
constexpr float ATTN_SCALE  = 0.125f;  // 1/sqrt(HEAD_DIM=64), exact in fp32

// ── Sub-modules ──────────────────────────────────────────────────────
struct SelfAttention {
    mx::array to_qkv;    // (3*EMBED, EMBED)
    mx::array to_out;    // (EMBED, EMBED)
    mx::array q_norm_w;  // (HEAD_DIM,)
    mx::array k_norm_w;  // (HEAD_DIM,)

    mx::array operator()(const mx::array& x) const;
};

struct CrossAttention {
    mx::array to_q;      // (EMBED, EMBED)
    mx::array to_kv;     // (2*EMBED, EMBED)
    mx::array to_out;    // (EMBED, EMBED)
    mx::array q_norm_w;  // (HEAD_DIM,)
    mx::array k_norm_w;  // (HEAD_DIM,)

    mx::array operator()(const mx::array& x, const mx::array& context) const;
};

struct FeedForward {
    mx::array proj_w;    // (2*FF_INNER, EMBED) — ff.0.proj
    mx::array proj_b;    // (2*FF_INNER,)
    mx::array out_w;     // (EMBED, FF_INNER) — ff.2
    mx::array out_b;     // (EMBED,)

    mx::array operator()(const mx::array& x) const;
};

struct LocalEmbedSeq {
    mx::array seq0_w;    // (EMBED, LOCAL_ADD_COND_DIM)
    mx::array seq0_b;    // (EMBED,)
    mx::array seq2_w;    // (EMBED, EMBED)
    mx::array seq2_b;    // (EMBED,)

    mx::array operator()(const mx::array& x) const;
};

struct TransformerBlock {
    mx::array pre_norm_w;
    SelfAttention self_attn;
    mx::array cross_attend_norm_w;
    CrossAttention cross_attn;
    mx::array ff_norm_w;
    FeedForward ff;
    mx::array to_scale_shift_gate;   // (6*EMBED,)
    LocalEmbedSeq to_local_embed;

    mx::array operator()(
        const mx::array& x,
        const mx::array& context,
        const mx::array& global_cond,        // (B, 6*EMBED)
        const mx::array& local_emb_padded) const;
};

struct ContinuousTransformer {
    mx::array project_in_w;          // (EMBED, IO_CHANNELS)
    mx::array project_out_w;         // (IO_CHANNELS, EMBED)
    mx::array memory_tokens;         // (NUM_MEMORY_TOKENS, EMBED)
    mx::array gce0_w;                // (EMBED, EMBED)        — global_cond_embedder.0
    mx::array gce0_b;                // (EMBED,)
    mx::array gce2_w;                // (6*EMBED, EMBED)      — global_cond_embedder.2
    mx::array gce2_b;                // (6*EMBED,)
    std::vector<TransformerBlock> layers;  // DEPTH = 20

    mx::array operator()(
        const mx::array& x,
        const mx::array& context,
        const mx::array& global_embed,
        const mx::array& local_add_cond) const;
};

// Timestep features: t (B,) → (B, TIMESTEP_FEAT_DIM) via cos/sin of t*freqs.
struct ExpoFourierFeatures {
    mx::array freqs;  // (TIMESTEP_FEAT_DIM/2,) — precomputed at construction
    mx::array operator()(const mx::array& t) const;
};

struct DiT {
    int T_lat;
    mx::array preprocess_conv_w;     // (IO_CHANNELS, 1, IO_CHANNELS)
    mx::array postprocess_conv_w;    // (IO_CHANNELS, 1, IO_CHANNELS)
    mx::array tce0_w;                // (EMBED, COND_TOKEN_DIM)   — to_cond_embed.0
    mx::array tce2_w;                // (EMBED, EMBED)            — to_cond_embed.2
    mx::array tge0_w;                // (EMBED, GLOBAL_COND_DIM)  — to_global_embed.0
    mx::array tge2_w;                // (EMBED, EMBED)            — to_global_embed.2
    mx::array tte0_w;                // (EMBED, TIMESTEP_FEAT_DIM)— to_timestep_embed.0
    mx::array tte0_b;                // (EMBED,)
    mx::array tte2_w;                // (EMBED, EMBED)            — to_timestep_embed.2
    mx::array tte2_b;                // (EMBED,)
    ExpoFourierFeatures timestep_features;
    ContinuousTransformer transformer;

    mx::array operator()(
        const mx::array& x,
        const mx::array& t,
        const mx::array& cross_attn_cond_raw,
        const mx::array& global_cond_raw,
        const std::optional<mx::array>& local_add_cond = std::nullopt) const;
};

// ── Loader ───────────────────────────────────────────────────────────
// Loads everything from a safetensors tensor map. Ignores the "cond.*" keys
// (those belong to the conditioner, loaded via sa3::load_conditioner).
DiT load_dit(
    const std::unordered_map<std::string, mx::array>& tensors,
    int T_lat = 320,
    mx::Dtype dtype = mx::float16);

}  // namespace dit_sm
}  // namespace sa3
