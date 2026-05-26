#include "dit.h"

#include <cmath>
#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace dit {

// ── DifferentialSelfAttention ────────────────────────────────────────
mx::array DifferentialSelfAttention::operator()(const mx::array& x) const {
    const auto& s = x.shape();
    const int B = s[0], T = s[1];
    constexpr int H = NUM_HEADS, D = HEAD_DIM;

    // 5×QKV split
    mx::array qkv = mx::matmul(x, mx::transpose(to_qkv));  // (B, T, 5*EMBED)
    auto parts = mx::split(qkv, 5, -1);
    auto to_heads = [&](const mx::array& t) {
        return mx::transpose(mx::reshape(t, {B, T, H, D}), {0, 2, 1, 3});
    };
    mx::array q      = to_heads(parts[0]);
    mx::array k      = to_heads(parts[1]);
    mx::array v      = to_heads(parts[2]);
    mx::array q_diff = to_heads(parts[3]);
    mx::array k_diff = to_heads(parts[4]);

    // qk RMSNorm (weight-only)
    q      = mx::fast::rms_norm(q,      q_norm_w, QK_NORM_EPS);
    k      = mx::fast::rms_norm(k,      k_norm_w, QK_NORM_EPS);
    q_diff = mx::fast::rms_norm(q_diff, q_norm_w, QK_NORM_EPS);
    k_diff = mx::fast::rms_norm(k_diff, k_norm_w, QK_NORM_EPS);

    // Partial RoPE on first 32 of 64 head dims
    auto rope = [&](const mx::array& t) {
        return mx::fast::rope(t, ROPE_DIMS, /*traditional=*/false,
                              /*base=*/10000.0f, /*scale=*/1.0f, /*offset=*/0);
    };
    q = rope(q); k = rope(k);
    q_diff = rope(q_diff); k_diff = rope(k_diff);

    // Two SDPAs, subtract — that's the "differential attention" trick
    mx::array out_main = mx::fast::scaled_dot_product_attention(q,      k,      v, ATTN_SCALE);
    mx::array out_diff = mx::fast::scaled_dot_product_attention(q_diff, k_diff, v, ATTN_SCALE);
    mx::array out = out_main - out_diff;

    out = mx::transpose(out, {0, 2, 1, 3});
    out = mx::reshape(out, {B, T, EMBED_DIM});
    return mx::matmul(out, mx::transpose(to_out));
}

// ── DifferentialCrossAttention ───────────────────────────────────────
mx::array DifferentialCrossAttention::operator()(
    const mx::array& x, const mx::array& context) const
{
    const int B  = x.shape()[0];
    const int Tx = x.shape()[1];
    const int Tc = context.shape()[1];
    constexpr int H = NUM_HEADS, D = HEAD_DIM;

    auto q_parts  = mx::split(mx::matmul(x, mx::transpose(to_q)),       2, -1);  // 2× (B, Tx, EMBED)
    auto kv_parts = mx::split(mx::matmul(context, mx::transpose(to_kv)), 3, -1); // 3× (B, Tc, EMBED)

    auto to_q_heads = [&](const mx::array& t) {
        return mx::transpose(mx::reshape(t, {B, Tx, H, D}), {0, 2, 1, 3});
    };
    auto to_c_heads = [&](const mx::array& t) {
        return mx::transpose(mx::reshape(t, {B, Tc, H, D}), {0, 2, 1, 3});
    };
    mx::array q      = to_q_heads(q_parts[0]);
    mx::array q_diff = to_q_heads(q_parts[1]);
    mx::array k      = to_c_heads(kv_parts[0]);
    mx::array k_diff = to_c_heads(kv_parts[1]);
    mx::array v      = to_c_heads(kv_parts[2]);

    q      = mx::fast::rms_norm(q,      q_norm_w, QK_NORM_EPS);
    k      = mx::fast::rms_norm(k,      k_norm_w, QK_NORM_EPS);
    q_diff = mx::fast::rms_norm(q_diff, q_norm_w, QK_NORM_EPS);
    k_diff = mx::fast::rms_norm(k_diff, k_norm_w, QK_NORM_EPS);

    mx::array out_main = mx::fast::scaled_dot_product_attention(q,      k,      v, ATTN_SCALE);
    mx::array out_diff = mx::fast::scaled_dot_product_attention(q_diff, k_diff, v, ATTN_SCALE);
    mx::array out = out_main - out_diff;

    out = mx::transpose(out, {0, 2, 1, 3});
    out = mx::reshape(out, {B, Tx, EMBED_DIM});
    return mx::matmul(out, mx::transpose(to_out));
}

// ── FeedForward (GLU + SiLU + Linear) ────────────────────────────────
mx::array FeedForward::operator()(const mx::array& x) const {
    mx::array h = mx::matmul(x, mx::transpose(proj_w)) + proj_b;     // (B, T, 2*FF_INNER)
    auto parts = mx::split(h, 2, -1);
    // _GLUWrap: x * silu(gate)  →  parts[0] * (parts[1] * sigmoid(parts[1]))
    mx::array gated = parts[0] * (parts[1] * mx::sigmoid(parts[1]));
    return mx::matmul(gated, mx::transpose(out_w)) + out_b;
}

// ── LocalEmbedSeq (Linear → SiLU → Linear) ───────────────────────────
mx::array LocalEmbedSeq::operator()(const mx::array& x) const {
    mx::array h = mx::matmul(x, mx::transpose(seq0_w)) + seq0_b;
    h = h * mx::sigmoid(h);                                          // silu
    return mx::matmul(h, mx::transpose(seq2_w)) + seq2_b;
}

// ── TransformerBlock ─────────────────────────────────────────────────
mx::array TransformerBlock::operator()(
    const mx::array& x_in,
    const mx::array& context,
    const mx::array& global_cond,        // (B, 6*EMBED)
    const mx::array& local_emb_padded) const
{
    const mx::Dtype dt = x_in.dtype();
    const mx::array one = mx::array(1.0f, dt);  // wrap scalars to avoid C++ fp32 promotion

    // ss = (to_scale_shift_gate + global_cond)[:, None, :]
    mx::array ss = to_scale_shift_gate + global_cond;                // (B, 6*EMBED)
    ss = mx::expand_dims(ss, 1);                                     // (B, 1, 6*EMBED)
    auto ss_parts = mx::split(ss, 6, -1);
    const mx::array& scale_self = ss_parts[0];
    const mx::array& shift_self = ss_parts[1];
    const mx::array& gate_self  = ss_parts[2];
    const mx::array& scale_ff   = ss_parts[3];
    const mx::array& shift_ff   = ss_parts[4];
    const mx::array& gate_ff    = ss_parts[5];

    // Self-attn block
    mx::array residual = x_in;
    mx::array h = mx::fast::rms_norm(x_in, pre_norm_w, NORM_EPS);
    h = h * (one + scale_self) + shift_self;
    h = self_attn(h);
    h = h * mx::sigmoid(one - gate_self);
    mx::array x = h + residual;

    // Cross-attn
    x = x + cross_attn(mx::fast::rms_norm(x, cross_attend_norm_w, NORM_EPS), context);

    // Local add-cond (residual)
    x = x + local_emb_padded;

    // FF block
    residual = x;
    h = mx::fast::rms_norm(x, ff_norm_w, NORM_EPS);
    h = h * (one + scale_ff) + shift_ff;
    h = ff(h);
    h = h * mx::sigmoid(one - gate_ff);
    return h + residual;
}

// ── ContinuousTransformer ────────────────────────────────────────────
mx::array ContinuousTransformer::operator()(
    const mx::array& x_in,
    const mx::array& context,
    const mx::array& global_embed,
    const mx::array& local_add_cond) const
{
    const int B = x_in.shape()[0];

    // project_in: (B, T, IO_CHANNELS) → (B, T, EMBED). No bias.
    mx::array x = mx::matmul(x_in, mx::transpose(project_in_w));

    // Prepend memory tokens
    mx::array mem = mx::broadcast_to(
        mx::expand_dims(memory_tokens, 0),  // (1, NUM_MEM, EMBED)
        {B, NUM_MEMORY_TOKENS, EMBED_DIM});
    x = mx::concatenate({mem, x}, 1);                                 // (B, NUM_MEM+T, EMBED)

    // global_cond_embedder: Linear → SiLU → Linear  →  (B, 6*EMBED)
    mx::array g = mx::matmul(global_embed, mx::transpose(gce0_w)) + gce0_b;
    g = g * mx::sigmoid(g);
    mx::array global_cond = mx::matmul(g, mx::transpose(gce2_w)) + gce2_b;

    // Per-layer local embedding (each layer has its own MLP). Zero-pad on the
    // left for the memory-token slots.
    mx::array zero_pad = mx::zeros({B, NUM_MEMORY_TOKENS, EMBED_DIM}, x.dtype());
    for (const auto& layer : layers) {
        mx::array local_emb = layer.to_local_embed(local_add_cond);   // (B, T, EMBED)
        mx::array local_emb_padded = mx::concatenate({zero_pad, local_emb}, 1);
        x = layer(x, context, global_cond, local_emb_padded);
    }

    // Drop memory tokens, project back to IO_CHANNELS
    const int T_with_mem = x.shape()[1];
    x = mx::slice(x,
                  mx::Shape{0, NUM_MEMORY_TOKENS, 0},
                  mx::Shape{B, T_with_mem, EMBED_DIM});
    return mx::matmul(x, mx::transpose(project_out_w));                // (B, T, IO_CHANNELS)
}

// ── ExpoFourierFeatures ──────────────────────────────────────────────
mx::array ExpoFourierFeatures::operator()(const mx::array& t) const {
    // t shape: (B,) → args (B, half), output (B, dim)
    mx::array args = mx::expand_dims(t, -1) * freqs;
    return mx::concatenate({mx::cos(args), mx::sin(args)}, -1);
}

static ExpoFourierFeatures make_timestep_features(mx::Dtype /*dt*/) {
    constexpr int half = TIMESTEP_FEAT_DIM / 2;
    constexpr float MIN_FREQ = 0.5f;
    constexpr float MAX_FREQ = 10000.0f;
    // Compute log min/max in fp64 then cast — matches Python's math.log behavior.
    const double log_min_d = std::log(static_cast<double>(MIN_FREQ));
    const double log_max_d = std::log(static_cast<double>(MAX_FREQ));
    const float scale = static_cast<float>(log_max_d - log_min_d);
    const float bias  = static_cast<float>(log_min_d);
    mx::array ramp = mx::linspace(0.0f, 1.0f, half, mx::float32);
    mx::array freqs = mx::exp(ramp * scale + bias);
    // Match Python: `freqs * 2 * math.pi` (left-associative).
    freqs = freqs * 2.0f * static_cast<float>(M_PI);
    mx::eval(freqs);
    // Keep freqs at fp32 regardless of DiT dtype: Python's ExpoFourierFeatures
    // creates this buffer with mx.linspace (default fp32) and never casts it.
    // Downstream `t (fp16) * freqs (fp32)` then promotes to fp32 — which is
    // what cascades global_embed and the per-block scale/shift/gate into fp32.
    // Casting freqs to fp16 here breaks bit-exactness against Python.
    return ExpoFourierFeatures{freqs};
}

// ── DiT::operator() ──────────────────────────────────────────────────
mx::array DiT::operator()(
    const mx::array& x,
    const mx::array& t,
    const mx::array& cross_attn_cond_raw,
    const mx::array& global_cond_raw,
    const std::optional<mx::array>& local_add_cond) const
{
    const int B = x.shape()[0];

    // Cross-attention conditioning projection (Linear → SiLU → Linear).
    mx::array c = mx::matmul(cross_attn_cond_raw, mx::transpose(tce0_w));
    c = c * mx::sigmoid(c);
    mx::array context = mx::matmul(c, mx::transpose(tce2_w));

    // Global conditioning projection.
    mx::array g = mx::matmul(global_cond_raw, mx::transpose(tge0_w));
    g = g * mx::sigmoid(g);
    mx::array global_pre = mx::matmul(g, mx::transpose(tge2_w));

    // Timestep features + MLP.
    mx::array tf = timestep_features(t);
    tf = mx::matmul(tf, mx::transpose(tte0_w)) + tte0_b;
    tf = tf * mx::sigmoid(tf);
    mx::array t_embed = mx::matmul(tf, mx::transpose(tte2_w)) + tte2_b;

    mx::array global_embed = global_pre + t_embed;

    // preprocess_conv + residual on channels-first input.
    mx::array x_lc = mx::transpose(x, {0, 2, 1});                     // (B, T_lat, IO_CHANNELS)
    mx::array x_pp = mx::conv1d(x_lc, preprocess_conv_w) + x_lc;

    // local_add_cond: explicit or zero default sized to the *input's* T_lat.
    // The stored `T_lat` field is a vestigial preallocation hint from the
    // Python port; x.shape()[2] is the ground truth for the actual seq length.
    const int actual_T_lat = x.shape()[2];
    mx::array local = local_add_cond.has_value()
        ? *local_add_cond
        : mx::zeros({B, actual_T_lat, LOCAL_ADD_COND_DIM}, x_pp.dtype());

    mx::array h = transformer(x_pp, context, global_embed, local);    // (B, T_lat, IO_CHANNELS)

    mx::array out = mx::conv1d(h, postprocess_conv_w) + h;
    return mx::transpose(out, {0, 2, 1});                              // back to (B, IO, T_lat)
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

DiT load_dit(
    const std::unordered_map<std::string, mx::array>& tensors,
    int T_lat,
    mx::Dtype dtype)
{
    auto get = [&](const std::string& k) { return mx::astype(at(tensors, k), dtype); };

    // ContinuousTransformer
    ContinuousTransformer ct{
        /*project_in_w=*/  get("transformer.project_in.weight"),
        /*project_out_w=*/ get("transformer.project_out.weight"),
        /*memory_tokens=*/ get("transformer.memory_tokens"),
        /*gce0_w=*/        get("transformer.global_cond_embedder.0.weight"),
        /*gce0_b=*/        get("transformer.global_cond_embedder.0.bias"),
        /*gce2_w=*/        get("transformer.global_cond_embedder.2.weight"),
        /*gce2_b=*/        get("transformer.global_cond_embedder.2.bias"),
        /*layers=*/        {},
    };
    ct.layers.reserve(DEPTH);
    for (int i = 0; i < DEPTH; ++i) {
        const std::string p = "transformer.layers." + std::to_string(i) + ".";
        ct.layers.push_back(TransformerBlock{
            /*pre_norm_w=*/ get(p + "pre_norm.weight"),
            /*self_attn=*/ DifferentialSelfAttention{
                get(p + "self_attn.to_qkv.weight"),
                get(p + "self_attn.to_out.weight"),
                get(p + "self_attn.q_norm.weight"),
                get(p + "self_attn.k_norm.weight"),
            },
            /*cross_attend_norm_w=*/ get(p + "cross_attend_norm.weight"),
            /*cross_attn=*/ DifferentialCrossAttention{
                get(p + "cross_attn.to_q.weight"),
                get(p + "cross_attn.to_kv.weight"),
                get(p + "cross_attn.to_out.weight"),
                get(p + "cross_attn.q_norm.weight"),
                get(p + "cross_attn.k_norm.weight"),
            },
            /*ff_norm_w=*/ get(p + "ff_norm.weight"),
            /*ff=*/ FeedForward{
                get(p + "ff.ff.0.proj.weight"),
                get(p + "ff.ff.0.proj.bias"),
                get(p + "ff.ff.2.weight"),
                get(p + "ff.ff.2.bias"),
            },
            /*to_scale_shift_gate=*/ get(p + "to_scale_shift_gate"),
            /*to_local_embed=*/ LocalEmbedSeq{
                get(p + "to_local_embed.seq.0.weight"),
                get(p + "to_local_embed.seq.0.bias"),
                get(p + "to_local_embed.seq.2.weight"),
                get(p + "to_local_embed.seq.2.bias"),
            },
        });
    }

    DiT dit{
        /*T_lat=*/ T_lat,
        /*preprocess_conv_w=*/  get("preprocess_conv.weight"),
        /*postprocess_conv_w=*/ get("postprocess_conv.weight"),
        /*tce0_w=*/ get("to_cond_embed.0.weight"),
        /*tce2_w=*/ get("to_cond_embed.2.weight"),
        /*tge0_w=*/ get("to_global_embed.0.weight"),
        /*tge2_w=*/ get("to_global_embed.2.weight"),
        /*tte0_w=*/ get("to_timestep_embed.0.weight"),
        /*tte0_b=*/ get("to_timestep_embed.0.bias"),
        /*tte2_w=*/ get("to_timestep_embed.2.weight"),
        /*tte2_b=*/ get("to_timestep_embed.2.bias"),
        /*timestep_features=*/ make_timestep_features(dtype),
        /*transformer=*/ std::move(ct),
    };
    return dit;
}

}  // namespace dit
}  // namespace sa3
