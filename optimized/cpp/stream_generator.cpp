// StreamGenerator — implementation. Port of realtime/runtime/generator.py.
#include "stream_generator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <mlx/mlx.h>

#include "dit_sm.h"
#include "sa3_pipeline.h"      // patched_decode, apply_prompt_padding
#include "samel_common.h"      // to_row_contiguous
#include "sames_decoder.h"     // sames::decode_chunked

namespace sa3 {
namespace rt {

static constexpr int CYCLIC_PAD = 8;   // latent frames of wrap-around context

// ── length plan: gen = loop + 2*margin, all even for SAME-S ──────────
static void plan_lengths(float seconds, float margin_s, int& gen_T,
                         int& margin_T, int& loop_T) {
    auto even_ceil = [](float s) {
        int f = std::max(2, (int)std::ceil(s * orch::SAMPLE_RATE / orch::SAMPLES_PER_LATENT));
        return f + (f % 2);
    };
    loop_T = even_ceil(seconds);
    int m = std::max(0, (int)std::lround(margin_s * orch::SAMPLE_RATE / orch::SAMPLES_PER_LATENT));
    margin_T = (m / 2) * 2;
    gen_T = loop_T + 2 * margin_T;
}

// patch_audio: rearrange("b c (l h) -> b (c h) l", h=patch). (1,2,T)->(1,512,T/256).
static mx::array patch_audio(const mx::array& audio, int patch = 256) {
    const int B = audio.shape()[0], C = audio.shape()[1], T = audio.shape()[2];
    const int L = T / patch;
    mx::array x = mx::reshape(audio, {B, C, L, patch});
    x = mx::transpose(x, {0, 1, 3, 2});               // (B, C, patch, L)
    return mx::reshape(x, {B, C * patch, L});
}

// cyclic_decode: pad latent with wrap-around context, decode, trim audio margins.
static mx::array cyclic_decode(const sames::SAMESDecoder& dec, const mx::array& lat,
                               int chunk, int ovl, int pad = CYCLIC_PAD) {
    const int C = lat.shape()[1], T = lat.shape()[2];
    mx::array head = mx::slice(lat, {0, 0, T - pad}, {1, C, T});
    mx::array tail = mx::slice(lat, {0, 0, 0}, {1, C, pad});
    mx::array padded = mx::concatenate({head, lat, tail}, 2);   // [1, C, T+2*pad]
    const int Tp = padded.shape()[2];
    const int kernel = chunk + 2 * ovl;
    mx::array patches = (Tp > kernel) ? sames::decode_chunked(dec, padded, chunk, ovl)
                                      : dec(padded);
    const int W = patches.shape()[2];
    return mx::slice(patches, {0, 0, pad * 16}, {1, patches.shape()[1], W - pad * 16});
}

StreamGenerator::StreamGenerator(AudioRing& ring, std::string models_dir,
                                 std::vector<std::vector<float>> source, GenConfig cfg)
    : ring_(ring), models_dir_(std::move(models_dir)), src_(std::move(source)),
      cfg_(cfg), cfg_styled_(cfg.cfg_styled), evolve_(cfg.evolve),
      seed_(cfg.seed_base) {}

StreamGenerator::~StreamGenerator() { stop(); }

void StreamGenerator::start() {
    thread_ = std::thread(&StreamGenerator::run, this);
}

void StreamGenerator::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
}

bool StreamGenerator::wait_ready(double timeout_s) {
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::duration<double>(timeout_s);
    while (!ready_.load()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

// ── control API ──────────────────────────────────────────────────────
void StreamGenerator::set_morph_amount(float v) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_); morph_amount_ = v;
}
void StreamGenerator::set_style_intensity(float v) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    if (v != style_intensity_) { style_intensity_ = v; need_retarget_ = true; }
}
void StreamGenerator::set_cfg(float v) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    if (v != cfg_styled_) { cfg_styled_ = v; need_retarget_ = true; }
}
void StreamGenerator::set_shared_curve(const std::string& name, float value) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    if (name == "velocity_scale") velocity_scale_ = value;
}
void StreamGenerator::clear_shared_curve(const std::string& name) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    if (name == "velocity_scale") velocity_scale_.reset();
}
void StreamGenerator::set_prompt(const std::string& text) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_); new_prompt_ = text;
}
void StreamGenerator::set_evolve(bool on) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_); evolve_ = on;
}
GenStats StreamGenerator::stats() const {
    std::lock_guard<std::mutex> lk(ctrl_mutex_); return stats_;
}

// ── setup (worker thread) ────────────────────────────────────────────
void StreamGenerator::setup() {
    // Playthrough (source longer than the window) needs INTERIOR margins so the
    // playhead never reads SA3's intro/outro fade at a chunk edge (else the chunk
    // crossfade fades into silence). Static loops cyclic-decode instead, so keep
    // their configured margin (0 by default).
    const int src_n0 = src_.empty() ? 0 : (int)src_[0].size();
    scanning_ = ((float)src_n0 / orch::SAMPLE_RATE > cfg_.seconds + 0.05f);
    if (scanning_ && cfg_.margin_s <= 0.0f) cfg_.margin_s = 0.75f;

    int margin_T, loop_T;
    plan_lengths(cfg_.seconds, cfg_.margin_s, gen_T_, margin_T, loop_T);
    crop_start_ = margin_T * orch::SAMPLES_PER_LATENT;
    loop_samples_ = loop_T * orch::SAMPLES_PER_LATENT;
    const int gen_samples = gen_T_ * orch::SAMPLES_PER_LATENT;
    const float gen_seconds = (float)gen_samples / orch::SAMPLE_RATE;

    // Load the model family: Live (sm-music + SAME-S) or Quality (medium + SAME-L).
    auto p = [&](const char* n) { return models_dir_ + "/" + n; };
    const bool quality = (cfg_.family == orch::Family::Medium);
    pipe_models_ = std::make_unique<orch::Pipeline>(quality
        ? orch::load_pipeline(
              p("t5gemma_f16.safetensors"), p("dit_medium_f16.safetensors"),
              p("same_l_encoder_f32.safetensors"), p("same_l_decoder_f32.safetensors"),
              orch::Family::Medium, cfg_.dtype)
        : orch::load_pipeline(
              p("t5gemma_f16.safetensors"), p("dit_sm-music_f16.safetensors"),
              p("same_s_encoder_f32.safetensors"), p("same_s_decoder_f32.safetensors"),
              orch::Family::SmMusic, cfg_.dtype));

    scan_start_ = 0;
    last_boundary_pos_ = 0;
    encode_window(0);   // sets init_latents_ (static loop); scan re-encodes per chunk

    // Conditioner: condition on the TRUE generated duration (so SA3's intro/outro
    // lands in the cropped margins).
    secs_e_ = mx::astype(pipe_models_->conditioner.seconds_embedder(gen_seconds), cfg_.dtype);
    encode_prompt(cfg_.prompt);

    // Codec decode params + DiT forward binding, per family. SAME-S: chunk=8/2;
    // SAME-L: chunk=128/8 (its larger receptive field). The DiT call signature is
    // identical across families, so a std::function captures either.
    if (quality) {
        chunk_ = 128; ovl_ = 8;
        auto& dit = *pipe_models_->dit_medium;
        dit_fwd_ = [&dit](const mx::array& x, const mx::array& t,
                          const mx::array& c, const mx::array& g) { return dit(x, t, c, g); };
        auto& d = *pipe_models_->samel_decoder;
        codec_decode_ = [&d](const mx::array& lat, int chunk, int ovl, bool /*cyclic*/) {
            // SAME-L has no cyclic-wrap helper; chunked decode covers all sizes.
            return samel::decode_chunked(d, lat, chunk, ovl);
        };
    } else {
        chunk_ = 8; ovl_ = 2;
        auto& dit = *pipe_models_->dit_small;
        dit_fwd_ = [&dit](const mx::array& x, const mx::array& t,
                          const mx::array& c, const mx::array& g) { return dit(x, t, c, g); };
        auto& d = *pipe_models_->sames_decoder;
        codec_decode_ = [&d](const mx::array& lat, int chunk, int ovl, bool cyclic) {
            const int T = lat.shape()[2], kernel = chunk + 2 * ovl;
            if (cyclic && T > 2 * CYCLIC_PAD) return cyclic_decode(d, lat, chunk, ovl);
            return (T > kernel) ? sames::decode_chunked(d, lat, chunk, ovl) : d(lat);
        };
    }
    pipe_ = std::make_unique<stream::StreamPipeline>(
        dit_fwd_, gen_T_, cfg_.steps, cfg_.depth, cfg_.dtype);

    // Warm up chunk 0 so the first audio isn't a cold start.
    if (scanning_) {
        // Playthrough: per-chunk banked-target morph. Encode chunk 0's source, generate
        // its styled target, pre-fill the buffer with the morph at the current Amount.
        apply_scan_controls();                       // re-encode prompt (if set) + velocity
        cur_src_ = encode_window_latent(0 - crop_start_);   // chunk 0: interior starts at source 0
        submit_target(*cur_src_);
        for (int i = 0; i < cfg_.steps + cfg_.depth + 2; ++i) {
            auto lat = pipe_->tick();
            if (lat.has_value()) { cur_latent_ = *lat; break; }
        }
        float A; { std::lock_guard<std::mutex> lk(ctrl_mutex_); A = morph_amount_; }
        if (cur_latent_.has_value()) {
            mx::array lat = (A <= 1e-4f) ? *cur_src_
                : *cur_src_ * mx::array(1.0f - A, cur_src_->dtype())
                  + *cur_latent_ * mx::array(A, cur_latent_->dtype());
            ring_.write_loop(decode_crop(lat));      // full-window initial fill
        }
        // chunk 1's target is built one-ahead by scan_tick's look-ahead (nxt_src_ empty).
    } else {
        compute_styled_target();
        submit_morph();
        for (int i = 0; i < cfg_.steps + cfg_.depth + 2; ++i) {
            auto lat = pipe_->tick();
            if (lat.has_value()) { last_latent_ = *lat; ring_.write_loop(decode_crop(*lat)); break; }
        }
    }
    ready_.store(true);
}

void StreamGenerator::encode_prompt(const std::string& text) {
    // Trim to decide whether a real style prompt is present (enables CFG).
    std::string trimmed = text;
    size_t b = trimmed.find_first_not_of(" \t\n\r");
    if (b == std::string::npos) trimmed.clear();
    else trimmed = trimmed.substr(b, trimmed.find_last_not_of(" \t\n\r") - b + 1);
    prompt_active_ = !trimmed.empty();

    auto encode_one = [&](const std::string& t) {
        auto enc = pipe_models_->t5.encode({t}, 256);
        mx::array embeds = mx::astype(enc.first, cfg_.dtype);
        mx::array ep = apply_prompt_padding(
            embeds, enc.second, mx::astype(pipe_models_->conditioner.padding_embedding, cfg_.dtype));
        return mx::concatenate({ep, *secs_e_}, 1);              // [1,257,768]
    };
    cross_ = encode_one(trimmed);
    gcond_ = mx::reshape(*secs_e_, {secs_e_->shape()[0], dit_sm::COND_TOKEN_DIM});  // [1,768]
    // Null (empty-prompt) conditioning is the CFG uncond branch. Encode once;
    // reused every tick while the prompt is active.
    if (prompt_active_) {
        neg_cross_ = encode_one("");
        mx::eval(*cross_, *gcond_, *neg_cross_);
    } else {
        neg_cross_.reset();
        mx::eval(*cross_, *gcond_);
    }
}

// LINEAR encode of the generated window source[gen_start, gen_start+gen_samples)
// (out-of-range -> silence). Used by the playthrough: gen_start = the chunk's
// interior start MINUS the leading margin, so SA3's intro/outro fade lands in the
// margins (which the playhead never decodes).
mx::array StreamGenerator::encode_window_latent(long gen_start) {
    const int ch = (int)src_.size();
    const int src_n = ch ? (int)src_[0].size() : 0;
    const int gen_samples = gen_T_ * orch::SAMPLES_PER_LATENT;
    const bool quality = (cfg_.family == orch::Family::Medium);
    std::vector<float> planar((size_t)ch * gen_samples, 0.0f);
    for (int c = 0; c < ch; ++c)
        for (int i = 0; i < gen_samples; ++i) {
            long s = gen_start + i;
            planar[(size_t)c * gen_samples + i] =
                (s >= 0 && s < src_n) ? src_[c][(size_t)s] : 0.0f;
        }
    mx::array extended(planar.data(), mx::Shape{1, ch, gen_samples}, mx::float32);
    mx::array patches = patch_audio(extended, 256);
    mx::array lat = mx::astype(quality
        ? (*pipe_models_->samel_encoder)(patches)
        : (*pipe_models_->sames_encoder)(patches), cfg_.dtype);
    mx::eval(lat);
    return lat;
}

// Static-loop encode: tile the source loop across the generated length (margins are
// a musically-continuous continuation of the loop). Sets init_latents_.
void StreamGenerator::encode_window(long start) {
    const int ch = (int)src_.size();
    const int src_n = ch ? (int)src_[0].size() : 0;
    const int gen_samples = gen_T_ * orch::SAMPLES_PER_LATENT;
    const bool quality = (cfg_.family == orch::Family::Medium);
    std::vector<float> planar((size_t)ch * gen_samples, 0.0f);
    for (int c = 0; c < ch; ++c)
        for (int i = 0; i < gen_samples; ++i) {
            long idx = ((long)i - crop_start_) % loop_samples_;
            if (idx < 0) idx += loop_samples_;
            long s = start + idx;
            planar[(size_t)c * gen_samples + i] =
                (s >= 0 && s < src_n) ? src_[c][(size_t)s] : 0.0f;
        }
    mx::array extended(planar.data(), mx::Shape{1, ch, gen_samples}, mx::float32);
    mx::array patches = patch_audio(extended, 256);
    init_latents_ = mx::astype(quality
        ? (*pipe_models_->samel_encoder)(patches)
        : (*pipe_models_->sames_encoder)(patches), cfg_.dtype);
    mx::eval(*init_latents_);
}

// Build the "wet" endpoint: a one-shot, full-strength styled generation of the
// source (explore + CFG -> strong, clean transfer), banked as a latent. The live
// loop then morphs the source toward THIS in latent space by Amount. Recomputed
// whenever the Style or its intensity (Character) changes. No prompt -> no target
// (Amount has nothing to morph toward -> the loop just reconstructs the source).
void StreamGenerator::compute_styled_target() {
    if (!prompt_active_ || !cross_.has_value() || !neg_cross_.has_value()) {
        styled_target_.reset();
        return;
    }
    float intensity, cfg_strength;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        intensity = style_intensity_;
        cfg_strength = cfg_styled_;
    }
    // Character -> how far the styled endpoint departs from the source (its
    // sigma_max). 0.45 (recognizably the loop, restyled) .. 0.95 (wild).
    const float style_denoise =
        0.45f + 0.5f * std::max(0.0f, std::min(1.0f, intensity));

    stream::StreamPipeline tgt(dit_fwd_, gen_T_, cfg_.steps, /*depth=*/1, cfg_.dtype);
    stream::SlotRequest req{*cross_, *gcond_, cfg_.seed_base, style_denoise, *init_latents_};
    req.sde_denoise_curve = stream::CurveValue{1.0f};   // explore (stochastic SDE path)
    req.cfg = cfg_strength;
    req.apg = cfg_.apg;
    req.rcfg_mode = "full";
    req.neg_cross_attn = *neg_cross_;
    tgt.submit(req);
    std::optional<mx::array> lat;
    for (int i = 0; i < cfg_.steps + 2 && !lat.has_value(); ++i) lat = tgt.tick();
    if (lat.has_value()) { mx::eval(*lat); styled_target_ = *lat; }
    else styled_target_.reset();
}

void StreamGenerator::submit_morph() {
    float amount, vs; bool has_vs; bool evolve;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        amount = morph_amount_;
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        evolve = evolve_;
    }
    if (evolve) seed_ += 1;
    const uint64_t seed = evolve ? seed_ : cfg_.seed_base;

    // Base = source-anchored reconstruction conditioned on the EMPTY prompt, so
    // Amount=0 is the original loop (the Style feeds styled_target_, not the base).
    // No sde curve -> the euler-x0 path, so the x0-target morph below is respected
    // (an sde re-noise step would re-inject the source and discard the blend).
    const mx::array& base_cross = (prompt_active_ && neg_cross_.has_value())
        ? *neg_cross_ : *cross_;
    stream::SlotRequest req{base_cross, *gcond_, seed, cfg_.denoise, *init_latents_};
    if (has_vs) req.velocity_scale = stream::CurveValue{vs};      // Motion
    // Amount = morph original->styled: blend the base x0 toward the banked styled
    // latent in the refinement half. Strength is ALSO a shared curve (see
    // apply_shared_curves) so a slider move lands on in-flight slots next tick.
    if (styled_target_.has_value()) {
        req.x0_target = *styled_target_;
        req.x0_target_strength = stream::CurveValue{amount};
    }
    pipe_->submit(req);
}

void StreamGenerator::apply_shared_curves() {
    float amount, vs; bool has_vs; bool retarget; std::optional<std::string> np;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        amount = morph_amount_;
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        np = new_prompt_; new_prompt_.reset();
        retarget = need_retarget_; need_retarget_ = false;
    }
    // Re-encode the prompt and/or rebuild the styled target BEFORE pushing curves
    // (both run on this worker thread; the ring keeps playing meanwhile).
    if (np.has_value()) { encode_prompt(*np); retarget = true; }
    if (retarget) compute_styled_target();

    // Amount morph strength: 1-tick onset on all in-flight slots (refinement-half
    // gated). Cleared when there's no target (no Style -> nothing to morph toward).
    if (styled_target_.has_value())
        pipe_->set_shared_curve("x0_target_strength", stream::CurveValue{amount});
    else
        pipe_->clear_shared_curve("x0_target_strength");
    if (has_vs) pipe_->set_shared_curve("velocity_scale", stream::CurveValue{vs});
    else        pipe_->clear_shared_curve("velocity_scale");
}

// ── decode + interior crop -> planar [ch][loop_samples] ──────────────
std::vector<std::vector<float>> StreamGenerator::decode_crop(const mx::array& lat) {
    mx::array latf = mx::astype(lat, mx::float32);
    // Cyclic wrap only when there's no interior-window crop (crop_start_==0).
    const bool use_cyclic = (crop_start_ == 0 && cfg_.cyclic);
    mx::array patches = codec_decode_(latf, chunk_, ovl_, use_cyclic);
    mx::array audio = sa3::patched_decode(patches, 256, 2);     // [1, 2, gen*4096]
    audio = samel::to_row_contiguous(mx::astype(audio, mx::float32));
    mx::eval(audio);
    const int ch = audio.shape()[1];
    const int total = audio.shape()[2];
    const float* d = audio.data<float>();   // [1, ch, total] row-major
    const int s = crop_start_;
    const int n = std::min(loop_samples_, total - s);
    std::vector<std::vector<float>> out(ch, std::vector<float>(std::max(0, n)));
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < n; ++i) out[c][i] = d[(size_t)c * total + (s + i)];
    }
    return out;
}

// ── the loop ─────────────────────────────────────────────────────────
void StreamGenerator::run() {
    setup();
    while (running_.load()) {
        if (scanning_) scan_tick();
        else           static_tick();
    }
}

// One iteration of the looping morph (short source == one window, repeated). Uses
// the banked-target morph (Amount = x0_target_strength).
void StreamGenerator::static_tick() {
    apply_shared_curves();
    while (pipe_->num_active() + pipe_->queue_size() < pipe_->depth()) submit_morph();

    auto t0 = std::chrono::steady_clock::now();
    auto lat = pipe_->tick();
    double tick_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    double dec_ms = 0.0;
    bool did_decode = false;
    if (lat.has_value()) {
        bool skip = false;
        if (last_latent_.has_value() && cfg_.mse_skip > 0.0f) {
            mx::array diff = *lat - *last_latent_;
            mx::array mse = mx::mean(diff * diff);
            mx::eval(mse);
            skip = mse.item<float>() < cfg_.mse_skip;
        }
        last_latent_ = *lat;
        if (!skip) {
            auto t1 = std::chrono::steady_clock::now();
            ring_.write_loop(decode_crop(*lat));
            dec_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t1).count();
            did_decode = true;
        }
    }

    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    stats_.ticks += 1;
    stats_.tick_ms = tick_ms;
    stats_.full_samples = 0;   // not scanning
    if (lat.has_value()) {
        stats_.completions += 1;
        if (did_decode) { stats_.decodes += 1; stats_.decode_ms = dec_ms; }
        else            { stats_.skips += 1; }
    }
}

// ── forward-streaming playthrough (DEMON-style) ──────────────────────
// Amount rides the shared sde_denoise_curve (0=source-anchored, 1=explore/styled)
// -> hits all in-flight slots, ~1-tick latency, no banked target on the hot path.
void StreamGenerator::apply_scan_controls() {
    bool has_vs; float vs; std::optional<std::string> np;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        np = new_prompt_; new_prompt_.reset();
    }
    // A Style change re-encodes the prompt; subsequently-built chunk targets pick it
    // up (held targets refresh as the playhead advances). Amount is NOT a curve here
    // — it's the decode-time latent lerp toward the styled target.
    if (np.has_value()) encode_prompt(*np);
    if (has_vs) pipe_->set_shared_curve("velocity_scale", stream::CurveValue{vs});
    else        pipe_->clear_shared_curve("velocity_scale");
}

void StreamGenerator::submit_target(const mx::array& source) {
    // Styled TARGET generation for a chunk: full explore (sde=1) + CFG -> strong;
    // generated ONCE per chunk and HELD as the Amount=1 endpoint of the latent morph.
    // Deterministic (fixed seed) + held -> the playthrough is STABLE (no per-
    // completion stochastic silence — that was the dropout bug).
    stream::SlotRequest req{*cross_, *gcond_, cfg_.seed_base, cfg_.scan_denoise, source};
    req.sde_denoise_curve = stream::CurveValue{1.0f};   // full explore
    if (prompt_active_ && neg_cross_.has_value()) {
        req.cfg = cfg_styled_;
        req.apg = cfg_.apg;
        req.rcfg_mode = "full";
        req.neg_cross_attn = *neg_cross_;
    }
    pipe_->submit(req);
}

// Decode a sub-window [off, off+win] (latent frames) with codec context, returning
// planar audio [ch][win*SAMPLES_PER_LATENT].
std::vector<std::vector<float>> StreamGenerator::decode_window(
        const mx::array& lat, int off_frames, int win_frames) {
    const int C = lat.shape()[1], T = lat.shape()[2];
    const int ctx = chunk_ + 2 * ovl_;
    const int lo = std::max(0, off_frames - ctx);
    const int hi = std::min(T, off_frames + win_frames + ctx);
    mx::array slice = mx::slice(mx::astype(lat, mx::float32), {0, 0, lo}, {1, C, hi});
    mx::array patches = codec_decode_(slice, chunk_, ovl_, /*cyclic=*/false);
    mx::array audio = sa3::patched_decode(patches, 256, 2);
    audio = samel::to_row_contiguous(mx::astype(audio, mx::float32));
    mx::eval(audio);
    const int ch = audio.shape()[1];
    const int total = audio.shape()[2];
    const float* d = audio.data<float>();
    const int s = (off_frames - lo) * orch::SAMPLES_PER_LATENT;
    const int n = std::min(win_frames * orch::SAMPLES_PER_LATENT, total - s);
    std::vector<std::vector<float>> out(ch, std::vector<float>(std::max(0, n)));
    for (int c = 0; c < ch; ++c)
        for (int i = 0; i < n; ++i) out[c][i] = d[(size_t)c * total + (s + i)];
    return out;
}

// Latent morph: lerp(src, tgt, amount) in latent space (Amount=0 -> source, 1 ->
// styled target), then windowed-decode at the playhead. The lerp is cheap, so
// Amount is ~1-tick (no regeneration).
std::vector<std::vector<float>> StreamGenerator::decode_morph(
        const mx::array& src, const mx::array& tgt, float amount, int off_frames, int win_frames) {
    const float a = std::max(0.0f, std::min(1.0f, amount));
    if (a <= 1e-4f)  return decode_window(src, off_frames, win_frames);
    if (a >= 0.9999f) return decode_window(tgt, off_frames, win_frames);
    mx::array lat = src * mx::array(1.0f - a, src.dtype()) + tgt * mx::array(a, tgt.dtype());
    return decode_window(lat, off_frames, win_frames);
}

void StreamGenerator::scan_tick() {
    apply_scan_controls();

    const int  SPL = orch::SAMPLES_PER_LATENT;
    const int  lead = SPL * 2;                                          // ~2 frames ahead
    const int  margin_f = crop_start_ / SPL;                           // intro/outro frames skipped
    const int  interior_f = std::max(2, loop_samples_ / SPL);         // playable interior frames
    // Crossfade overlap; <=0.4*interior so non-transition time remains. cur & next
    // decode the SAME source samples in their INTERIORS (frame-aligned) -> a clean
    // blend, never SA3's intro/outro fade.
    const int  win_f = std::max(2, std::min((int)(3.0 * orch::SAMPLE_RATE) / SPL,
                                            (int)(interior_f * 0.4)));
    const int  hop_f = std::max(1, interior_f - win_f);               // interior advance per chunk
    const long hop = (long)hop_f * SPL;
    const long xfade_dur = (long)(0.6 * orch::SAMPLE_RATE);
    const long ph = ring_.total_read();
    const long base = last_boundary_pos_;
    float A; { std::lock_guard<std::mutex> lk(ctrl_mutex_); A = morph_amount_; }

    // LOOK-AHEAD: build the next chunk's source + styled target one chunk ahead. Its
    // generation window starts a leading margin BEFORE its interior (so the intro
    // fade lands in the margin), hence encode_window_latent(interior_start - margin).
    if (!nxt_src_.has_value()) {
        long ns = scan_start_ + hop;
        if (ns + loop_samples_ > (long)src_[0].size()) ns = 0;
        next_scan_ = ns;
        next_base_ = base + hop;
        nxt_src_ = encode_window_latent(ns - crop_start_);
        pipe_->clear();
        submit_target(*nxt_src_);
        next_latent_.reset();
        xfade_start_ = -1;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto lat = pipe_->tick();
    double tick_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (lat.has_value() && !next_latent_.has_value()) next_latent_ = *lat;   // next target sharp

    // Crossfade to the next chunk once the playhead enters the overlap AND the next
    // target is ready (it is, with ~hop of lead time to generate it).
    const bool blending = ((ph + lead) - base >= hop) && next_latent_.has_value();
    if (blending && xfade_start_ < 0) xfade_start_ = ph;

    // Frame-aligned INTERIOR position; the latent offset adds the leading margin so
    // the playhead reads the chunk interior, never the intro/outro fade. cur & next
    // round to the SAME frame and write to the SAME position -> no cancellation.
    const long pos_f = std::max(0L, std::min((long)interior_f,
        (long)(((ph + lead) - base + SPL / 2) / SPL)));
    const int  off_cur_f = margin_f + (int)pos_f;
    const long write_pos = base + pos_f * SPL;

    double dec_ms = 0.0;
    if (cur_src_.has_value() && cur_latent_.has_value()) {
        auto t1 = std::chrono::steady_clock::now();
        // Output = latent morph lerp(source, styled target, Amount) at the playhead.
        std::vector<std::vector<float>> out = decode_morph(*cur_src_, *cur_latent_, A, off_cur_f, win_f);
        if (blending && xfade_start_ >= 0 && nxt_src_.has_value() && next_latent_.has_value()) {
            // EQUAL-POWER crossfade to the next chunk's morph (identical source pos,
            // also in its interior: off_next = margin + (interior pos - hop)).
            const int off_next_f = margin_f + (int)std::max(0L, pos_f - hop_f);
            auto nx = decode_morph(*nxt_src_, *next_latent_, A, off_next_f, win_f);
            const int ch = (int)out.size();
            for (int c = 0; c < ch; ++c) {
                const std::vector<float>& ns = nx[std::min(c, (int)nx.size() - 1)];
                const int m = std::min((int)out[c].size(), (int)ns.size());
                for (int k = 0; k < m; ++k) {
                    float r = (float)((write_pos + k) - xfade_start_) / (float)xfade_dur;
                    r = std::max(0.0f, std::min(1.0f, r));
                    out[c][k] = out[c][k] * std::cos(r * 1.57079633f) + ns[k] * std::sin(r * 1.57079633f);
                }
            }
        }
        ring_.write_forward(write_pos, out, /*xfade=*/256);
        dec_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t1).count();
    }

    // Promote next -> cur once the crossfade completes; the look-ahead block then
    // starts building the chunk AFTER next (nxt_src_ is reset).
    if (blending && xfade_start_ >= 0 && ph >= xfade_start_ + xfade_dur) {
        cur_src_ = nxt_src_;
        cur_latent_ = next_latent_;
        last_boundary_pos_ = next_base_;
        scan_start_ = next_scan_;
        nxt_src_.reset();
        next_latent_.reset();
        xfade_start_ = -1;
    }

    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    stats_.ticks += 1;
    stats_.tick_ms = tick_ms;
    if (dec_ms > 0.0) { stats_.decode_ms = dec_ms; stats_.decodes += 1; }
    stats_.scan_start = scan_start_;
    stats_.chunk_base = last_boundary_pos_;
    stats_.window_samples = loop_samples_;
    stats_.full_samples = (long)src_[0].size();
    if (lat.has_value()) stats_.completions += 1;
}

}  // namespace rt
}  // namespace sa3
