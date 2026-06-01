// StreamGenerator — implementation. Port of realtime/runtime/generator.py.
#include "stream_generator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include <mlx/mlx.h>

#include "dit_sm.h"
#include "sa3_pipeline.h"      // patched_decode, apply_prompt_padding
#include "samel_common.h"      // to_row_contiguous
#include "sames_decoder.h"     // sames::decode_chunked

namespace sa3 {
namespace rt {

static constexpr int CYCLIC_PAD = 8;   // latent frames of wrap-around context

// Amount -> img2img init sigma (defined after the class methods; forward-declared so
// setup()/submit_morph() can use it).
static float sigma_for_amount(float amount, float intensity);

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
    // Intensity caps Amount=1's sigma; applied to the NEXT submitted window.
    std::lock_guard<std::mutex> lk(ctrl_mutex_); style_intensity_ = v;
}
void StreamGenerator::set_cfg(float v) {
    // CFG applied to the NEXT submitted window.
    std::lock_guard<std::mutex> lk(ctrl_mutex_); cfg_styled_ = v;
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

    // Warm up window 0 so the first audio isn't a cold start.
    float A, intensity;
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); A = morph_amount_; intensity = style_intensity_; }
    if (scanning_) {
        // Playthrough: window 0 is a clean img2img of source[0:] at sigma(Amount).
        apply_scan_controls();                       // re-encode prompt (if set) + velocity
        mx::array src0 = encode_window_latent(0 - crop_start_);   // interior starts at source 0
        submit_window(src0, sigma_for_amount(A, intensity));
        for (int i = 0; i < cfg_.steps + cfg_.depth + 2; ++i) {
            auto lat = pipe_->tick();
            if (lat.has_value()) { cur_latent_ = *lat; break; }
        }
        if (cur_latent_.has_value()) ring_.write_loop(decode_crop(*cur_latent_));  // initial fill
        // window 1 is built one-ahead by scan_tick's look-ahead (next_pending_ false).
    } else {
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

void StreamGenerator::submit_morph() {
    float amount, intensity, vs; bool has_vs; bool evolve;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        amount = morph_amount_;
        intensity = style_intensity_;
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        evolve = evolve_;
    }
    if (evolve) seed_ += 1;
    const uint64_t seed = evolve ? seed_ : cfg_.seed_base;

    // Short-loop morph = clean img2img of the loop at sigma(Amount): Amount=0 ~ the
    // original loop, Amount=1 ~ strongly styled toward the prompt. Same coherent path
    // as the playthrough (no x0-target lerp / no sde source-blend — SA3 can't re-cohere
    // those). Steady controls -> identical latent -> MSE-skip keeps the loop coherent.
    stream::SlotRequest req{*cross_, *gcond_, seed, sigma_for_amount(amount, intensity), *init_latents_};
    if (has_vs) req.velocity_scale = stream::CurveValue{vs};      // Motion (per-step v-scale, coherent)
    if (prompt_active_ && neg_cross_.has_value()) {
        req.cfg = cfg_styled_;
        req.apg = cfg_.apg;
        req.rcfg_mode = "full";
        req.neg_cross_attn = *neg_cross_;
    }
    pipe_->submit(req);
}

void StreamGenerator::apply_shared_curves() {
    float vs; bool has_vs; std::optional<std::string> np;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        np = new_prompt_; new_prompt_.reset();
    }
    // Re-encode the prompt on this worker thread (the ring keeps playing). Amount is
    // baked into each window's init sigma at submit (per-request), not a shared curve.
    if (np.has_value()) encode_prompt(*np);
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

// ── forward-streaming playthrough: streaming img2img ─────────────────
// Each window is a clean img2img generation at sigma(Amount); Amount/Style/Intensity
// are baked per-window at submit (per-window latency). Motion (velocity) is a 1-tick
// per-step v-scale and stays a shared curve here.
// Returns true if a new prompt was applied this tick (so scan_tick can rebuild the
// already-committed next window instead of waiting a whole extra window for it).
bool StreamGenerator::apply_scan_controls() {
    bool has_vs; float vs; std::optional<std::string> np;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        np = new_prompt_; new_prompt_.reset();
    }
    // A Style change re-encodes the prompt; subsequently-built windows pick it up.
    if (np.has_value()) encode_prompt(*np);
    if (has_vs) pipe_->set_shared_curve("velocity_scale", stream::CurveValue{vs});
    else        pipe_->clear_shared_curve("velocity_scale");
    return np.has_value();
}

// Amount -> img2img init sigma (sigma_max), the ONLY structure-vs-style knob SA3
// supports coherently. The init is mixed ONCE (source*(1-sigma)+noise*sigma) and the
// trajectory is a clean text-conditioned generation; SA3's DiT has no source input to
// re-anchor to mid-trajectory (unlike ACE-Step/DEMON), so per-frame latent source-
// blends produce off-manifold garbage (the "frequency sweep") — img2img does not.
//   Amount=0 -> small sigma ~ source reconstruction (the "original");
//   Amount=1 -> large sigma ~ strongly regenerated toward the Style prompt.
// Intensity caps how far Amount=1 departs (sigma ceiling): subtle..wild.
static float sigma_for_amount(float amount, float intensity) {
    amount    = std::max(0.0f, std::min(1.0f, amount));
    intensity = std::max(0.0f, std::min(1.0f, intensity));
    const float sig_min = 0.08f;
    const float sig_max = 0.55f + 0.40f * intensity;   // 0.55 (subtle) .. 0.95 (wild)
    return sig_min + (sig_max - sig_min) * amount;
}

// Submit ONE clean windowed img2img generation: init-mix the source window to `sigma`,
// then a text-conditioned euler trajectory to completion (NO sde source-blend, NO
// x0-target morph). The FINISHED latent is decoded directly — never a mid-trajectory
// or lerped latent. This is SA3's native, coherent a2a path.
void StreamGenerator::submit_window(const mx::array& source, float sigma) {
    stream::SlotRequest req{*cross_, *gcond_, cfg_.seed_base, sigma, source};
    // Sampler: pingpong (sde curve = 1.0 -> re-noise toward the model's x0 with fresh
    // per-step noise, source term zeroes out) is SA3's trained rf_denoiser default and
    // what the offline remix path uses; euler is deterministic/smoother. Per-seed
    // deterministic either way. TEMP env SA3_PINGPONG to A/B; default pingpong.
    bool pingpong = true;
    if (const char* e = std::getenv("SA3_PINGPONG")) pingpong = (std::stoi(e) != 0);
    if (pingpong) req.sde_denoise_curve = stream::CurveValue{1.0f};
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

void StreamGenerator::scan_tick() {
    const bool prompt_changed = apply_scan_controls();

    // A Style change makes the already-committed next window stale (it was built one
    // window ahead with the OLD prompt). Unless we've already started crossfading into
    // it, drop it and let the look-ahead rebuild it with the new prompt THIS tick — so
    // a style change lands ~1 window out, not ~2. (The currently-playing window can't
    // change; that's the per-window-img2img floor — shrink Window to reduce it.)
    if (prompt_changed && next_pending_ && xfade_start_ < 0) {
        pipe_->clear();
        next_pending_ = false;
        next_latent_.reset();
    }

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
    float A, intensity;
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); A = morph_amount_; intensity = style_intensity_; }

    // LOOK-AHEAD: generate the NEXT window's img2img one chunk ahead. Its generation
    // window starts a leading margin BEFORE its interior (so SA3's intro fade lands in
    // the margin the playhead never decodes), hence encode_window_latent(start-margin).
    // Amount is baked into this window's init sigma at submit time (per-window latency).
    if (!next_pending_) {
        long ns = scan_start_ + hop;
        if (ns + loop_samples_ > (long)src_[0].size()) ns = 0;
        next_scan_ = ns;
        next_base_ = base + hop;
        mx::array nsrc = encode_window_latent(ns - crop_start_);
        pipe_->clear();
        submit_window(nsrc, sigma_for_amount(A, intensity));
        next_latent_.reset();
        next_pending_ = true;
        xfade_start_ = -1;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto lat = pipe_->tick();
    double tick_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (lat.has_value() && !next_latent_.has_value()) next_latent_ = *lat;   // next window done

    // Crossfade to the next window once the playhead enters the overlap AND the next
    // window has finished generating (it has, with ~hop of lead time).
    const bool blending = ((ph + lead) - base >= hop) && next_latent_.has_value();
    if (blending && xfade_start_ < 0) xfade_start_ = ph;

    // Frame-aligned INTERIOR position; the latent offset adds the leading margin so the
    // playhead reads the window interior, never the intro/outro fade. cur & next round
    // to the SAME frame and write to the SAME position -> no cancellation.
    const long pos_f = std::max(0L, std::min((long)interior_f,
        (long)(((ph + lead) - base + SPL / 2) / SPL)));
    const int  off_cur_f = margin_f + (int)pos_f;
    const long write_pos = base + pos_f * SPL;

    double dec_ms = 0.0;
    if (cur_latent_.has_value()) {
        auto t1 = std::chrono::steady_clock::now();
        // Output = the current window's FINISHED img2img latent, windowed at the playhead.
        std::vector<std::vector<float>> out = decode_window(*cur_latent_, off_cur_f, win_f);
        if (blending && xfade_start_ >= 0 && next_latent_.has_value()) {
            // EQUAL-POWER crossfade to the next window (identical source pos, also in
            // its interior: off_next = margin + (interior pos - hop)).
            const int off_next_f = margin_f + (int)std::max(0L, pos_f - hop_f);
            auto nx = decode_window(*next_latent_, off_next_f, win_f);
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
    // starts building the window AFTER next (next_pending_ cleared).
    if (blending && xfade_start_ >= 0 && ph >= xfade_start_ + xfade_dur) {
        cur_latent_ = next_latent_;
        last_boundary_pos_ = next_base_;
        scan_start_ = next_scan_;
        next_latent_.reset();
        next_pending_ = false;
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
