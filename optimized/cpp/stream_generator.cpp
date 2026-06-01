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
      cfg_(cfg), denoise_target_(cfg.denoise), evolve_(cfg.evolve),
      seed_(cfg.seed_base), denoise_eff_(cfg.denoise) {}

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
void StreamGenerator::set_denoise(float v) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_); denoise_target_ = v;
}
void StreamGenerator::set_shared_curve(const std::string& name, float value) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    if (name == "sde_denoise_curve") sde_curve_ = value;
    else if (name == "velocity_scale") velocity_scale_ = value;
}
void StreamGenerator::clear_shared_curve(const std::string& name) {
    std::lock_guard<std::mutex> lk(ctrl_mutex_);
    if (name == "sde_denoise_curve") sde_curve_.reset();
    else if (name == "velocity_scale") velocity_scale_.reset();
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
    int margin_T, loop_T;
    plan_lengths(cfg_.seconds, cfg_.margin_s, gen_T_, margin_T, loop_T);
    crop_start_ = margin_T * orch::SAMPLES_PER_LATENT;
    loop_samples_ = loop_T * orch::SAMPLES_PER_LATENT;
    const int gen_samples = gen_T_ * orch::SAMPLES_PER_LATENT;
    const float gen_seconds = (float)gen_samples / orch::SAMPLE_RATE;

    // Load models (SmMusic family) via the orchestrator loader.
    auto p = [&](const char* n) { return models_dir_ + "/" + n; };
    pipe_models_ = std::make_unique<orch::Pipeline>(orch::load_pipeline(
        p("t5gemma_f16.safetensors"), p("dit_sm-music_f16.safetensors"),
        p("same_s_encoder_f32.safetensors"), p("same_s_decoder_f32.safetensors"),
        orch::Family::SmMusic, cfg_.dtype));

    // Build the extended source: tile one loop period so the interior crop aligns
    // and the margins are a musically-continuous continuation (source is a loop).
    const int ch = (int)src_.size();
    const int src_n = ch ? (int)src_[0].size() : 0;
    std::vector<float> planar((size_t)ch * gen_samples, 0.0f);
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < gen_samples; ++i) {
            long idx = ((long)i - crop_start_) % loop_samples_;
            if (idx < 0) idx += loop_samples_;
            planar[(size_t)c * gen_samples + i] =
                (idx < src_n) ? src_[c][idx] : 0.0f;   // loop_unit zero-padded if short
        }
    }
    mx::array extended(planar.data(), mx::Shape{1, ch, gen_samples}, mx::float32);
    mx::array patches = patch_audio(extended, 256);
    init_latents_ = mx::astype((*pipe_models_->sames_encoder)(patches), cfg_.dtype);
    mx::eval(*init_latents_);

    // Conditioner: condition on the TRUE generated duration (so SA3's intro/outro
    // lands in the cropped margins).
    secs_e_ = mx::astype(pipe_models_->conditioner.seconds_embedder(gen_seconds), cfg_.dtype);
    encode_prompt(cfg_.prompt);

    chunk_ = 8; ovl_ = 2;

    // Build the streaming pipeline with a dit_sm forward binding.
    auto& dit = *pipe_models_->dit_small;
    stream::DiTForward dit_fwd = [&dit](const mx::array& x, const mx::array& t,
                                        const mx::array& c, const mx::array& g) {
        return dit(x, t, c, g);
    };
    pipe_ = std::make_unique<stream::StreamPipeline>(
        dit_fwd, gen_T_, cfg_.steps, cfg_.depth, cfg_.dtype);

    // Warmup: one full generation so the first audible loop isn't a cold start.
    submit_morph();
    for (int i = 0; i < cfg_.steps + cfg_.depth + 2; ++i) {
        auto lat = pipe_->tick();
        if (lat.has_value()) { ring_.write_loop(decode_crop(*lat)); break; }
    }
    ready_.store(true);
}

void StreamGenerator::encode_prompt(const std::string& text) {
    auto enc = pipe_models_->t5.encode({text}, 256);
    mx::array embeds = mx::astype(enc.first, cfg_.dtype);
    mx::array ep = apply_prompt_padding(
        embeds, enc.second, mx::astype(pipe_models_->conditioner.padding_embedding, cfg_.dtype));
    cross_ = mx::concatenate({ep, *secs_e_}, 1);                 // [1,257,768]
    gcond_ = mx::reshape(*secs_e_, {secs_e_->shape()[0], dit_sm::COND_TOKEN_DIM});  // [1,768]
    mx::eval(*cross_, *gcond_);
}

void StreamGenerator::submit_morph() {
    float denoise_target, sdc; bool has_sdc; float vs; bool has_vs; bool evolve;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        denoise_target = denoise_target_;
        has_sdc = sde_curve_.has_value();   sdc = has_sdc ? *sde_curve_ : 0.0f;
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        evolve = evolve_;
    }
    // Slew the effective denoise toward the target (gradual morph on slider jumps).
    const float slew = cfg_.denoise_slew;
    const float delta = denoise_target - denoise_eff_;
    denoise_eff_ += std::max(-slew, std::min(slew, delta));
    const float denoise = std::round(denoise_eff_ * 1e4f) / 1e4f;

    if (evolve) seed_ += 1;
    const uint64_t seed = evolve ? seed_ : cfg_.seed_base;

    stream::SlotRequest req{*cross_, *gcond_, seed, denoise, *init_latents_};
    if (has_sdc) req.sde_denoise_curve = stream::CurveValue{sdc};
    if (has_vs)  req.velocity_scale    = stream::CurveValue{vs};
    pipe_->submit(req);
}

void StreamGenerator::apply_shared_curves() {
    bool has_sdc, has_vs; float sdc, vs; std::optional<std::string> np;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        has_sdc = sde_curve_.has_value();   sdc = has_sdc ? *sde_curve_ : 0.0f;
        has_vs = velocity_scale_.has_value(); vs = has_vs ? *velocity_scale_ : 1.0f;
        np = new_prompt_; new_prompt_.reset();
    }
    // Shared-mutable overrides hit all in-flight slots next tick (1-tick onset).
    if (has_sdc) pipe_->set_shared_curve("sde_denoise_curve", stream::CurveValue{sdc});
    else         pipe_->clear_shared_curve("sde_denoise_curve");
    if (has_vs)  pipe_->set_shared_curve("velocity_scale", stream::CurveValue{vs});
    else         pipe_->clear_shared_curve("velocity_scale");
    if (np.has_value()) encode_prompt(*np);
}

// ── decode + interior crop -> planar [ch][loop_samples] ──────────────
std::vector<std::vector<float>> StreamGenerator::decode_crop(const mx::array& lat) {
    mx::array latf = mx::astype(lat, mx::float32);
    const int kernel = chunk_ + 2 * ovl_;
    mx::array patches = (crop_start_ == 0 && cfg_.cyclic && gen_T_ > 2 * CYCLIC_PAD)
        ? cyclic_decode(*pipe_models_->sames_decoder, latf, chunk_, ovl_)
        : ((gen_T_ > kernel)
            ? sames::decode_chunked(*pipe_models_->sames_decoder, latf, chunk_, ovl_)
            : (*pipe_models_->sames_decoder)(latf));
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
        apply_shared_curves();
        // keep the ring full: count both in-flight slots AND queued requests, or
        // we'd submit forever (submit only queues; slots fill in tick()).
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
        if (lat.has_value()) {
            stats_.completions += 1;
            if (did_decode) { stats_.decodes += 1; stats_.decode_ms = dec_ms; }
            else            { stats_.skips += 1; }
        }
    }
}

}  // namespace rt
}  // namespace sa3
