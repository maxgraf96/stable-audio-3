// MorphEngine — implementation.
#include "MorphEngine.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "sa3_orchestrator.h"   // SAMPLE_RATE, SAMPLES_PER_LATENT

namespace sa3morph {

namespace {

// Resolve the directory holding the sm-music + SAME-S safetensors. Mirrors
// VariationsEngine::resolveModelsDir but for "SA3 Realtime".
juce::File resolveModelsDirImpl() {
    if (const char* env = std::getenv("SA3_MODELS_DIR"); env && *env) {
        juce::File f(juce::String::fromUTF8(env));
        if (f.isDirectory()) return f;
    }
    auto exe = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    auto bundled = exe.getParentDirectory().getParentDirectory()
                      .getChildFile("Resources").getChildFile("models");
    if (bundled.isDirectory()) return bundled;
    // Prefer the Realtime-specific dir, else share the Variations plugin's
    // models (same safetensors family — both download the same SA3 weights).
    auto appsup = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Application Support");
    auto rtDir = appsup.getChildFile("SA3 Realtime").getChildFile("models");
    if (rtDir.isDirectory()) return rtDir;
    return appsup.getChildFile("SA3 Variations").getChildFile("models");
}

constexpr int kSampleRate = sa3::orch::SAMPLE_RATE;   // 44100
constexpr double kMaxScanSeconds = 300.0;   // cap the scanned track length (memory/encode)

// Loop length plan in 44.1k samples for a given source duration (matches
// StreamGenerator's even_ceil(loop_T) * SAMPLES_PER_LATENT with margin 0).
int loopLenSamples(float seconds) {
    int f = std::max(2, (int)std::ceil(seconds * sa3::orch::SAMPLE_RATE
                                       / sa3::orch::SAMPLES_PER_LATENT));
    f += (f % 2);
    return f * sa3::orch::SAMPLES_PER_LATENT;
}

}  // namespace

// kSampleRate is referenced inside the anonymous namespace above; the member
// functions below use sa3::orch::SAMPLE_RATE directly.

// ── Audio decode helpers (reused from the Variations engine) ─────────
juce::AudioBuffer<float> MorphEngine::decodeToStereo44k(const juce::MemoryBlock& bytes) {
    if (bytes.getSize() == 0) throw std::runtime_error("source audio is empty");
    juce::AudioFormatManager fmt;
    fmt.registerBasicFormats();
    auto stream = std::make_unique<juce::MemoryInputStream>(bytes, false);
    std::unique_ptr<juce::AudioFormatReader> reader(fmt.createReaderFor(std::move(stream)));
    if (!reader) throw std::runtime_error("unsupported audio format (WAV/AIFF/FLAC/Ogg)");
    const int    src_channels = (int)reader->numChannels;
    const int    src_samples  = (int)reader->lengthInSamples;
    const double src_sr       = reader->sampleRate;
    if (src_samples <= 0 || src_channels < 1) throw std::runtime_error("source audio has no samples");

    juce::AudioBuffer<float> src(2, src_samples);
    src.clear();
    reader->read(&src, 0, src_samples, 0, true, true);
    if (src_channels == 1) src.copyFrom(1, 0, src, 0, 0, src_samples);

    if (std::abs(src_sr - kSampleRate) <= 0.5) return src;
    const int dst_samples = (int)std::ceil(src_samples * (kSampleRate / src_sr));
    juce::AudioBuffer<float> out(2, dst_samples);
    out.clear();
    const double ratio = src_sr / (double)kSampleRate;
    for (int c = 0; c < 2; ++c) {
        juce::LagrangeInterpolator interp;
        interp.reset();
        interp.process(ratio, src.getReadPointer(c), out.getWritePointer(c), dst_samples);
    }
    return out;
}

std::vector<float> MorphEngine::extractPeaks(const juce::AudioBuffer<float>& buf, int n, int len) {
    std::vector<float> peaks((size_t)std::max(0, n), 0.0f);
    if (len < 0) len = buf.getNumSamples();
    len = std::min(len, buf.getNumSamples());
    if (len <= 0 || n <= 0) return peaks;
    const int nch = std::max(1, buf.getNumChannels());
    const float* L = buf.getReadPointer(0);
    const float* R = (nch > 1) ? buf.getReadPointer(1) : L;
    const double win = (double)len / (double)n;
    for (int i = 0; i < n; ++i) {
        const int start = (int)std::floor(i * win);
        const int end   = std::min(len, (int)std::floor((i + 1) * win));
        float maxV = 0.0f;
        for (int j = start; j < end; ++j)
            maxV = std::max(maxV, std::max(std::abs(L[j]), std::abs(R[j])));
        peaks[(size_t)i] = std::min(1.0f, std::max(0.04f, maxV));
    }
    return peaks;
}

// ── Lifecycle ────────────────────────────────────────────────────────
MorphEngine::MorphEngine() = default;

MorphEngine::~MorphEngine() {
    playing_.store(false);
    if (generator_) generator_->stop();
    if (loader_.joinable()) loader_.join();
}

std::string MorphEngine::resolveModelsDir() const {
    return resolveModelsDirImpl().getFullPathName().toStdString();
}

void MorphEngine::writeStatus(juce::String s) {
    std::lock_guard<std::mutex> lk(status_mutex_);
    status_ = std::move(s);
}

juce::String MorphEngine::getStatus() const {
    std::lock_guard<std::mutex> lk(status_mutex_);
    return status_;
}

std::vector<float> MorphEngine::getSourcePeaks() const {
    std::lock_guard<std::mutex> lk(peaks_mutex_);
    return source_peaks_;
}

// ── Source load (async on a one-shot loader thread) ──────────────────
bool MorphEngine::requestLoadSource(juce::MemoryBlock audioBytes, int peaks_n,
                                    CompletionFn completion) {
    bool expected = false;
    if (!load_in_flight_.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel)) {
        return false;   // a load is already running
    }
    if (loader_.joinable()) loader_.join();   // join the previous one-shot
    phase_.store(Phase::Loading);
    writeStatus("Loading...");
    loader_ = std::thread(&MorphEngine::loaderThreadMain, this,
                          std::move(audioBytes), peaks_n, std::move(completion));
    return true;
}

void MorphEngine::loaderThreadMain(juce::MemoryBlock bytes, int peaks_n,
                                   CompletionFn completion) {
    auto fail = [&](const juce::String& msg) {
        phase_.store(Phase::Error);
        writeStatus(msg);
        juce::DynamicObject::Ptr err = new juce::DynamicObject();
        err->setProperty("ok", false);
        err->setProperty("error", msg);
        load_in_flight_.store(false, std::memory_order_release);
        if (completion) completion(juce::var(err.get()));
    };

    try {
        writeStatus("Decoding source...");
        juce::AudioBuffer<float> full = decodeToStereo44k(bytes);
        int full_n = full.getNumSamples();

        // Cap the scanned track length (bounds memory + the per-window encode). A
        // longer file is truncated to the first kMaxScanSeconds; the playthrough
        // starts at the top and slides the window forward through it.
        const int max_n = (int)std::lround(kMaxScanSeconds * (double)kSampleRate);
        if (full_n > max_n) full_n = max_n;

        // The WINDOW is the morph basis (the engine slides it across the track as it
        // plays). cfg.seconds = the window length; the source handed to the engine is
        // the WHOLE (capped) track, which it scans. A file shorter than the window is
        // just one static loop (no scanning).
        const double full_secs = (double)full_n / kSampleRate;
        const double win_s = window_seconds_.load(std::memory_order_acquire);
        const double window_secs = (win_s > 0.0) ? std::min(full_secs, win_s) : full_secs;

        const int   src_n   = full_n;
        const double src_secs = full_secs;   // reported duration = the whole track

        // peaks over the WHOLE track (the waveform the scan playhead travels across)
        std::vector<float> peaks = extractPeaks(full, peaks_n, full_n);
        {
            std::lock_guard<std::mutex> lk(peaks_mutex_);
            source_peaks_ = peaks;
        }

        // Planar full track for the generator: [ch][samples].
        std::vector<std::vector<float>> planar(2, std::vector<float>(src_n));
        for (int c = 0; c < 2; ++c)
            std::copy(full.getReadPointer(c), full.getReadPointer(c) + src_n,
                      planar[(size_t)c].begin());

        // Build the ring sized to ONE WINDOW (the playable loop), with the loop-fold.
        // Tear down any prior generator first (joins its MLX thread).
        if (generator_) { generator_->stop(); generator_.reset(); }

        sa3::rt::GenConfig cfg;
        cfg.seconds = (float)window_secs;
        cfg.depth   = 2;
        cfg.steps   = 8;
        cfg.family  = quality_.load(std::memory_order_acquire)
            ? sa3::orch::Family::Medium : sa3::orch::Family::SmMusic;
        // cfg.denoise is the FIXED base trajectory (GenConfig default); the UI
        // "Amount" is the morph strength, not the base denoise, so don't override.
        const int L = loopLenSamples(cfg.seconds);
        const int loop_xfade = (int)(cfg.loop_xfade_s * kSampleRate);
        ring_ = std::make_unique<sa3::rt::AudioRing>(L, 2, 2048, loop_xfade);
        loop_len_.store(ring_->length());
        loop_secs_.store((double)ring_->length() / kSampleRate);

        writeStatus("Loading models...");
        generator_ = std::make_unique<sa3::rt::StreamGenerator>(
            *ring_, resolveModelsDir(), planar, cfg);
        generator_->start();
        if (!generator_->wait_ready(180.0)) {
            fail("model load / warmup timed out");
            return;
        }

        // Apply any controls set before load completed.
        {
            std::lock_guard<std::mutex> lk(ctrl_mutex_);
            generator_->set_morph_amount(denoise_);
            generator_->set_style_intensity(source_blend_);
            generator_->set_cfg(cfg_styled_);
            generator_->set_shared_curve("velocity_scale", velocity_);
            generator_->set_evolve(evolve_);
        }

        phase_.store(Phase::Running);
        writeStatus("Ready");
        playing_.store(true);
        fade_in_remaining_ = kFadeSamples;

        juce::DynamicObject::Ptr ok = new juce::DynamicObject();
        ok->setProperty("ok", true);
        ok->setProperty("duration", src_secs);
        juce::Array<juce::var> parr;
        parr.ensureStorageAllocated((int)peaks.size());
        for (float p : peaks) parr.add(juce::var(p));
        ok->setProperty("peaks", juce::var(parr));
        load_in_flight_.store(false, std::memory_order_release);
        if (completion) completion(juce::var(ok.get()));
    }
    catch (const std::exception& ex) {
        fail(juce::String(ex.what()));
    }
}

// ── transport ────────────────────────────────────────────────────────
void MorphEngine::startPlayback() {
    if (phase_.load() == Phase::Running) {
        fade_in_remaining_ = kFadeSamples;
        playing_.store(true);
    }
}
void MorphEngine::stopPlayback() { playing_.store(false); }

// ── live controls ────────────────────────────────────────────────────
void MorphEngine::setDenoise(float v) {   // UI "Amount" -> morph original<->styled
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); denoise_ = v; }
    if (generator_) generator_->set_morph_amount(v);
}
void MorphEngine::setSourceBlend(float v) {   // UI "Intensity" -> styled-target strength
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); source_blend_ = v; }
    if (generator_) generator_->set_style_intensity(v);
}
void MorphEngine::setVelocity(float v) {
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); velocity_ = v; }
    if (generator_) generator_->set_shared_curve("velocity_scale", v);
}
void MorphEngine::setCfg(float v) {   // UI "CFG" -> styled-target guidance strength
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); cfg_styled_ = v; }
    if (generator_) generator_->set_cfg(v);
}
void MorphEngine::setWindowSeconds(double s) {
    window_seconds_.store(s, std::memory_order_release);   // applied on next load
}
void MorphEngine::setEvolve(bool on) {
    { std::lock_guard<std::mutex> lk(ctrl_mutex_); evolve_ = on; }
    if (generator_) generator_->set_evolve(on);
}
void MorphEngine::setPrompt(const std::string& text) {
    if (generator_) generator_->set_prompt(text);
}
bool MorphEngine::setQuality(bool on) {
    const bool prev = quality_.exchange(on, std::memory_order_acq_rel);
    return prev != on;   // changed -> caller reloads the source to apply
}

MorphEngine::MorphState MorphEngine::getMorphState() const {
    MorphState s;
    s.phase    = (int)phase_.load();
    s.playing  = playing_.load();
    s.loopSecs = loop_secs_.load();
    const int L = loop_len_.load();
    const double phase = (ring_ && L > 0) ? (double)(ring_->pos() % L) / (double)L : 0.0;
    {
        std::lock_guard<std::mutex> lk(ctrl_mutex_);
        s.denoise = denoise_; s.sourceBlend = source_blend_;
        s.velocity = velocity_; s.evolve = evolve_;
    }
    if (generator_) {
        auto gs = generator_->stats();
        s.tickMs = gs.tick_ms; s.decodeMs = gs.decode_ms; s.completions = gs.completions;
        // Playhead: when scanning a long track, the track position is the chunk's
        // source start + how far the play cursor has advanced past the chunk's output
        // base (NOT the loop phase — that sawtooths against the chunk geometry and made
        // the cursor jump). total_read() is the live monotonic play cursor.
        if (gs.full_samples > 0 && gs.window_samples > 0) {
            const long tr = ring_ ? ring_->total_read() : 0;
            double trackPos = (double)gs.scan_start + (double)(tr - gs.chunk_base);
            trackPos = std::max(0.0, std::min((double)gs.full_samples, trackPos));
            s.progress = trackPos / (double)gs.full_samples;
            s.loopSecs = (double)gs.full_samples / kSampleRate;   // playhead spans the track
        } else {
            s.progress = phase;
        }
    } else {
        s.progress = phase;
    }
    return s;
}

// ── AudioSource ──────────────────────────────────────────────────────
void MorphEngine::prepareToPlay(int samplesPerBlockExpected, double sampleRate) {
    host_sample_rate_ = sampleRate;
    interp_l_.reset();
    interp_r_.reset();
    const double ratio = (double)kSampleRate / host_sample_rate_;
    const size_t cap = (size_t)std::ceil(samplesPerBlockExpected * ratio) + 8;
    temp_in_l_.assign(cap, 0.0f);
    temp_in_r_.assign(cap, 0.0f);
    ring_read_.assign(2, std::vector<float>((size_t)samplesPerBlockExpected + 8, 0.0f));
}

void MorphEngine::releaseResources() {}

void MorphEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& info) {
    info.clearActiveBufferRegion();
    if (!playing_.load(std::memory_order_acquire)) return;
    sa3::rt::AudioRing* ring = ring_.get();
    if (ring == nullptr || !ring->has_audio()) return;

    const int wanted = info.numSamples;
    const double ratio = (double)kSampleRate / host_sample_rate_;
    const int numIn = (int)std::ceil(wanted * ratio) + 4;
    if (numIn > (int)temp_in_l_.size()) return;   // prepareToPlay underspec'd

    // PEEK numIn 44.1k frames from the ring (no cursor advance), resample to the
    // host block, then advance the cursor by exactly what the interpolator
    // consumed — so the play cursor tracks wall-clock playback at any host SR.
    if ((int)ring_read_[0].size() < numIn) {
        ring_read_[0].resize(numIn);
        ring_read_[1].resize(numIn);
    }
    ring->read_peek(numIn, ring_read_);
    for (int i = 0; i < numIn; ++i) {
        temp_in_l_[(size_t)i] = ring_read_[0][(size_t)i];
        temp_in_r_[(size_t)i] = ring_read_[1][(size_t)i];
    }

    const int outChans = info.buffer->getNumChannels();
    float* outL = info.buffer->getWritePointer(0, info.startSample);
    float* outR = (outChans > 1) ? info.buffer->getWritePointer(1, info.startSample) : outL;
    const int used = interp_l_.process(ratio, temp_in_l_.data(), outL, wanted);
    if (outR != outL) interp_r_.process(ratio, temp_in_r_.data(), outR, wanted);
    ring->advance(used);

    // Fade-in ramp (masks the click + the interpolator's cold-start transient).
    if (fade_in_remaining_ > 0) {
        const int n = std::min(wanted, fade_in_remaining_);
        for (int i = 0; i < n; ++i) {
            const float g = (float)(kFadeSamples - fade_in_remaining_) / (float)kFadeSamples;
            outL[i] *= g;
            if (outR != outL) outR[i] *= g;
            --fade_in_remaining_;
        }
    }

    // Soft-clip safety limiter. Style-transfer CFG can push peaks past ±1 (esp.
    // with aggressive prompts / high Character). tanh(x) asymptotes to ±1 for ANY
    // input, so the output is hard-bounded — no digital clipping is possible. It's
    // near-transparent at low levels (tanh(x)≈x for |x|<0.4) and smoothly
    // saturates as it approaches the rails. We only engage it when the block
    // actually has hot samples (>0.7), so quiet/in-range audio passes untouched.
    float blockPeak = 0.0f;
    for (int i = 0; i < wanted; ++i) {
        blockPeak = std::max(blockPeak, std::abs(outL[i]));
        if (outR != outL) blockPeak = std::max(blockPeak, std::abs(outR[i]));
    }
    if (blockPeak > 0.7f) {
        for (int i = 0; i < wanted; ++i) {
            outL[i] = std::tanh(outL[i]);
            if (outR != outL) outR[i] = std::tanh(outR[i]);
        }
    }
}

}  // namespace sa3morph
