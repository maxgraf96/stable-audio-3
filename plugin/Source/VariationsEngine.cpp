#include "VariationsEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "sa3_orchestrator.h"

namespace sa3plugin {

namespace {

constexpr const char* kT5GemmaPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/t5gemma_f16.safetensors";
constexpr const char* kDitPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/dit_medium_f16.safetensors";
constexpr const char* kEncoderPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/same_l_encoder_f32.safetensors";
constexpr const char* kDecoderPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/same_l_decoder_f32.safetensors";

constexpr int kSampleRate = sa3::orch::SAMPLE_RATE;   // 44100

juce::var peaksToVar(const std::vector<float>& peaks) {
    juce::Array<juce::var> arr;
    arr.ensureStorageAllocated(static_cast<int>(peaks.size()));
    for (float p : peaks) arr.add(juce::var(p));
    return juce::var(arr);
}

}  // namespace

// ── Audio decoding helpers ───────────────────────────────────────────

juce::AudioBuffer<float> VariationsEngine::decodeToStereo44k(const juce::MemoryBlock& bytes)
{
    if (bytes.getSize() == 0) {
        throw std::runtime_error("source audio is empty");
    }
    juce::AudioFormatManager fmt;
    fmt.registerBasicFormats();
    auto stream = std::make_unique<juce::MemoryInputStream>(bytes, false);
    std::unique_ptr<juce::AudioFormatReader> reader(fmt.createReaderFor(std::move(stream)));
    if (!reader) {
        throw std::runtime_error(
            "unsupported audio format (expected WAV/AIFF/FLAC/Ogg)");
    }
    const int    src_channels = static_cast<int>(reader->numChannels);
    const int    src_samples  = static_cast<int>(reader->lengthInSamples);
    const double src_sr       = reader->sampleRate;
    if (src_samples <= 0 || src_channels < 1) {
        throw std::runtime_error("source audio has no samples");
    }

    juce::AudioBuffer<float> src(2, src_samples);
    src.clear();
    reader->read(&src, /*startSample=*/0, src_samples,
                 /*readerStartSample=*/0,
                 /*useLeftChan=*/true, /*useRightChan=*/true);
    if (src_channels == 1) {
        src.copyFrom(/*destChan=*/1, /*destOffset=*/0,
                     src, /*srcChan=*/0, /*srcOffset=*/0, src_samples);
    }

    const bool needs_resample = std::abs(src_sr - kSampleRate) > 0.5;
    if (! needs_resample) return src;

    const int dst_samples = static_cast<int>(
        std::ceil(src_samples * (kSampleRate / src_sr)));
    juce::AudioBuffer<float> out(2, dst_samples);
    out.clear();
    const double ratio = src_sr / static_cast<double>(kSampleRate);
    for (int c = 0; c < 2; ++c) {
        juce::LagrangeInterpolator interp;
        interp.reset();
        interp.process(ratio,
                       src.getReadPointer(c),
                       out.getWritePointer(c),
                       dst_samples);
    }
    return out;
}

std::vector<float> VariationsEngine::extractPeaks(
    const juce::AudioBuffer<float>& buf, int n)
{
    std::vector<float> peaks(static_cast<size_t>(n), 0.0f);
    const int len = buf.getNumSamples();
    if (len <= 0 || n <= 0) return peaks;
    const int   nch = std::max(1, buf.getNumChannels());
    const float* L = buf.getReadPointer(0);
    const float* R = (nch > 1) ? buf.getReadPointer(1) : L;

    const double win = static_cast<double>(len) / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        const int start = static_cast<int>(std::floor(i * win));
        const int end   = std::min(len, static_cast<int>(std::floor((i + 1) * win)));
        float maxV = 0.0f;
        for (int j = start; j < end; ++j) {
            const float v = std::max(std::abs(L[j]), std::abs(R[j]));
            if (v > maxV) maxV = v;
        }
        peaks[static_cast<size_t>(i)] = std::min(1.0f, std::max(0.04f, maxV));
    }
    return peaks;
}

// ── VariationsEngine ─────────────────────────────────────────────────

VariationsEngine::VariationsEngine()
    : juce::Thread("SA3 MLX Worker")
{
    startThread();
}

VariationsEngine::~VariationsEngine()
{
    shutdown_.store(true, std::memory_order_release);
    notify();
    stopThread(30000);
}

// ── Pipeline load ────────────────────────────────────────────────────

void VariationsEngine::requestLoad()
{
    LoadPhase expected = LoadPhase::NotLoaded;
    if (load_phase_.compare_exchange_strong(expected, LoadPhase::Loading,
                                            std::memory_order_acq_rel)) {
        load_pending_.store(true, std::memory_order_release);
        writeStatus("Loading models...");
        notify();
    }
}

// ── Job submission ───────────────────────────────────────────────────

bool VariationsEngine::requestUploadSource(juce::MemoryBlock audioBytes,
                                           int peaks_n,
                                           CompletionFn completion)
{
    {
        std::lock_guard<std::mutex> lk(job_mutex_);
        if (busy_.load(std::memory_order_acquire) || pending_job_.has_value()) {
            return false;
        }
        Job j;
        j.kind         = JobKind::UploadSource;
        j.source_bytes = std::move(audioBytes);
        j.peaks_n      = peaks_n;
        j.completion   = std::move(completion);
        pending_job_ = std::move(j);
        busy_.store(true, std::memory_order_release);
    }
    notify();
    return true;
}

bool VariationsEngine::requestGenerate(GenerateRequest req, int peaks_n,
                                       CompletionFn completion)
{
    if (load_phase_.load(std::memory_order_acquire) != LoadPhase::Loaded) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(play_mutex_);
        if (source_buf_.getNumSamples() <= 0) return false;
    }
    {
        std::lock_guard<std::mutex> lk(job_mutex_);
        if (busy_.load(std::memory_order_acquire) || pending_job_.has_value()) {
            return false;
        }
        Job j;
        j.kind       = JobKind::Generate;
        j.gen_req    = std::move(req);
        j.peaks_n    = peaks_n;
        j.completion = std::move(completion);
        pending_job_ = std::move(j);
        busy_.store(true, std::memory_order_release);
    }
    notify();
    return true;
}

// ── Status / state accessors ─────────────────────────────────────────

juce::String VariationsEngine::getStatus() const
{
    std::lock_guard<std::mutex> lk(peaks_mutex_);
    return status_;
}

void VariationsEngine::writeStatus(juce::String s)
{
    std::lock_guard<std::mutex> lk(peaks_mutex_);
    status_ = std::move(s);
}

std::vector<float> VariationsEngine::getSourcePeaks() const
{
    std::lock_guard<std::mutex> lk(peaks_mutex_);
    return source_peaks_;
}

std::vector<float> VariationsEngine::getVariationPeaks(int idx) const
{
    std::lock_guard<std::mutex> lk(peaks_mutex_);
    if (idx < 0 || idx >= static_cast<int>(variation_peaks_.size())) return {};
    return variation_peaks_[static_cast<size_t>(idx)];
}

int VariationsEngine::getVariationCount() const
{
    std::lock_guard<std::mutex> lk(peaks_mutex_);
    return static_cast<int>(variation_peaks_.size());
}

// ── Playback control ─────────────────────────────────────────────────

void VariationsEngine::play(int idx)
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    // Validate the target buffer exists; on failure, stop.
    const juce::AudioBuffer<float>* target = nullptr;
    if (idx == -1) {
        if (source_buf_.getNumSamples() > 0) target = &source_buf_;
    } else if (idx >= 0 && idx < static_cast<int>(variation_bufs_.size())) {
        if (variation_bufs_[static_cast<size_t>(idx)].getNumSamples() > 0)
            target = &variation_bufs_[static_cast<size_t>(idx)];
    }
    if (target == nullptr) {
        active_idx_ = -2; active_playing_ = false; return;
    }

    if (active_idx_ != idx) {
        // Switching buffers — preserve the playback fraction (0..1) so A/B
        // comparison between variations feels continuous. If we had no
        // active buffer (-2) the fraction stays 0.
        double fraction = 0.0;
        const juce::AudioBuffer<float>* prev = nullptr;
        if (active_idx_ == -1)                                                          prev = &source_buf_;
        else if (active_idx_ >= 0 && active_idx_ < static_cast<int>(variation_bufs_.size()))
            prev = &variation_bufs_[static_cast<size_t>(active_idx_)];
        if (prev != nullptr && prev->getNumSamples() > 0) {
            fraction = static_cast<double>(active_position_) /
                       static_cast<double>(prev->getNumSamples());
            fraction = std::max(0.0, std::min(1.0, fraction));
        }
        active_idx_      = idx;
        active_position_ = static_cast<int64_t>(
            fraction * static_cast<double>(target->getNumSamples()));
        if (active_position_ >= target->getNumSamples()) active_position_ = 0;
    }
    active_playing_ = true;
}

void VariationsEngine::pausePlayback()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    active_playing_ = false;
}

void VariationsEngine::resumePlayback()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    if (active_idx_ != -2) active_playing_ = true;
}

void VariationsEngine::stopPlayback()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    active_idx_      = -2;
    active_position_ = 0;
    active_playing_  = false;
}

void VariationsEngine::seek(double fraction)
{
    fraction = std::max(0.0, std::min(1.0, fraction));
    std::lock_guard<std::mutex> lk(play_mutex_);
    int64_t len = 0;
    if (active_idx_ == -1)                                 len = source_buf_.getNumSamples();
    else if (active_idx_ >= 0 && active_idx_ < static_cast<int>(variation_bufs_.size()))
        len = variation_bufs_[static_cast<size_t>(active_idx_)].getNumSamples();
    if (len > 0) {
        active_position_ = static_cast<int64_t>(fraction * static_cast<double>(len));
    }
}

PlayState VariationsEngine::getPlayState() const
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    PlayState s;
    s.idx      = active_idx_;
    s.playing  = active_playing_;
    int64_t len = 0;
    if (active_idx_ == -1)                                 len = source_buf_.getNumSamples();
    else if (active_idx_ >= 0 && active_idx_ < static_cast<int>(variation_bufs_.size()))
        len = variation_bufs_[static_cast<size_t>(active_idx_)].getNumSamples();
    if (len > 0) {
        s.progress = static_cast<double>(active_position_) / static_cast<double>(len);
        s.duration = static_cast<double>(len) / static_cast<double>(kSampleRate);
    }
    return s;
}

// ── AudioSource (audio thread) ───────────────────────────────────────

void VariationsEngine::prepareToPlay(int /*samplesPerBlockExpected*/,
                                     double /*sampleRate*/)
{
    // Buffers are stored at 44.1 kHz; if the host runs at a different SR
    // playback will pitch-shift. We'll add a resampler stage in a follow-up
    // when plugin-mode lands. For now the standalone defaults to 44.1k on
    // Mac so this is a non-issue.
}

void VariationsEngine::releaseResources() {}

void VariationsEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& info)
{
    info.clearActiveBufferRegion();
    // Try-lock keeps the audio thread real-time-ish: if the message thread
    // is mid-swap we just play silence for this block rather than blocking.
    std::unique_lock<std::mutex> lk(play_mutex_, std::try_to_lock);
    if (!lk.owns_lock()) return;
    if (active_idx_ == -2 || !active_playing_) return;

    const juce::AudioBuffer<float>* src = nullptr;
    if (active_idx_ == -1) {
        src = &source_buf_;
    } else if (active_idx_ >= 0 && active_idx_ < static_cast<int>(variation_bufs_.size())) {
        src = &variation_bufs_[static_cast<size_t>(active_idx_)];
    }
    if (src == nullptr || src->getNumSamples() <= 0) return;

    const int srcLen   = src->getNumSamples();
    const int srcChans = src->getNumChannels();
    const int outChans = info.buffer->getNumChannels();
    const int wanted   = info.numSamples;

    // Loop forever (until pause/stop). Handle the case where a single
    // output block straddles the buffer end — wrap inline so there's no
    // half-sample gap on the loop point.
    int written = 0;
    while (written < wanted) {
        const int remaining = static_cast<int>(srcLen - active_position_);
        const int n         = std::min(wanted - written, remaining);
        for (int ch = 0; ch < outChans; ++ch) {
            const int srcCh = std::min(ch, srcChans - 1);
            info.buffer->copyFrom(ch, info.startSample + written,
                                  *src, srcCh,
                                  static_cast<int>(active_position_), n);
        }
        written += n;
        active_position_ += n;
        if (active_position_ >= srcLen) active_position_ = 0;
    }
}

// ── Worker thread ────────────────────────────────────────────────────

void VariationsEngine::run()
{
    while (! shutdown_.load(std::memory_order_acquire)) {
        wait(-1);
        if (shutdown_.load(std::memory_order_acquire)) break;

        // One-shot pipeline load.
        if (load_pending_.exchange(false, std::memory_order_acq_rel)) {
            doLoad();
        }

        std::optional<Job> job;
        {
            std::lock_guard<std::mutex> lk(job_mutex_);
            job = std::move(pending_job_);
            pending_job_.reset();
        }
        if (! job.has_value()) {
            busy_.store(false, std::memory_order_release);
            continue;
        }

        juce::var result;
        if (job->kind == JobKind::UploadSource) {
            result = doUploadSource(std::move(job->source_bytes), job->peaks_n);
        } else if (job->kind == JobKind::Generate) {
            result = doGenerate(std::move(*job->gen_req), job->peaks_n);
        }

        busy_.store(false, std::memory_order_release);
        if (job->completion) job->completion(result);
    }
}

void VariationsEngine::doLoad()
{
    try {
        auto p = std::make_unique<sa3::orch::Pipeline>(
            sa3::orch::load_pipeline(kT5GemmaPath, kDitPath,
                                     kEncoderPath, kDecoderPath,
                                     mlx::core::float16));
        pipeline_ = std::move(p);
        load_phase_.store(LoadPhase::Loaded, std::memory_order_release);
        writeStatus("Ready");
    }
    catch (const std::exception& ex) {
        load_phase_.store(LoadPhase::Error, std::memory_order_release);
        writeStatus(juce::String("Load failed: ") + ex.what());
    }
}

juce::var VariationsEngine::doUploadSource(juce::MemoryBlock bytes, int peaks_n)
{
    try {
        writeStatus("Decoding source...");
        auto buf   = decodeToStereo44k(bytes);
        auto peaks = extractPeaks(buf, peaks_n);
        const double dur = static_cast<double>(buf.getNumSamples()) /
                           static_cast<double>(kSampleRate);

        {
            std::lock_guard<std::mutex> lk(play_mutex_);
            // Wipe any active playback so we don't read a half-replaced
            // buffer from the audio thread on the next block.
            active_idx_      = -2;
            active_playing_  = false;
            active_position_ = 0;
            source_buf_      = std::move(buf);
        }
        {
            std::lock_guard<std::mutex> lk(peaks_mutex_);
            source_peaks_ = peaks;
            // New source invalidates old variation peaks/buffers — make the
            // UI render an empty result grid.
            variation_peaks_.clear();
        }
        {
            std::lock_guard<std::mutex> lk(play_mutex_);
            variation_bufs_.clear();
        }
        writeStatus("Ready");

        juce::DynamicObject::Ptr obj = new juce::DynamicObject();
        obj->setProperty("ok",       true);
        obj->setProperty("duration", dur);
        obj->setProperty("peaks",    peaksToVar(peaks));
        return juce::var(obj.get());
    }
    catch (const std::exception& ex) {
        writeStatus(juce::String("Error: ") + ex.what());
        juce::DynamicObject::Ptr obj = new juce::DynamicObject();
        obj->setProperty("ok",    false);
        obj->setProperty("error", juce::String(ex.what()));
        return juce::var(obj.get());
    }
}

juce::var VariationsEngine::doGenerate(GenerateRequest req, int peaks_n)
{
    try {
        if (! pipeline_) throw std::runtime_error("pipeline not loaded");

        // Snapshot the source buffer for use as init_audio (planar fp32
        // pointer + sample count — the orchestrator copies it internally).
        std::vector<float> src_planar;
        int src_samples = 0;
        {
            std::lock_guard<std::mutex> lk(play_mutex_);
            src_samples = source_buf_.getNumSamples();
            if (src_samples <= 0) throw std::runtime_error("no source loaded");
            src_planar.resize(static_cast<size_t>(2) * src_samples);
            for (int c = 0; c < 2; ++c) {
                const int srcCh = std::min(c, source_buf_.getNumChannels() - 1);
                std::copy(source_buf_.getReadPointer(srcCh),
                          source_buf_.getReadPointer(srcCh) + src_samples,
                          src_planar.data() + c * src_samples);
            }
        }

        const double src_seconds = static_cast<double>(src_samples) /
                                   static_cast<double>(kSampleRate);
        float seconds = req.seconds > 0.0f ? req.seconds : static_cast<float>(src_seconds);
        seconds = std::clamp(seconds, 1.0f, 30.0f);

        sa3::orch::InitAudio init_audio{
            src_planar.data(), 2, src_samples,
        };

        std::vector<sa3::orch::CandidateSpec> specs;
        if (req.preset == "free") {
            specs = sa3::orch::build_free_preset(seconds, req.seed, req.noise);
        } else if (req.preset == "app") {
            specs = sa3::orch::build_app_preset(
                seconds, req.seed, req.user_prompt, req.bpm, req.key,
                req.beats_per_bar, req.noise, req.cfg_a2a, req.cfg_inpaint, req.apg);
        } else {
            throw std::runtime_error("unknown preset '" + req.preset + "' (expected free|app)");
        }

        const int total = static_cast<int>(specs.size());

        // Clear previous variation buffers + peaks atomically.
        {
            std::lock_guard<std::mutex> lk(play_mutex_);
            variation_bufs_.assign(total, juce::AudioBuffer<float>{});
            // If a previous variation was active, stop playback so we
            // don't read a stale buffer.
            if (active_idx_ >= 0) {
                active_idx_      = -2;
                active_playing_  = false;
                active_position_ = 0;
            }
        }
        {
            std::lock_guard<std::mutex> lk(peaks_mutex_);
            variation_peaks_.assign(total, std::vector<float>{});
        }

        juce::Array<juce::var> slots;
        for (int i = 0; i < total; ++i) {
            writeStatus(juce::String::formatted("Generating %d/%d...", i + 1, total));
            auto outs = sa3::orch::run_variations(
                *pipeline_, init_audio, {specs[static_cast<size_t>(i)]},
                seconds, req.steps);
            if (outs.size() != 1) {
                throw std::runtime_error("run_variations returned unexpected count");
            }
            const auto& v = outs[0];

            // Build an AudioBuffer<float> from the planar fp32 output.
            juce::AudioBuffer<float> ab(2, v.samples);
            ab.clear();
            for (int c = 0; c < 2; ++c) {
                const int srcCh = std::min(c, v.channels - 1);
                std::copy(v.audio.data() + srcCh * v.samples,
                          v.audio.data() + srcCh * v.samples + v.samples,
                          ab.getWritePointer(c));
            }
            auto peaks = extractPeaks(ab, peaks_n);
            const double dur = static_cast<double>(v.samples) /
                               static_cast<double>(kSampleRate);

            {
                std::lock_guard<std::mutex> lk(play_mutex_);
                variation_bufs_[static_cast<size_t>(i)] = std::move(ab);
            }
            {
                std::lock_guard<std::mutex> lk(peaks_mutex_);
                variation_peaks_[static_cast<size_t>(i)] = peaks;
            }

            juce::DynamicObject::Ptr slot = new juce::DynamicObject();
            slot->setProperty("idx",      i);
            slot->setProperty("steer",    juce::String(v.spec.steer));
            slot->setProperty("mode",     juce::String(v.spec.mode));
            slot->setProperty("duration", dur);
            slot->setProperty("peaks",    peaksToVar(peaks));
            slots.add(juce::var(slot.get()));
        }
        writeStatus("Ready");

        juce::DynamicObject::Ptr obj = new juce::DynamicObject();
        obj->setProperty("ok",    true);
        obj->setProperty("count", total);
        obj->setProperty("slots", juce::var(slots));
        return juce::var(obj.get());
    }
    catch (const std::exception& ex) {
        writeStatus(juce::String("Error: ") + ex.what());
        juce::DynamicObject::Ptr obj = new juce::DynamicObject();
        obj->setProperty("ok",    false);
        obj->setProperty("error", juce::String(ex.what()));
        return juce::var(obj.get());
    }
}

}  // namespace sa3plugin
