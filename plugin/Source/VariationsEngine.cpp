#include "VariationsEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "sa3_orchestrator.h"

namespace sa3plugin {

namespace {

constexpr const char* kT5GemmaFile = "t5gemma_f16.safetensors";
constexpr const char* kDitFile     = "dit_medium_f16.safetensors";
constexpr const char* kEncoderFile = "same_l_encoder_f32.safetensors";
constexpr const char* kDecoderFile = "same_l_decoder_f32.safetensors";

// Dev fallback when the plugin runs from a build tree without bundled
// models. Production bundles ship the safetensors under
// Contents/Resources/models/ and never touch this path.
constexpr const char* kDevModelsDir =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx";

// Where the four safetensors files live for the currently-running plugin.
// Resolution order:
//   1. $SA3_MODELS_DIR — dev override so we don't have to copy >1 GB of
//      weights into every bundle on each build.
//   2. <plugin>/Contents/Resources/models — production layout, written by
//      the CMake post-build step. juce::currentExecutableFile returns the
//      plugin's own binary even when loaded inside a DAW.
//   3. kDevModelsDir — last-resort source-tree path.
juce::File resolveModelsDir()
{
    if (const char* env = std::getenv("SA3_MODELS_DIR"); env && *env) {
        juce::File f(juce::String::fromUTF8(env));
        if (f.isDirectory()) return f;
    }
    auto exe = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    auto bundled = exe.getParentDirectory()        // Contents/MacOS
                      .getParentDirectory()         // Contents
                      .getChildFile("Resources")
                      .getChildFile("models");
    if (bundled.isDirectory()) return bundled;
    return juce::File(kDevModelsDir);
}

constexpr int kSampleRate = sa3::orch::SAMPLE_RATE;   // 44100

// ~5.8 ms at 44.1 k / ~5.3 ms at 48 k — long enough to kill the click at
// start/stop, short enough to feel instant. Doubles as a warm-up runway
// for LagrangeInterpolator (which interpolates against zeroed history for
// the first ~4 samples after reset, so an audible transient is masked).
constexpr int kFadeSamples = 256;

// ~3 ms at 44.1 k — equal-time linear crossfade when cycling between
// variations while playing. Two variations of the same source have
// near-identical envelopes but different phase/note choices, so an
// instant buffer swap produces a small click; this masks it without
// being audibly slow.
constexpr int kCrossfadeSamples = 128;

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

juce::File VariationsEngine::writeVariationToTempFile(int idx,
                                                      const juce::String& baseName,
                                                      const juce::String& stamp)
{
    // Copy the buffer out under the lock — encoding + disk I/O happen
    // unlocked so the audio thread isn't blocked.
    juce::AudioBuffer<float> buf;
    {
        std::lock_guard<std::mutex> lk(play_mutex_);
        if (idx < 0 || idx >= static_cast<int>(variation_bufs_.size())) return {};
        const auto& src = variation_bufs_[static_cast<size_t>(idx)];
        if (src.getNumSamples() <= 0) return {};
        buf = src;   // AudioBuffer copy ctor
    }

    // Sanitize the filename: strip extension, restrict to a safe charset,
    // fall back to "SA3" if empty after stripping.
    juce::String safe = baseName;
    const int dot = safe.lastIndexOfChar('.');
    if (dot > 0) safe = safe.substring(0, dot);
    safe = safe.retainCharacters(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ");
    safe = safe.trim();
    if (safe.isEmpty()) safe = "SA3";

    // Sanitize stamp using the same charset, then append after the var
    // number so the filename is `<base>_var<N>_<stamp>.wav`. Each
    // generation feeds a fresh stamp from JS, so dropping a variation
    // into a DAW project never collides with an earlier generation's
    // file (or another plugin instance's) — DAW clips that reference it
    // stay valid.
    juce::String safeStamp = stamp.retainCharacters(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_");
    juce::String fileName = safe + "_var" + juce::String(idx + 1);
    if (safeStamp.isNotEmpty()) fileName += "_" + safeStamp;
    fileName += ".wav";
    const auto out = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile(fileName);
    out.deleteFile();   // replaceWithData would do this too but JUCE's writer needs a fresh stream

    // Stream-based WAV write via juce::WavAudioFormat. The format writer
    // takes ownership of the output stream once created — we release the
    // unique_ptr after the createWriterFor call succeeds.
    auto stream = out.createOutputStream();
    if (stream == nullptr) return {};

    juce::WavAudioFormat fmt;
    std::unique_ptr<juce::AudioFormatWriter> writer(
        fmt.createWriterFor(stream.get(),
                            static_cast<double>(kSampleRate),
                            /*numChannels=*/2,
                            /*bitsPerSample=*/16,
                            /*metadataValues=*/{},
                            /*qualityOptionIndex=*/0));
    if (writer == nullptr) return {};
    stream.release();   // writer now owns the stream
    writer->writeFromAudioSampleBuffer(buf, 0, buf.getNumSamples());
    writer.reset();     // flush + close
    return out;
}

std::vector<std::byte> VariationsEngine::getVariationWavBytes(int idx)
{
    juce::AudioBuffer<float> buf;
    {
        std::lock_guard<std::mutex> lk(play_mutex_);
        if (idx < 0 || idx >= static_cast<int>(variation_bufs_.size())) return {};
        const auto& src = variation_bufs_[static_cast<size_t>(idx)];
        if (src.getNumSamples() <= 0) return {};
        buf = src;
    }

    // Encode into a MemoryBlock-backed stream. The writer takes ownership
    // of the stream pointer once createWriterFor succeeds; its destructor
    // flushes + closes the stream, after which the MemoryBlock holds the
    // final WAV bytes.
    juce::MemoryBlock mb;
    {
        auto stream = std::make_unique<juce::MemoryOutputStream>(mb, false);
        juce::WavAudioFormat fmt;
        std::unique_ptr<juce::AudioFormatWriter> writer(
            fmt.createWriterFor(stream.get(),
                                static_cast<double>(kSampleRate),
                                /*numChannels=*/2,
                                /*bitsPerSample=*/16,
                                /*metadataValues=*/{},
                                /*qualityOptionIndex=*/0));
        if (writer == nullptr) return {};
        stream.release();   // writer owns
        writer->writeFromAudioSampleBuffer(buf, 0, buf.getNumSamples());
    }   // writer destructor flushes & closes — mb is complete now

    std::vector<std::byte> out(mb.getSize());
    std::memcpy(out.data(), mb.getData(), mb.getSize());
    return out;
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
        active_idx_ = -2; active_playing_ = false;
        fade_in_remaining_ = 0; fade_out_remaining_ = 0;
        return;
    }

    if (active_idx_ != idx) {
        // Snapshot the outgoing source position for the crossfade — but only
        // if we were actually producing audio. A cold start (active_idx_ ==
        // -2) has nothing to fade out from; a mid-fade-out also doesn't
        // crossfade because the user explicitly paused.
        const bool can_crossfade = active_playing_ && active_idx_ != -2
                                && fade_out_remaining_ == 0;
        if (can_crossfade) {
            prev_idx_            = active_idx_;
            prev_position_       = active_position_;
            prev_sub_pos_        = 0.0;          // start fractional read fresh
            crossfade_remaining_ = kCrossfadeSamples;
        }

        if (one_shot_mode_) {
            // One-shot: cycling between variations restarts from the top
            // — typical "drum hit / kick" comparison flow.
            active_idx_      = idx;
            active_position_ = 0;
        } else {
            // Loop: preserve the playback fraction (0..1) so A/B
            // comparison between variations feels continuous.
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
        // Fresh interpolator state for the new buffer — the old buffer's
        // history lives on in interp_*_old_ for the crossfade window.
        interp_l_.reset();
        interp_r_.reset();
    }

    const bool was_silent = !active_playing_;
    active_playing_ = true;
    if (was_silent) {
        // Cold start — full fade-in, also masks the interpolator warm-up.
        fade_in_remaining_ = kFadeSamples;
        interp_l_.reset();
        interp_r_.reset();
    } else if (fade_out_remaining_ > 0) {
        // Play interrupted an in-flight fade-out — pick up the gain ramp
        // upward from wherever the fade-out left it, so we don't pop back
        // to full gain.
        fade_in_remaining_ = kFadeSamples - fade_out_remaining_;
    }
    fade_out_remaining_ = 0;
}

void VariationsEngine::pausePlayback()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    if (! active_playing_) return;
    if (fade_out_remaining_ > 0) return;            // already fading out
    if (fade_in_remaining_ > 0) {
        // Pause mid-fade-in — invert the ramp to come back down from the
        // gain we'd reached, not from full.
        fade_out_remaining_ = kFadeSamples - fade_in_remaining_;
        fade_in_remaining_  = 0;
    } else {
        fade_out_remaining_ = kFadeSamples;
    }
    // active_playing_ stays true until the fade completes (in
    // getNextAudioBlock); UI poll briefly shows "playing" while it fades.
}

void VariationsEngine::resumePlayback()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    if (active_idx_ == -2) return;
    const bool was_silent = !active_playing_;
    active_playing_ = true;
    if (was_silent) {
        fade_in_remaining_ = kFadeSamples;
        interp_l_.reset();
        interp_r_.reset();
    }
    fade_out_remaining_ = 0;
}

void VariationsEngine::stopPlayback()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    active_idx_      = -2;
    active_position_ = 0;
    active_playing_  = false;
    fade_in_remaining_  = 0;
    fade_out_remaining_ = 0;
    crossfade_remaining_ = 0;
    prev_idx_           = -2;
    prev_sub_pos_       = 0.0;
    interp_l_.reset();
    interp_r_.reset();
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
    // Reset interpolators on seek — otherwise the old fractional position
    // would cause a brief click at the jump point.
    interp_l_.reset();
    interp_r_.reset();
}

PlayState VariationsEngine::getPlayState() const
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    PlayState s;
    s.idx      = active_idx_;
    s.playing  = active_playing_;
    s.oneShot  = one_shot_mode_;
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

void VariationsEngine::setOneShotMode(bool oneShot)
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    one_shot_mode_ = oneShot;
}

bool VariationsEngine::isOneShotMode() const
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    return one_shot_mode_;
}

// ── AudioSource (audio thread) ───────────────────────────────────────

void VariationsEngine::prepareToPlay(int samplesPerBlockExpected,
                                     double sampleRate)
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    host_sample_rate_ = sampleRate > 0.0 ? sampleRate : 44100.0;
    // Lagrange consumes up to ceil(numOut * sourceSR/destSR) input samples
    // per block; +8 for safety margin and the 4-sample interpolator window.
    const double ratio = static_cast<double>(kSampleRate) / host_sample_rate_;
    const int    maxIn = static_cast<int>(
        std::ceil(samplesPerBlockExpected * ratio)) + 8;
    temp_in_l_.assign(static_cast<size_t>(maxIn), 0.0f);
    temp_in_r_.assign(static_cast<size_t>(maxIn), 0.0f);
    scratch_l_.assign(static_cast<size_t>(samplesPerBlockExpected), 0.0f);
    scratch_r_.assign(static_cast<size_t>(samplesPerBlockExpected), 0.0f);
    interp_l_.reset();
    interp_r_.reset();
    crossfade_remaining_ = 0;
    prev_idx_            = -2;
    prev_sub_pos_        = 0.0;
}

void VariationsEngine::releaseResources()
{
    std::lock_guard<std::mutex> lk(play_mutex_);
    interp_l_.reset();
    interp_r_.reset();
    crossfade_remaining_ = 0;
    prev_idx_            = -2;
    prev_sub_pos_        = 0.0;
}

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

    const int    srcLen   = src->getNumSamples();
    const int    srcChans = src->getNumChannels();
    const int    outChans = info.buffer->getNumChannels();
    const int    wanted   = info.numSamples;
    const double ratio    = static_cast<double>(kSampleRate) / host_sample_rate_;

    // Lagrange consumes at most ceil(wanted * ratio) input samples — pad
    // by 4 for the interpolator's internal lookahead so we never read
    // past the end of our temp buffer.
    const int numIn = static_cast<int>(std::ceil(wanted * ratio)) + 4;
    if (numIn > static_cast<int>(temp_in_l_.size())) return;   // prepareToPlay underspec'd

    // Fill the temp buffers from the source. In loop mode: wrap so the
    // interpolator sees a continuous stream across the loop boundary.
    // In one-shot mode: any read past the end of the buffer is silence,
    // so the sample plays through once and then trails into nothing.
    int64_t srcPos = active_position_;
    const float* L = src->getReadPointer(0);
    const float* R = (srcChans > 1) ? src->getReadPointer(1) : L;
    for (int i = 0; i < numIn; ++i) {
        if (srcPos >= srcLen) {
            if (one_shot_mode_) {
                temp_in_l_[static_cast<size_t>(i)] = 0.0f;
                temp_in_r_[static_cast<size_t>(i)] = 0.0f;
                ++srcPos;
                continue;
            }
            srcPos = 0;   // loop wrap
        }
        temp_in_l_[static_cast<size_t>(i)] = L[srcPos];
        temp_in_r_[static_cast<size_t>(i)] = R[srcPos];
        ++srcPos;
    }

    // Resample (or memcpy at ratio == 1.0, which Lagrange handles itself).
    float* outL = info.buffer->getWritePointer(0, info.startSample);
    float* outR = (outChans > 1) ? info.buffer->getWritePointer(1, info.startSample)
                                 : outL;
    const int used = interp_l_.process(ratio, temp_in_l_.data(), outL, wanted);
    if (outR != outL) {
        interp_r_.process(ratio, temp_in_r_.data(), outR, wanted);
    }

    // ── Crossfade: mix outgoing buffer over the first N samples of out ──
    // The previous variation reads through a simple linear interpolator
    // (juce::LagrangeInterpolator is non-copyable so we can't snapshot its
    // warm state — but a 3 ms tail through linear interp is inaudible).
    if (crossfade_remaining_ > 0 && prev_idx_ != -2) {
        const juce::AudioBuffer<float>* psrc = nullptr;
        if (prev_idx_ == -1)
            psrc = &source_buf_;
        else if (prev_idx_ >= 0 && prev_idx_ < static_cast<int>(variation_bufs_.size()))
            psrc = &variation_bufs_[static_cast<size_t>(prev_idx_)];
        if (psrc != nullptr && psrc->getNumSamples() > 0) {
            const int pSrcLen   = psrc->getNumSamples();
            const int pSrcChans = psrc->getNumChannels();
            const float* pL = psrc->getReadPointer(0);
            const float* pR = (pSrcChans > 1) ? psrc->getReadPointer(1) : pL;

            // Read up to `n` destination samples from the previous buffer at
            // host SR, advancing prev_position_ + prev_sub_pos_ by `ratio`
            // per output sample. Linear interp between adjacent source
            // samples. Wraps at end of buffer (loop mode for the outgoing
            // tail — one-shot mode's natural end is too far away to matter
            // over 128 samples).
            const int n = std::min(wanted, crossfade_remaining_);
            int64_t pPos = prev_position_;
            double  pSub = prev_sub_pos_;
            for (int i = 0; i < n; ++i) {
                if (pPos >= pSrcLen) pPos -= pSrcLen;
                const int64_t pNext = (pPos + 1 < pSrcLen) ? (pPos + 1) : 0;
                const float fSub = static_cast<float>(pSub);
                const float oldL = pL[pPos] * (1.0f - fSub) + pL[pNext] * fSub;
                const float oldR = pR[pPos] * (1.0f - fSub) + pR[pNext] * fSub;

                const float t = static_cast<float>(kCrossfadeSamples - crossfade_remaining_) /
                                static_cast<float>(kCrossfadeSamples);
                outL[i] = outL[i] * t + oldL * (1.0f - t);
                if (outR != outL)
                    outR[i] = outR[i] * t + oldR * (1.0f - t);
                --crossfade_remaining_;

                pSub += ratio;
                while (pSub >= 1.0) { pSub -= 1.0; ++pPos; }
            }
            prev_position_ = pPos % pSrcLen;
            prev_sub_pos_  = pSub;
            if (crossfade_remaining_ == 0) {
                prev_idx_     = -2;
                prev_sub_pos_ = 0.0;
            }
        } else {
            // Outgoing buffer got dropped/cleared — bail.
            crossfade_remaining_ = 0;
            prev_idx_            = -2;
            prev_sub_pos_        = 0.0;
        }
    }

    // Advance source position by samples actually consumed (the interpolator
    // returns this exactly; using ratio*wanted would accumulate float error).
    active_position_ += used;
    if (one_shot_mode_) {
        // One-shot reached or passed the end → stop. The interpolator may
        // have a few samples of tail it's already mixed into this block
        // (those are at zero gain because temp_in_*_ went to silence past
        // end), so we don't need a fade-out; just latch playing=false.
        if (active_position_ >= srcLen) {
            active_position_ = 0;
            active_playing_  = false;
            interp_l_.reset();
            interp_r_.reset();
        }
    } else {
        active_position_ %= srcLen;
    }

    // Fade-in ramp (0 → 1 over kFadeSamples).
    if (fade_in_remaining_ > 0) {
        const int n = std::min(wanted, fade_in_remaining_);
        for (int i = 0; i < n; ++i) {
            const float g = static_cast<float>(kFadeSamples - fade_in_remaining_) /
                            static_cast<float>(kFadeSamples);
            outL[i] *= g;
            if (outR != outL) outR[i] *= g;
            --fade_in_remaining_;
        }
    }
    // Fade-out ramp (1 → 0 over kFadeSamples). When it completes, latch
    // active_playing_ false so subsequent blocks early-return silence.
    if (fade_out_remaining_ > 0) {
        const int n = std::min(wanted, fade_out_remaining_);
        for (int i = 0; i < n; ++i) {
            const float g = static_cast<float>(fade_out_remaining_) /
                            static_cast<float>(kFadeSamples);
            outL[i] *= g;
            if (outR != outL) outR[i] *= g;
            --fade_out_remaining_;
        }
        // Silence the remainder of the block past the fade endpoint.
        for (int i = n; i < wanted; ++i) {
            outL[i] = 0.0f;
            if (outR != outL) outR[i] = 0.0f;
        }
        if (fade_out_remaining_ == 0) active_playing_ = false;
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
        const auto models = resolveModelsDir();
        const auto t5  = models.getChildFile(kT5GemmaFile).getFullPathName();
        const auto dit = models.getChildFile(kDitFile).getFullPathName();
        const auto enc = models.getChildFile(kEncoderFile).getFullPathName();
        const auto dec = models.getChildFile(kDecoderFile).getFullPathName();
        for (const auto& p : { t5, dit, enc, dec }) {
            if (! juce::File(p).existsAsFile())
                throw std::runtime_error(
                    ("missing model file: " + p).toStdString());
        }
        auto p = std::make_unique<sa3::orch::Pipeline>(
            sa3::orch::load_pipeline(t5.toStdString(), dit.toStdString(),
                                     enc.toStdString(), dec.toStdString(),
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
