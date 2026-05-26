// Single MLX-owning worker thread + audio playback engine.
//
// MLX binds default streams to the thread that first touches them — arrays
// created on one thread can't be evaluated from another without the
// runtime emitting "There is no Stream(gpu, N) in current thread". So the
// plugin has ONE worker that loads the pipeline and runs every generate();
// not a separate loader + engine.
//
// The engine also acts as the plugin's audio source. It owns:
//   - One stereo AudioBuffer<float> for the user's source loop (decoded
//     on uploadSource).
//   - Five stereo AudioBuffer<float> for the generated variations
//     (decoded from the in-memory WAV after each candidate is generated).
//   - Cached peak envelopes for source and each variation (so the
//     WebView doesn't have to re-decode them in JS).
//
// Playback is driven by Processor::processBlock pulling
// getNextAudioBlock() — one buffer is "active" at a time (idx -1 = source,
// 0..4 = slot, -2 = none). play / pause / seek mutate the active idx and
// position atomically; the audio thread reads under a short mutex.
//
// Threading:
//   - Worker thread: load pipeline, decode source, generate, decode
//     variations, write peaks.
//   - Audio thread:  read active buffer + position via play_mutex_.
//   - Message thread: submits jobs, polls status / play state, sets
//     active idx via play().
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <JuceHeader.h>

namespace sa3 { namespace orch { struct Pipeline; } }

namespace sa3plugin {

struct GenerateRequest {
    std::string preset;
    float       seconds       = 5.0f;
    float       noise         = 0.45f;
    std::optional<float> bpm;
    std::string key;
    std::string user_prompt;
    int         beats_per_bar = 4;
    float       cfg_a2a       = 4.0f;
    float       cfg_inpaint   = 4.0f;
    float       apg           = 1.0f;
    uint64_t    seed          = 1234;
    int         steps         = 8;
};

// Snapshot of which buffer is active + how far through it the audio thread
// is. Returned by getPlayState() — consumed by the UI poll.
struct PlayState {
    int    idx       = -2;        // -2 = idle, -1 = source, 0..4 = variation
    bool   playing   = false;     // false ⇒ paused or stopped
    double progress  = 0.0;       // 0..1 into the active buffer
    double duration  = 0.0;       // seconds, 0 if no active buffer
    bool   oneShot   = false;     // playback mode (loop = false, one-shot = true)
};

class VariationsEngine : private juce::Thread,
                         public  juce::AudioSource
{
public:
    enum class LoadPhase { NotLoaded, Loading, Loaded, Error };

    // Generate completion result is shaped like:
    //   { ok: true,  slots: [{ steer, mode, peaks: [floats], duration }] }
    //   { ok: false, error: "..." }
    // uploadSource completion result:
    //   { ok: true,  duration, peaks: [floats] }
    //   { ok: false, error: "..." }
    using CompletionFn = std::function<void(juce::var)>;

    VariationsEngine();
    ~VariationsEngine() override;

    VariationsEngine(const VariationsEngine&)            = delete;
    VariationsEngine& operator=(const VariationsEngine&) = delete;

    // ── Pipeline load ────────────────────────────────────────────────
    void requestLoad();
    LoadPhase    getLoadPhase() const { return load_phase_.load(std::memory_order_acquire); }
    bool         isBusy()       const { return busy_.load(std::memory_order_acquire); }
    juce::String getStatus()    const;

    // ── Source / generate ────────────────────────────────────────────
    // Submit decoded source audio to the worker.  The completion fires
    // when the worker has finished decoding + computing peaks (after
    // which getSourcePeaks and play(-1) start working).  Returns false
    // if the worker is busy with a generate.
    bool requestUploadSource(juce::MemoryBlock audioBytes, int peaks_n,
                             CompletionFn completion);

    // Returns false if a job is already running OR no source is loaded
    // OR the pipeline isn't ready.
    bool requestGenerate(GenerateRequest req, int peaks_n, CompletionFn completion);

    // ── Peak accessors (cheap; mutex-protected copy) ─────────────────
    std::vector<float> getSourcePeaks() const;
    std::vector<float> getVariationPeaks(int idx) const;
    int                getVariationCount() const;

    // ── Playback control (cheap, message-thread-safe) ────────────────
    void  play(int idx);              // -1 source, 0..4 slot, -2 stop
    void  pausePlayback();
    void  resumePlayback();
    void  stopPlayback();
    void  seek(double fraction);      // 0..1 into active buffer
    PlayState getPlayState() const;

    // One-shot vs loop mode. Affects two behaviours:
    //   - getNextAudioBlock: one-shot stops at end of buffer instead of
    //     wrapping (active_playing_ flips to false on natural end).
    //   - play(idx) on a different idx: one-shot resets position to 0
    //     instead of preserving the fraction (so cycling variations of a
    //     drum hit always plays from the attack).
    void setOneShotMode(bool oneShot);
    bool isOneShotMode() const;

    // ── AudioSource (called on the audio thread) ─────────────────────
    void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock(const juce::AudioSourceChannelInfo& info) override;

private:
    void run() override;

    // ── Worker-thread jobs ───────────────────────────────────────────
    void doLoad();
    juce::var doUploadSource(juce::MemoryBlock bytes, int peaks_n);
    juce::var doGenerate(GenerateRequest req, int peaks_n);

    void writeStatus(juce::String s);

    // Decode raw audio bytes (WAV/AIFF/FLAC/Ogg) into stereo fp32 at 44.1k.
    // Resamples and mixes/duplicates channels to fit (channels=2).
    static juce::AudioBuffer<float> decodeToStereo44k(const juce::MemoryBlock& bytes);
    // Mono peak envelope (max(|L|, |R|) per window). Returns n peaks in [0..1].
    static std::vector<float> extractPeaks(const juce::AudioBuffer<float>& buf, int n);

    // ── Worker job queue ─────────────────────────────────────────────
    enum class JobKind { LoadPipeline, UploadSource, Generate };
    struct Job {
        JobKind kind;
        // UploadSource payload
        juce::MemoryBlock                source_bytes;
        int                              peaks_n = 0;
        // Generate payload
        std::optional<GenerateRequest>   gen_req;
        // Common
        CompletionFn                     completion;
    };
    std::mutex             job_mutex_;
    std::optional<Job>     pending_job_;

    // ── Pipeline (worker-thread owned once loaded) ───────────────────
    std::unique_ptr<sa3::orch::Pipeline> pipeline_;

    // ── Decoded audio + peaks ────────────────────────────────────────
    // play_mutex_ protects source_buf_, variation_bufs_, active_idx_,
    // active_position_, active_playing_. The audio thread takes the lock
    // briefly per block (no allocations inside the critical section).
    mutable std::mutex                       play_mutex_;
    juce::AudioBuffer<float>                 source_buf_;        // (2, N) or empty
    std::vector<juce::AudioBuffer<float>>    variation_bufs_;    // each (2, N)
    int                                      active_idx_      = -2;   // -2 idle
    int64_t                                  active_position_ = 0;
    bool                                     active_playing_  = false;

    // Cached peaks (computed alongside buffer decode). Separate mutex so a
    // peak readback doesn't block the audio thread.
    mutable std::mutex                       peaks_mutex_;
    std::vector<float>                       source_peaks_;
    std::vector<std::vector<float>>          variation_peaks_;

    // Resampling + click suppression (audio-thread state, protected by
    // play_mutex_).
    double                                   host_sample_rate_ = 44100.0;
    juce::LagrangeInterpolator               interp_l_, interp_r_;
    std::vector<float>                       temp_in_l_, temp_in_r_;
    int                                      fade_in_remaining_  = 0;
    int                                      fade_out_remaining_ = 0;
    bool                                     one_shot_mode_      = false;

    // Crossfade state for variation switching while playing. When a play()
    // call swaps buffers mid-stream we keep the outgoing buffer's source
    // position + a fractional sub-sample around for ~3 ms and mix
    // old*(1-t) + new*t into the output, then drop the old side.
    // The old stream uses linear interpolation (the crossfade tail is too
    // brief for cubic to matter, and juce::LagrangeInterpolator state can't
    // be copied so we can't preserve its warm history).
    std::vector<float>                       scratch_l_, scratch_r_;
    int                                      prev_idx_           = -2;
    int64_t                                  prev_position_      = 0;
    double                                   prev_sub_pos_       = 0.0;
    int                                      crossfade_remaining_ = 0;

    // Status string (uses peaks_mutex_).
    juce::String                             status_{"Not loaded"};

    std::atomic<LoadPhase> load_phase_{LoadPhase::NotLoaded};
    std::atomic<bool>      busy_{false};
    std::atomic<bool>      shutdown_{false};
    std::atomic<bool>      load_pending_{false};

    JUCE_LEAK_DETECTOR(VariationsEngine)
};

}  // namespace sa3plugin
