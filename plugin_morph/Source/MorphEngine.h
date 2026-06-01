// MorphEngine — real-time style-morph engine for the SA3 Morph plugin.
//
// Owns the C++ streaming stack (optimized/cpp): an AudioRing the audio thread
// reads from, and a StreamGenerator that owns the MLX worker thread (loads the
// sm-music DiT + SAME-S codec, continuously ticks the ring buffer, decodes, and
// fills the AudioRing). This class is the JUCE-facing wrapper:
//
//   - juce::AudioSource: getNextAudioBlock resamples the 44.1k ring into the
//     host's block at its sample rate, with fade-in/out (audio-thread safe via
//     try-lock; plays silence rather than blocking).
//   - Message-thread control API: load source (async), start/stop the morph,
//     and live knobs (denoise / source-blend / velocity / evolve / prompt) that
//     forward to the StreamGenerator's lock-protected control state.
//   - A lightweight loader thread handles the blocking source decode + model
//     load + StreamGenerator startup so the message thread never blocks.
//
// MLX thread-affinity: ALL MLX work happens on StreamGenerator's own worker
// thread (it loads the pipeline there too), so MorphEngine itself touches no
// MLX — it only reads the AudioRing (plain float) and forwards control values.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <JuceHeader.h>

#include "audio_ring.h"
#include "stream_generator.h"

namespace sa3morph {

class MorphEngine : public juce::AudioSource {
public:
    enum class Phase { Idle, Loading, Running, Error };

    // Live state pushed to the UI (via the editor's emitEvent timer).
    struct MorphState {
        int    phase     = 0;      // Phase as int
        bool   playing   = false;
        double progress  = 0.0;    // 0..1 play cursor into the loop
        double loopSecs  = 0.0;
        float  denoise   = 0.5f;
        float  sourceBlend = 1.0f;
        float  velocity  = 1.0f;
        bool   evolve    = false;
        double tickMs    = 0.0;
        double decodeMs  = 0.0;
        long   completions = 0;
    };

    using CompletionFn = std::function<void(juce::var)>;

    MorphEngine();
    ~MorphEngine() override;

    MorphEngine(const MorphEngine&)            = delete;
    MorphEngine& operator=(const MorphEngine&) = delete;

    // ── Source load + morph lifecycle (message thread) ───────────────
    // Decode raw audio bytes, build the ring + generator, and start morphing.
    // Async: completion fires { ok, duration, peaks } when the first loop is
    // audible, or { ok:false, error }. Returns false if a load is already in
    // flight.
    bool requestLoadSource(juce::MemoryBlock audioBytes, int peaks_n,
                           CompletionFn completion);

    void startPlayback();
    void stopPlayback();
    bool isPlaying() const { return playing_.load(std::memory_order_acquire); }

    Phase getPhase() const { return phase_.load(std::memory_order_acquire); }
    juce::String getStatus() const;
    std::vector<float> getSourcePeaks() const;

    // ── Live controls (message thread → generator) ───────────────────
    void setDenoise(float v);       // "Amount": 0=original loop .. 1=styled target
    void setSourceBlend(float v);   // "Intensity": how extreme the styled endpoint is
    void setVelocity(float v);      // "Motion": velocity_scale (1=neutral)
    void setCfg(float v);           // "CFG": prompt-guidance strength for the styled target
    void setEvolve(bool on);
    void setPrompt(const std::string& text);

    // Source window: morph only the LAST n seconds of a dropped file (so an entire
    // track loops/morphs on its recent window, not its whole length). Applied on
    // the NEXT load; <=0 means use the whole file. The editor reloads to apply.
    void setWindowSeconds(double s);
    double windowSeconds() const { return window_seconds_.load(std::memory_order_acquire); }

    // Diffusion steps per window: fewer = faster render (linearly) but rougher; the
    // schedule is fixed when the generator is built, so this applies on the NEXT load.
    // Returns true if the value changed (caller reloads to apply).
    bool setSteps(int n);
    int steps() const { return steps_.load(std::memory_order_acquire); }

    // Quality mode: medium DiT + SAME-L (stronger style transfer, ~6x slower per
    // tick) vs Live: sm-music + SAME-S (fast). The model family is fixed when the
    // generator is built, so this takes effect on the NEXT load — the editor
    // reloads the current source after toggling. Returns true if the value
    // changed (caller should reload to apply).
    bool setQuality(bool on);
    bool isQuality() const { return quality_.load(std::memory_order_acquire); }

    MorphState getMorphState() const;

    // ── AudioSource (audio thread) ───────────────────────────────────
    void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock(const juce::AudioSourceChannelInfo& info) override;

private:
    void loaderThreadMain(juce::MemoryBlock bytes, int peaks_n, CompletionFn completion);
    static juce::AudioBuffer<float> decodeToStereo44k(const juce::MemoryBlock& bytes);
    static std::vector<float> extractPeaks(const juce::AudioBuffer<float>& buf, int n, int len = -1);
    std::string resolveModelsDir() const;
    void writeStatus(juce::String s);

    // Streaming stack. ring_ outlives generator_ (generator writes into it).
    std::unique_ptr<sa3::rt::AudioRing>        ring_;
    std::unique_ptr<sa3::rt::StreamGenerator>  generator_;

    std::atomic<Phase> phase_{Phase::Idle};
    std::atomic<bool>  playing_{false};
    std::atomic<bool>  load_in_flight_{false};
    std::atomic<bool>  quality_{true};      // default Quality(medium) — much stronger
                                            // style transfer; false=Live(sm-music, fast)
    std::atomic<double> window_seconds_{10.0};  // morph the last N s of a dropped file
    std::atomic<int>    steps_{8};              // diffusion steps per window (next load)
                                                // (<=0 = whole file); applied on load

    std::thread loader_;        // one-shot loader (joined on next load / dtor)

    // Loop geometry (set after load).
    std::atomic<int>   loop_len_{0};        // ring length in 44.1k samples
    std::atomic<double> loop_secs_{0.0};

    // Live control mirror (so getMorphState can report knob values).
    mutable std::mutex ctrl_mutex_;
    float denoise_ = 0.5f, source_blend_ = 0.6f, velocity_ = 1.0f, cfg_styled_ = 2.0f;
    bool  evolve_ = false;

    std::vector<float> source_peaks_;
    mutable std::mutex peaks_mutex_;

    juce::String status_{"Idle"};
    mutable std::mutex status_mutex_;

    // Audio-thread resampling state.
    double host_sample_rate_ = 44100.0;
    juce::LagrangeInterpolator interp_l_, interp_r_;
    std::vector<float> temp_in_l_, temp_in_r_;
    std::vector<std::vector<float>> ring_read_;   // [2][block] scratch for ring.read
    int fade_in_remaining_ = 0;
    static constexpr int kFadeSamples = 256;
    static constexpr int kSampleRate  = 44100;
};

}  // namespace sa3morph
