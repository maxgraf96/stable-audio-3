// Single MLX-owning worker thread for the plugin.
//
// MLX binds default streams to the thread that first touches them — arrays
// created on one thread can't be evaluated from another without the
// runtime emitting "There is no Stream(gpu, N) in current thread". So the
// plugin can have ONE worker that loads the pipeline and runs every
// generate(); not a separate loader + engine.
//
// This class owns that thread.  It accepts two kinds of work:
//   - requestLoad():     load all four safetensors files into a Pipeline
//                        (one-shot, idempotent).
//   - requestGenerate(): take a GenerateRequest (preset + params + raw
//                        source-audio bytes), decode the audio via JUCE's
//                        AudioFormatManager, resample to 44.1 kHz stereo,
//                        build the 5 CandidateSpecs and call
//                        sa3::orch::run_variations().  Resulting WAV buffers
//                        are stored in memory keyed by variation index,
//                        ready for the WebView's resource provider to serve
//                        as `juce://sa3.local/variations/N.wav`.
//
// One in-flight generate at a time; if Running, requestGenerate returns
// false and the caller surfaces the error to the UI.  The load and any
// generate calls share the same condition variable / worker loop.
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
    // Preset selector — "free" or "app". Anything else is a user error.
    std::string preset;
    float       seconds         = 5.0f;    // target output length
    float       noise           = 0.45f;   // σmax for a2a slots
    std::optional<float> bpm;              // app preset: bar-aware masks
    std::string key;                       // app preset: e.g. "A minor"
    std::string user_prompt;               // app preset: appended to base prompt
    int         beats_per_bar   = 4;
    float       cfg_a2a         = 4.0f;
    float       cfg_inpaint     = 4.0f;
    float       apg             = 1.0f;
    uint64_t    seed            = 1234;
    int         steps           = 8;

    // Raw bytes of the dropped source file (WAV/AIFF/FLAC/Ogg supported by
    // JUCE's AudioFormatManager::registerBasicFormats()). Owned by the
    // request; copied across the native bridge from JS.
    juce::MemoryBlock audio_bytes;
};

class VariationsEngine : private juce::Thread {
public:
    // Lifecycle is two-axis: load-phase + busy-flag.
    //
    //   load-phase: NotLoaded → Loading → Loaded   (one-way; or Error)
    //   busy-flag:  Idle      ↔ Running            (only meaningful once Loaded)
    //
    // We expose both as separate accessors so the UI can show "loading
    // models…" until Loaded, then "ready" or "generating N/5…".
    enum class LoadPhase { NotLoaded, Loading, Loaded, Error };

    // Called when generation completes. The argument is one of:
    //   { ok: true,  urls: ["juce://sa3.local/variations/0.wav", …] }
    //   { ok: false, error: "..." }
    using CompletionFn = std::function<void(juce::var)>;

    VariationsEngine();
    ~VariationsEngine() override;

    VariationsEngine(const VariationsEngine&)            = delete;
    VariationsEngine& operator=(const VariationsEngine&) = delete;

    // Trigger the one-shot pipeline load.  Idempotent: subsequent calls
    // (or calls after a successful load) are no-ops.  Returns immediately.
    void requestLoad();

    // Returns false if a job is already running OR the pipeline isn't loaded
    // yet — caller should surface an error to the UI.
    bool requestGenerate(GenerateRequest req, CompletionFn completion);

    // Snapshot of the current state — safe to call from any thread.
    LoadPhase    getLoadPhase() const { return load_phase_.load(std::memory_order_acquire); }
    bool         isBusy()       const { return busy_.load(std::memory_order_acquire); }
    juce::String getStatus()    const;

    // Result accessors used by the editor's resource provider for
    // `juce://sa3.local/variations/N.wav`. Each call returns a fresh copy so
    // we don't have to hold the mutex while the WebView writes the response.
    int                    getVariationCount() const;
    std::vector<std::byte> getVariationWav(int idx) const;

private:
    void run() override;

    void writeStatus(juce::String s);
    void clearStore();

    void doLoad();                                // executes on worker thread
    juce::var doGenerate(GenerateRequest req);    // executes on worker thread

    // One-shot load gate: set true by requestLoad(), worker consumes it.
    std::atomic<bool> load_pending_{false};

    // Generate request slot — submission writes here while holding the
    // mutex, then notifies the thread.
    std::mutex                          request_mutex_;
    std::optional<GenerateRequest>      pending_request_;
    CompletionFn                        pending_completion_;

    // Pipeline owned and used exclusively on the worker thread once loaded.
    // Lifetime: created in doLoad(), destroyed when the engine destructs
    // (during stopThread the worker exits the run loop, then the unique_ptr
    // goes out of scope cleanly).
    std::unique_ptr<sa3::orch::Pipeline> pipeline_;

    // Results — one WAV per variation index, in submission order.  Mutex
    // protects both the vector and the status string.
    mutable std::mutex                  store_mutex_;
    std::vector<std::vector<std::byte>> store_;
    juce::String                        status_{"Not loaded"};

    std::atomic<LoadPhase> load_phase_{LoadPhase::NotLoaded};
    std::atomic<bool>      busy_{false};
    std::atomic<bool>      shutdown_{false};

    JUCE_LEAK_DETECTOR(VariationsEngine)
};

}  // namespace sa3plugin
