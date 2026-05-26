// Background-thread loader for sa3::orch::Pipeline.
//
// Loading the medium DiT, T5Gemma, conditioner and SAME-L decoder takes
// roughly 1–2 seconds from disk (6.5 GB of safetensors), so we can't block
// the audio or UI thread on it. PipelineLoader spins up a juce::Thread on
// first request, exposes thread-safe state queries, and hands out the loaded
// Pipeline once ready.
//
// Headers stay sa3_orchestrator-free via forward declarations — only the
// .cpp drags in MLX/SA3 includes, so the plugin's other translation units
// don't recompile when the SA3 internals change.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include <JuceHeader.h>

namespace sa3 { namespace orch { struct Pipeline; } }

namespace sa3plugin {

class PipelineLoader : private juce::Thread {
public:
    enum class State {
        NotLoaded,
        Loading,
        Loaded,
        Error,
    };

    PipelineLoader();
    ~PipelineLoader() override;

    PipelineLoader(const PipelineLoader&)            = delete;
    PipelineLoader& operator=(const PipelineLoader&) = delete;

    // Kick off a background load.  Idempotent — no-op on subsequent calls or
    // once the load has completed.  Returns immediately.
    void requestLoad();

    // Thread-safe state query (UI poll target).
    State getState() const { return state_.load(std::memory_order_acquire); }

    // Human-readable status string ("Not loaded", "Loading models…", "Loaded",
    // or "Load failed: <reason>"). Cheap to call from a timer.
    juce::String getStatus() const;

    // Returns the loaded pipeline pointer once getState() == Loaded; nullptr
    // otherwise. The pointer is stable for the lifetime of this PipelineLoader.
    sa3::orch::Pipeline* getPipeline() const;

private:
    void run() override;   // juce::Thread entry point

    std::atomic<State> state_{State::NotLoaded};

    // Owned exclusively by the worker thread until state_ transitions to
    // Loaded; after that the pointer becomes immutable until ~PipelineLoader
    // joins the worker.
    std::unique_ptr<sa3::orch::Pipeline> pipeline_;

    // Protects the human-readable status_ string. Reads (from a UI timer)
    // are rare and the lock is contention-free in practice.
    mutable std::mutex status_mutex_;
    juce::String status_{"Not loaded"};

    JUCE_LEAK_DETECTOR(PipelineLoader)
};

}  // namespace sa3plugin
