// Non-Apple backend: a resident Python process running the torch pipeline.
//
// The C++ MLX orchestrator that MlxBackend uses is Metal-only, so off Apple
// Silicon the plugin drives the same Python inference stack the studio uses
// (sa3_pipeline_torch.Pipeline). The worker is spawned once and kept warm —
// loading sa3-medium costs ~16 s, and paying that per click would make the
// app unusable — then fed newline-delimited JSON on stdin, replying with one
// JSON object per line on stdout. One `candidate` event per finished
// variation keeps the progressive slot-filling the Apple path gives us.
//
// STAGE 1 (current): process plumbing is stubbed. The app builds, launches,
// renders its UI, and plays an uploaded source; generate reports that
// inference isn't wired yet. STAGE 2 fills in spawn/handshake/generate below.
#include "InferenceBackend.h"

#include <stdexcept>

#include <juce_core/juce_core.h>

namespace sa3plugin {

namespace {

class WorkerBackend final : public InferenceBackend {
public:
    MemoryInfo probeMemory() override
    {
        // STAGE 2: the worker reports real device memory (torch.cuda
        // total_memory, or system RAM when it falls back to CPU) in its
        // `ready` event, which is strictly better than guessing from here —
        // the plugin has no CUDA headers and shouldn't grow any.
        //
        // Until then, report system RAM and stay conservative about medium:
        // claiming support we haven't verified would push users into a
        // multi-minute swap instead of a clear "pick a smaller model".
        MemoryInfo info;
        const auto ramGB =
            static_cast<double>(juce::SystemStats::getMemorySizeInMegabytes()) / 1024.0;
        info.totalGB           = ramGB;
        info.workingSetGB      = ramGB;
        info.mediumSupported   = false;
        info.mediumComfortable = false;
        info.simulated         = false;
        info.defaultKind       = ModelKind::SMALL_MUSIC;
        return info;
    }

    void load(ModelKind, const MemoryInfo&) override
    {
        // STAGE 2: spawn `<repo>/.venv/Scripts/python.exe plugin/scripts/sa3_worker.py`
        // via juce::ChildProcess, send {"cmd":"load","model":...}, block until
        // the `ready` event, and surface any `error` event as this throw.
        throw std::runtime_error(
            "Inference is not wired up on this platform yet. The Windows build "
            "currently runs the UI and audio playback only; generation needs the "
            "Python worker (stage 2). Use sa3_studio.py in the meantime.");
    }

    void unload() override { loaded_ = false; }
    bool isLoaded() const override { return loaded_; }

    void runVariations(
        const float*,
        int,
        int,
        const GenerateRequest&,
        float,
        const std::function<void(const BackendVariation&)>&) override
    {
        // STAGE 2: write the source to a temp WAV, send
        // {"cmd":"generate",...}, and pump stdout — firing on_candidate for
        // every `candidate` event until `done`.
        throw std::runtime_error("inference backend not available on this platform yet");
    }

private:
    bool loaded_ = false;
};

}  // namespace

std::unique_ptr<InferenceBackend> InferenceBackend::create()
{
    return std::make_unique<WorkerBackend>();
}

}  // namespace sa3plugin
