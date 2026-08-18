// The seam between the engine and whatever actually runs SA3.
//
// VariationsEngine owns audio decode, playback, peaks, and the worker-thread
// job queue — none of which care how a variation gets generated. Everything
// that *does* care lives behind InferenceBackend, which has two impls:
//
//   MlxBackend.cpp    (APPLE)     in-process C++ MLX orchestrator, optimized/cpp
//   WorkerBackend.cpp (otherwise) a resident Python process running the torch
//                                 pipeline, spoken to over stdio
//
// The MLX orchestrator is Metal-only and can't be built off Apple Silicon,
// which is the whole reason this seam exists. Types the two impls share with
// the engine live here rather than in VariationsEngine.h so a backend .cpp
// can include just this header and stay clear of JUCE audio internals.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sa3plugin {

// SA3 is a 44.1 kHz model end to end. MlxBackend static_asserts this against
// sa3::orch::SAMPLE_RATE so the two can never drift apart silently.
constexpr int kSampleRate = 44100;

// Which of the three SA3 model bundles to run. MlxBackend static_casts these
// to sa3::orch::ModelKind (same integer values, asserted); WorkerBackend maps
// them to the Python runner's --dit names.
enum class ModelKind {
    MEDIUM      = 0,    // sa3-medium      (DiT-Medium + SAME-L)
    SMALL_MUSIC = 1,    // sa3-sm-music    (DiT-Small  + SAME-S)
    SMALL_SFX   = 2,    // sa3-sm-sfx      (DiT-Small  + SAME-S)
};

inline const char* modelDisplayName(ModelKind kind) {
    switch (kind) {
        case ModelKind::SMALL_MUSIC: return "SA3-sm-music";
        case ModelKind::SMALL_SFX:   return "SA3-sm-sfx";
        case ModelKind::MEDIUM:
        default:                     return "SA3-medium";
    }
}

// What the backend learned about this machine's memory, surfaced to the UI so
// it can pick a sensible default model and warn before someone loads one that
// won't fit. On Apple this is unified memory (mirrors sa3::orch::MemoryProfile);
// on a discrete-GPU box it describes VRAM.
struct MemoryInfo {
    double    totalGB           = 0.0;
    double    workingSetGB      = 0.0;
    bool      mediumSupported   = false;   // sa3-medium fits at all
    bool      mediumComfortable = false;   // ...with room for a real session
    bool      simulated         = false;   // SA3_SIMULATE_RAM_GB in effect
    ModelKind defaultKind       = ModelKind::SMALL_MUSIC;
};

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

// One finished candidate. Backend-agnostic mirror of
// sa3::orch::VariationOutput — planar fp32, channel-major.
struct BackendVariation {
    std::vector<float> audio;       // channels * samples, channel-major
    int                channels = 2;
    int                samples  = 0;
    std::string        steer;       // per-candidate musical direction (UI label)
    std::string        mode;        // "a2a" | "inpaint"
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // Progress text for the UI's status line, called from whichever thread is
    // inside load()/runVariations() — in practice the engine's worker thread.
    //
    // This exists because a backend can spend a long time somewhere the engine
    // can't see. The Python worker keeps one pipeline sized to a specific loop
    // length, so the first generate after a differently-sized source is dropped
    // rebuilds it (~12 s) *inside* runVariations. Without this the UI sits on
    // "Generating 1/5..." and the model load reads as a very slow generation.
    using StatusFn = std::function<void(const std::string&)>;
    void setStatusCallback(StatusFn fn) { status_ = std::move(fn); }

    // Called once at engine construction, before any load. Must not throw —
    // a backend that can't probe should return conservative defaults.
    virtual MemoryInfo probeMemory() = 0;

    // Bring up the model. Throws std::runtime_error with a user-facing
    // message on any missing file, insufficient memory, or load failure.
    virtual void load(ModelKind kind, const MemoryInfo& mem) = 0;

    // Tear down and free. Must be safe to call when not loaded.
    virtual void unload() = 0;
    virtual bool isLoaded() const = 0;

    // Generate a batch from init audio. `init_planar` is channel-major fp32
    // of `channels * samples`, owned by the caller for the duration of the
    // call. `on_candidate` fires once per finished variation, in order, on
    // the calling thread — the engine uses that to fill UI slots
    // progressively rather than making the user wait for all five.
    //
    // Each backend builds its own candidate specs from `req.preset`, since
    // the preset definitions live next to the inference code (sa3::orch on
    // Apple, sa3_variations.py under the worker).
    virtual void runVariations(
        const float* init_planar,
        int channels,
        int samples,
        const GenerateRequest& req,
        float seconds,
        const std::function<void(const BackendVariation&)>& on_candidate) = 0;

    // Builds the backend this platform was compiled for.
    static std::unique_ptr<InferenceBackend> create();

protected:
    void reportStatus(const std::string& text) const { if (status_) status_(text); }

private:
    StatusFn status_;
};

}  // namespace sa3plugin
