#include "PipelineLoader.h"

#include <exception>

#include "sa3_orchestrator.h"   // sa3::orch::Pipeline, load_pipeline

namespace sa3plugin {

namespace {
// Hardcoded for now — Phase 2 step 2 punts on path configuration UI.
// These paths are correct for the dev machine; in the shipped plugin we'll
// resolve weights from ~/Library/Application Support/SA3/models/ (or download
// from HuggingFace on first use, matching the Python optimized/mlx flow).
constexpr const char* kT5GemmaPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/t5gemma_f16.safetensors";
constexpr const char* kDitPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/dit_medium_f16.safetensors";
constexpr const char* kEncoderPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/same_l_encoder_f32.safetensors";
constexpr const char* kDecoderPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/same_l_decoder_f32.safetensors";
}  // namespace

PipelineLoader::PipelineLoader()
    : juce::Thread("SA3 Pipeline Loader") {}

PipelineLoader::~PipelineLoader() {
    // Wait up to 30s for an in-flight load to finish.  Once it finishes the
    // worker exits run() and the thread joins.  If the worker is mid-load
    // (libmlx.dylib reading safetensors) it can't be interrupted partway,
    // so we just wait — that's why we don't trigger the load at plugin
    // construction (plugin scan would be blocked here on teardown).
    stopThread(30000);
}

void PipelineLoader::requestLoad() {
    State expected = State::NotLoaded;
    if (state_.compare_exchange_strong(expected, State::Loading,
                                       std::memory_order_acq_rel)) {
        {
            std::lock_guard<std::mutex> lk(status_mutex_);
            status_ = "Loading models...";
        }
        startThread();
    }
    // else: Loading / Loaded / Error — no-op, request is a one-shot.
}

juce::String PipelineLoader::getStatus() const {
    std::lock_guard<std::mutex> lk(status_mutex_);
    return status_;
}

sa3::orch::Pipeline* PipelineLoader::getPipeline() const {
    return state_.load(std::memory_order_acquire) == State::Loaded
               ? pipeline_.get()
               : nullptr;
}

void PipelineLoader::run() {
    try {
        // Run at fp16 — same default as optimized/cpp/sa3_cli.
        auto p = std::make_unique<sa3::orch::Pipeline>(
            sa3::orch::load_pipeline(kT5GemmaPath, kDitPath,
                                     kEncoderPath, kDecoderPath,
                                     mlx::core::float16));
        pipeline_ = std::move(p);
        {
            std::lock_guard<std::mutex> lk(status_mutex_);
            status_ = "Loaded";
        }
        state_.store(State::Loaded, std::memory_order_release);
    }
    catch (const std::exception& ex) {
        {
            std::lock_guard<std::mutex> lk(status_mutex_);
            status_ = juce::String("Load failed: ") + ex.what();
        }
        state_.store(State::Error, std::memory_order_release);
    }
}

}  // namespace sa3plugin
