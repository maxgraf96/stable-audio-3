// Apple Silicon backend: the in-process C++ MLX orchestrator (optimized/cpp).
//
// This is the original inference path, moved out of VariationsEngine.cpp
// unchanged when the backend seam was introduced. It is compiled only when
// APPLE is set, because sa3_orchestrator pulls in MLX, which is Metal-only.
//
// MLX binds default streams to the thread that first touches them, so every
// method here must run on the engine's single worker thread — the engine
// guarantees that by only ever calling the backend from doLoad /
// doSwitchModel / doGenerate.
#include "InferenceBackend.h"

#include <cstdlib>
#include <stdexcept>

#include <juce_core/juce_core.h>

#include "sa3_orchestrator.h"

namespace sa3plugin {

static_assert(kSampleRate == sa3::orch::SAMPLE_RATE,
              "kSampleRate must mirror sa3::orch::SAMPLE_RATE");

// Keep our public enum in lock-step with the orchestrator's so the
// static_cast in load() below is well-defined.
static_assert(static_cast<int>(ModelKind::MEDIUM)
              == static_cast<int>(sa3::orch::ModelKind::MEDIUM),
              "ModelKind::MEDIUM must mirror sa3::orch::ModelKind::MEDIUM");
static_assert(static_cast<int>(ModelKind::SMALL_MUSIC)
              == static_cast<int>(sa3::orch::ModelKind::SMALL_MUSIC),
              "ModelKind::SMALL_MUSIC must mirror sa3::orch::ModelKind::SMALL_MUSIC");
static_assert(static_cast<int>(ModelKind::SMALL_SFX)
              == static_cast<int>(sa3::orch::ModelKind::SMALL_SFX),
              "ModelKind::SMALL_SFX must mirror sa3::orch::ModelKind::SMALL_SFX");

namespace {

// All three model kinds share the same T5Gemma encoder; only the DiT +
// autoencoder pair differs. End-users keep all candidate files in the
// same models directory and the engine picks the right four at load time.
constexpr const char* kT5GemmaFile = "t5gemma_f16.safetensors";

struct ModelFiles {
    const char* dit;
    const char* encoder;
    const char* decoder;
};

constexpr ModelFiles modelFilesFor(ModelKind kind) {
    switch (kind) {
        case ModelKind::SMALL_MUSIC:
            return {
                "dit_sm-music_f16.safetensors",
                "same_s_encoder_f32.safetensors",
                "same_s_decoder_f32.safetensors",
            };
        case ModelKind::SMALL_SFX:
            return {
                "dit_sm-sfx_f16.safetensors",
                "same_s_encoder_f32.safetensors",
                "same_s_decoder_f32.safetensors",
            };
        case ModelKind::MEDIUM:
        default:
            return {
                "dit_medium_f16.safetensors",
                "same_l_encoder_f32.safetensors",
                "same_l_decoder_f32.safetensors",
            };
    }
}

// Where the four safetensors files live for the currently-running plugin.
// Resolution order:
//   1. $SA3_MODELS_DIR — dev override; lets us run the build-tree binary
//      against a freshly downloaded model snapshot without rebuilding.
//   2. <bundle>/Contents/Resources/models — self-contained bundle layout
//      (CMake's SA3_VENDOR_MODELS=ON path; off by default because each
//      bundle would otherwise weigh ~7 GB).
//   3. ~/Library/Application Support/SA3 Variations/models — production
//      default. End-users download the models once and drop them here;
//      all three plugin formats share the single copy.
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
    // juce::userApplicationDataDirectory on macOS is `~/Library`, *not*
    // `~/Library/Application Support` — Apple bundles that into the path
    // and JUCE doesn't expose it as its own special location.
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Application Support")
        .getChildFile("SA3 Variations")
        .getChildFile("models");
}

class MlxBackend final : public InferenceBackend {
public:
    MemoryInfo probeMemory() override
    {
        const auto profile = sa3::orch::probe_memory();
        MemoryInfo info;
        info.totalGB           = profile.total_bytes / 1e9;
        info.workingSetGB      = profile.working_set_bytes / 1e9;
        info.mediumSupported   = profile.medium_supported;
        info.mediumComfortable = profile.medium_comfortable;
        info.simulated         = profile.simulated;
        info.defaultKind       = static_cast<ModelKind>(profile.default_kind);
        return info;
    }

    void load(ModelKind kind, const MemoryInfo& mem) override
    {
        // Refuse rather than thrash. On a machine that can't hold sa3-medium the
        // run doesn't fail — it swaps, and a generation that should take seconds
        // takes minutes with no indication of why. A clear error is kinder.
        if (kind == ModelKind::MEDIUM && ! mem.mediumSupported) {
            throw std::runtime_error(
                "SA3 Medium needs about 11 GB of unified memory; this Mac has "
                + juce::String(mem.totalGB, 1).toStdString()
                + " GB. Use SA3 Small Music or SA3 Small SFX instead — they need "
                  "about 3 GB and run faster.");
        }
        const auto models = resolveModelsDir();
        const auto files = modelFilesFor(kind);
        const auto t5  = models.getChildFile(kT5GemmaFile).getFullPathName();
        const auto dit = models.getChildFile(files.dit).getFullPathName();
        const auto enc = models.getChildFile(files.encoder).getFullPathName();
        const auto dec = models.getChildFile(files.decoder).getFullPathName();
        for (const auto& p : { t5, dit, enc, dec }) {
            if (! juce::File(p).existsAsFile())
                throw std::runtime_error(
                    ("missing model file: " + p).toStdString());
        }
        pipeline_ = std::make_unique<sa3::orch::Pipeline>(
            sa3::orch::load_pipeline(
                t5.toStdString(), dit.toStdString(),
                enc.toStdString(), dec.toStdString(),
                mlx::core::float16,
                static_cast<sa3::orch::ModelKind>(kind)));
    }

    void unload() override { pipeline_.reset(); }
    bool isLoaded() const override { return pipeline_ != nullptr; }

    void runVariations(
        const float* init_planar,
        int channels,
        int samples,
        const GenerateRequest& req,
        float seconds,
        const std::function<void(const BackendVariation&)>& on_candidate) override
    {
        if (! pipeline_) throw std::runtime_error("pipeline not loaded");

        sa3::orch::InitAudio init_audio{ init_planar, channels, samples };

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

        // One call for the whole batch: the source is encoded once rather
        // than once per candidate, and the callback still lands each result
        // as it finishes so the UI fills slots progressively.
        sa3::orch::run_variations(
            *pipeline_, init_audio, specs, seconds, req.steps,
            [&](const sa3::orch::VariationOutput& v) {
                BackendVariation out;
                out.channels = v.channels;
                out.samples  = v.samples;
                out.steer    = v.spec.steer;
                out.mode     = v.spec.mode;
                out.audio.assign(
                    v.audio.data(),
                    v.audio.data() + static_cast<size_t>(v.channels) * v.samples);
                on_candidate(out);
            });
    }

private:
    std::unique_ptr<sa3::orch::Pipeline> pipeline_;
};

}  // namespace

std::unique_ptr<InferenceBackend> InferenceBackend::create()
{
    return std::make_unique<MlxBackend>();
}

}  // namespace sa3plugin
