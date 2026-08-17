// End-to-end SA3 inference pipeline in C++:
//   prompt → T5Gemma → conditioner → DiT pingpong sampler → SAME-L decoder → WAV
//
// All five C++ ports (sa3_pipeline, samel, t5gemma, dit) wired together.
// Mirrors optimized/mlx/scripts/sa3_mlx.py for direct bit-exact comparison.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include <functional>
#include <optional>
#include <utility>
#include <variant>

#include "dit.h"
#include "dit_small.h"
#include "sa3_pipeline.h"
#include "samel_decoder.h"
#include "samel_encoder.h"
#include "sames_decoder.h"
#include "sames_encoder.h"
#include "t5gemma.h"

namespace sa3 {
namespace orch {

namespace mx = mlx::core;

constexpr int SAMPLE_RATE        = 44100;
constexpr int SAMPLES_PER_LATENT = 4096;   // PatchedPretransform 256 × SAME 16× expansion

// IO_CHANNELS and SAMPLES_PER_LATENT are identical across all three model
// kinds — verified at compile time so any future drift surfaces here.
static_assert(dit::IO_CHANNELS == dit_small::IO_CHANNELS,
              "DiT IO_CHANNELS must match across architectures");

// Which of the three SA3 model bundles a Pipeline holds. Determines which
// DiT/autoencoder variant is active and (downstream) which decode_chunked
// parameters to use. Both small variants share an architecture — they
// differ only in fine-tuning, so SMALL_MUSIC and SMALL_SFX use the same
// C++ code paths with different weights.
enum class ModelKind {
    MEDIUM       = 0,   // sa3-medium       (DiT-Medium  + SAME-L)
    SMALL_MUSIC  = 1,   // sa3-sm-music     (DiT-Small   + SAME-S)
    SMALL_SFX    = 2,   // sa3-sm-sfx       (DiT-Small   + SAME-S)
};

// ── Host memory policy ───────────────────────────────────────────────
//
// Every Mac this runs on has unified memory, so the model weights, the
// sampler's activations and MLX's buffer cache all come out of the same
// pool the OS and the host DAW are using. Overcommitting doesn't fail
// loudly — it swaps, and since a DiT step re-reads every weight, a run
// that fits in RAM in ~10 s takes minutes once it doesn't. So we measure
// the pool up front and size ourselves to it.
//
// Peak process footprints measured for a 5-candidate run of a 30 s clip
// (the longest the plugin allows) with the batched pipeline below. Only
// MEDIUM_PEAK_BYTES gates anything: SMALL_PEAK_BYTES fits inside the
// smallest Apple Silicon configuration Apple ships, so the small kinds are
// never refused — it's recorded here because the thresholds below only make
// sense next to the figure they're being compared against.
constexpr size_t MEDIUM_PEAK_BYTES = 8'600'000'000ull;
constexpr size_t SMALL_PEAK_BYTES  = 2'800'000'000ull;

// Headroom left for macOS, the host DAW and the WebView UI. The minimum is
// what a quiet machine needs to avoid swapping; the "good" figure assumes a
// loaded DAW project and other apps open.
constexpr size_t HOST_RESERVE_MIN_BYTES  =  2'000'000'000ull;
constexpr size_t HOST_RESERVE_GOOD_BYTES = 10'000'000'000ull;

// What we learned about this Mac, and what we intend to do about it.
struct MemoryProfile {
    size_t total_bytes       = 0;   // unified memory (sysctl hw.memsize)
    size_t working_set_bytes = 0;   // est. of Metal's max working set (~3/4)
    size_t cache_limit_bytes = 0;   // bound applied to MLX's buffer cache

    // sa3-medium fits at all / fits with room for a real DAW session.
    bool medium_supported   = false;
    bool medium_comfortable = false;

    // total_bytes came from SA3_SIMULATE_RAM_GB rather than the hardware.
    bool simulated = false;

    ModelKind default_kind = ModelKind::SMALL_MUSIC;
};

// Read the machine's unified memory and derive the policy above. Cheap,
// side-effect free, and deliberately touches neither MLX nor the GPU, so it
// is safe to call from a UI thread before the inference worker exists.
// Call apply_memory_policy() (on the MLX thread) to actually install it.
MemoryProfile probe_memory();

// Bound MLX's buffer cache to profile.cache_limit_bytes. MLX defaults it to
// ~95% of system RAM, so on a long run its pool of recycled buffers grows to
// several GB on top of the weights — measured 6.8 GB for a 21 s clip, which
// is what pushes a 16 GB machine into swap. Capping it costs no wall time.
void apply_memory_policy(const MemoryProfile& profile);

// Source audio for a2a / inpaint generation. Layout matches MLX/Python:
// planar channels-first, normalized fp32 in [-1, 1] at SAMPLE_RATE.
struct InitAudio {
    const float* data;   // pointer to (channels, samples) planar buffer
    int channels;        // expected to be 2 for SAME-L stereo
    int samples;         // per channel
};

// Polymorphic holders. Both alternatives in each variant have identical
// operator() signatures, so generate() can dispatch via std::visit without
// any architecture-specific code outside the helper methods below.
using DitVariant     = std::variant<dit::DiT,            dit_small::DiT>;
using EncoderVariant = std::variant<samel::SAMELEncoder, sames::SAMESEncoder>;
using DecoderVariant = std::variant<samel::SAMELDecoder, sames::SAMESDecoder>;

// Everything that is invariant across the candidates of one generation
// batch. Built once by Pipeline::begin_batch — which is the only phase that
// touches T5Gemma or the audio encoder — and then consumed by
// generate_from() for each candidate.
//
// This split exists for two reasons. The source encode and the prompt
// encode produce identical results for every candidate in a preset (all
// five share one source, and `free` shares one empty prompt), so doing them
// per candidate was pure repeated work. And once they only run in the
// prologue, T5Gemma and the encoder no longer need to stay resident while
// the DiT samples — which is where the memory goes.
struct BatchContext {
    int       T_lat;
    float     seconds;
    mx::array global_cond;                     // (1, GLOBAL_COND_DIM)
    std::optional<mx::array> init_latents;     // (1, IO_CHANNELS, T_lat) at dit_dtype

    // cross_attn (1, 257, COND_TOKEN_DIM) per distinct prompt string. Holds
    // negative prompts too — they go through the identical encode path.
    // Linear scan: a preset contributes at most six distinct strings.
    std::vector<std::pair<std::string, mx::array>> cross_attn_by_prompt;

    const mx::array& cross_attn_for(const std::string& prompt) const;
};

struct Pipeline {
    ModelKind           kind;          // determines which alternative is active in each variant
    // T5Gemma (fp16) + SentencePiece, and the audio encoder. Both are
    // prologue-only, so they're optional: begin_batch() faults them in and
    // drops them again before sampling starts. Measured a wash on wall time
    // even on a 48 GB machine (the reload is lazy and page-cache-warm, and
    // it amortizes over a whole batch) against 1.3 GB off the peak — so
    // there's no tier where holding them is worth it.
    std::optional<t5g::T5Gemma>    t5;
    LoadedConditioner   conditioner;   // padding_embedding + seconds_embedder (baked into the DiT safetensors)
    DitVariant          dit_var;       // velocity-prediction model
    std::optional<EncoderVariant>  encoder_var;   // audio → latents (for init_audio paths)
    DecoderVariant      decoder_var;   // latents → audio
    mx::Dtype           dit_dtype;     // dtype the DiT was loaded at — drives noise/sample dtype

    // Paths kept so the prologue models can be dropped and faulted back in.
    std::string         t5gemma_path;
    std::string         encoder_path;

    // Fault a prologue model in if it isn't currently held; no-op otherwise.
    // begin_batch drives these one at a time — the two are used in sequence,
    // so holding both costs peak memory for nothing.
    void ensure_text_encoder();
    void ensure_audio_encoder();
    void ensure_prologue_models();      // both, for callers outside begin_batch
    void release_text_encoder();
    void release_audio_encoder();
    void release_prologue_models();     // both, then return the pool to the OS

    // Architecture-agnostic dispatch helpers. Each visits the appropriate
    // variant and calls the underlying model's operator(); the call shapes
    // are identical across kinds so the caller doesn't need to care.
    mx::array dit_forward(
        const mx::array& x,
        const mx::array& t,
        const mx::array& cross_attn_cond_raw,
        const mx::array& global_cond_raw,
        const std::optional<mx::array>& local_add_cond = std::nullopt) const;

    mx::array encoder_forward(const mx::array& patches) const;

    // Calls decode_chunked from the active autoencoder's namespace. SAME-L
    // and SAME-S have different chunked-decode constants (sa3_mlx.py:
    // same-l → (128, 8), same-s → (8, 2)); the helper picks the right
    // pair based on `kind`.
    mx::array decoder_decode_chunked(const mx::array& latents) const;

    // ── Batch prologue ───────────────────────────────────────────────
    // Encode the source (once) and every distinct prompt (once each) for a
    // whole batch of candidates. `prompts` should list every candidate
    // prompt plus every negative prompt that will actually be encoded
    // (i.e. cfg != 1 and the negative prompt is non-empty); duplicates are
    // fine and are collapsed. Faults in T5Gemma + the encoder, and drops
    // them again on the way out.
    BatchContext begin_batch(
        const std::optional<InitAudio>& init_audio,
        float seconds,
        const std::vector<std::string>& prompts);

    // ── Per-candidate body ───────────────────────────────────────────
    // Sample one candidate against a prologue built by begin_batch. Touches
    // only the DiT and the decoder.
    //
    // Three modes selected by parameter combination:
    //   text-to-audio:   ctx.init_latents = nullopt, inpaint_range_seconds = nullopt
    //   audio-to-audio:  ctx.init_latents = some,    inpaint_range_seconds = nullopt
    //                    → init latents mixed with noise: lat*(1-σ) + noise*σ
    //   inpaint:         ctx.init_latents = some,    inpaint_range_seconds = some
    //                    → DiT sees inpaint mask + masked latents as local_add_cond;
    //                      unmasked latents are paste-back-exact every step.
    //
    // Returns (1, 2, T_audio) fp32 with T_audio = round(seconds * SAMPLE_RATE).
    mx::array generate_from(
        const BatchContext& ctx,
        const std::string& prompt,
        int   steps             = 8,
        uint64_t seed           = 42,
        float cfg               = 1.0f,
        float apg               = 1.0f,
        const std::string& negative_prompt = "",
        float sigma_max         = 1.0f,
        std::optional<std::pair<float, float>> inpaint_range_seconds = std::nullopt) const;

    // Single-shot convenience: begin_batch + generate_from for one prompt.
    // Equivalent to the pre-batching API and used by the CLI's non-variation
    // paths; prefer begin_batch/generate_from when running several
    // candidates off one source.
    mx::array generate(
        const std::string& prompt,
        float seconds,
        int   steps             = 8,
        uint64_t seed           = 42,
        float cfg               = 1.0f,
        float apg               = 1.0f,
        const std::string& negative_prompt = "",
        float sigma_max         = 1.0f,
        std::optional<InitAudio> init_audio = std::nullopt,
        std::optional<std::pair<float, float>> inpaint_range_seconds = std::nullopt);
};

// Load the full pipeline from four safetensors paths. The DiT safetensors
// also contains the conditioner under "cond.*" keys (extracted here).
// `kind` selects which DiT and which autoencoder code path to load — the
// caller is responsible for passing matching paths (e.g. dit_medium.* +
// same_l_*.* for MEDIUM, dit_sm-music_* / dit_sm-sfx_* + same_s_*.* for
// the small kinds). Defaulting to MEDIUM keeps the legacy call shape.
Pipeline load_pipeline(
    const std::string& t5gemma_path,
    const std::string& dit_path,
    const std::string& encoder_path,
    const std::string& decoder_path,
    mx::Dtype dit_dtype = mx::float16,
    ModelKind kind      = ModelKind::MEDIUM);

// ── Variation orchestration ──────────────────────────────────────────
// One generation candidate for the variations harness. Mirrors the
// CandidateSpec dataclass in sa3_variations.py: each candidate is a unique
// (seed, prompt, mode, noise, cfg, apg, inpaint_range) combination that
// produces one variation when fed through Pipeline::generate.
struct CandidateSpec {
    int           index = 0;
    std::string   mode;              // "a2a", "inpaint_middle_bar", "inpaint_first_2_bars"
    uint64_t      seed = 0;
    std::string   prompt;
    std::string   negative_prompt;
    std::string   steer;             // human-readable label (UI display)
    float         noise = 0.45f;     // init_noise_level / σmax
    float         cfg   = 1.0f;
    float         apg   = 1.0f;
    std::optional<std::pair<float, float>> inpaint_range_seconds;
};

// Output of one candidate run: spec + planar (channels=2, samples) fp32 audio
// in [-1, 1] at SAMPLE_RATE.
struct VariationOutput {
    CandidateSpec      spec;
    std::vector<float> audio;        // planar layout: audio[c * samples + t]
    int                channels = 2;
    int                samples  = 0;
};

// Five-shot "free variation" preset (sa3_variations.py:build_free_preset):
//   all a2a, empty prompt, cfg=1.0, varied seeds, single user-controlled noise.
std::vector<CandidateSpec> build_free_preset(
    float    duration_seconds,
    uint64_t seed_base   = 1234,
    float    noise_a2a   = 0.45f);

// Five-shot "preserve sound" preset (sa3_variations.py:build_app_preset):
//   3 × a2a (steered) + 2 × bar-aware inpaint.
//   Melodic only — drums/sfx/oneshot fall back to fractional masks.
//   When bpm is unset, the inpaint candidates use fractional masks
//   (duration * 0.35–0.65 and 0.0–0.5) as a graceful fallback.
std::vector<CandidateSpec> build_app_preset(
    float                    duration_seconds,
    uint64_t                 seed_base       = 1234,
    const std::string&       user_prompt     = "",
    std::optional<float>     bpm             = std::nullopt,
    const std::string&       key             = "",   // e.g. "A minor"
    int                      beats_per_bar   = 4,
    float                    noise_a2a       = 0.45f,
    float                    cfg_a2a         = 4.0f,
    float                    cfg_inpaint     = 4.0f,
    float                    apg             = 1.0f);

// Run a list of candidate specs against a single init-audio source.
//
// The source is encoded once for the whole batch and each distinct prompt
// once, rather than per candidate — for the 5-shot presets that is the
// difference between one encode and five. `on_candidate` fires as each
// candidate finishes so a UI can reveal results progressively instead of
// waiting for the whole batch; it runs on the calling thread.
//
// `steps` is the pingpong sampler step count (default 8, sa3_mlx.py's value).
void run_variations(
    Pipeline&                             pipe,
    const InitAudio&                      source_audio,
    const std::vector<CandidateSpec>&     specs,
    float                                 duration_seconds,
    int                                   steps,
    const std::function<void(const VariationOutput&)>& on_candidate);

// Collecting overload: returns one VariationOutput per spec, in order.
std::vector<VariationOutput> run_variations(
    Pipeline&                             pipe,
    const InitAudio&                      source_audio,
    const std::vector<CandidateSpec>&     specs,
    float                                 duration_seconds,
    int                                   steps = 8);

// Write 16-bit PCM stereo WAV at the given sample rate.
//   audio: (channels=2, T) fp32 in [-1, 1] — clipped on write.
void save_wav_pcm16(
    const std::string& path,
    const mx::array&   audio,
    int                sample_rate = SAMPLE_RATE);

// Read a 16-bit PCM WAV file. Caller must check sample rate / channels.
// Returns audio as planar (channels, samples) fp32 in [-1, 1]; out_channels
// and out_samples receive the layout. Throws on missing file, unsupported
// bit depth (must be 16), or unsupported encoding (must be PCM).
std::vector<float> read_wav_pcm16(
    const std::string& path,
    int&               out_sample_rate,
    int&               out_channels,
    int&               out_samples);

}  // namespace orch
}  // namespace sa3
