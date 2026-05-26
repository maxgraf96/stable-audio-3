// End-to-end SA3 inference pipeline in C++:
//   prompt → T5Gemma → conditioner → DiT pingpong sampler → SAME-L decoder → WAV
//
// All five C++ ports (sa3_pipeline, samel, t5gemma, dit) wired together.
// Mirrors optimized/mlx/scripts/sa3_mlx.py for direct bit-exact comparison.
#pragma once

#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include <optional>
#include <utility>

#include "dit.h"
#include "sa3_pipeline.h"
#include "samel_decoder.h"
#include "samel_encoder.h"
#include "t5gemma.h"

namespace sa3 {
namespace orch {

namespace mx = mlx::core;

constexpr int SAMPLE_RATE        = 44100;
constexpr int SAMPLES_PER_LATENT = 4096;   // PatchedPretransform 256 × SAME 16× expansion

// Source audio for a2a / inpaint generation. Layout matches MLX/Python:
// planar channels-first, normalized fp32 in [-1, 1] at SAMPLE_RATE.
struct InitAudio {
    const float* data;   // pointer to (channels, samples) planar buffer
    int channels;        // expected to be 2 for SAME-L stereo
    int samples;         // per channel
};

struct Pipeline {
    t5g::T5Gemma        t5;            // text encoder (fp16) + SentencePiece
    LoadedConditioner   conditioner;   // padding_embedding + seconds_embedder (baked into the DiT safetensors)
    dit::DiT            dit;           // velocity-prediction model (fp16 by default)
    samel::SAMELEncoder encoder;       // audio → SAME-L latents (fp32, used for init_audio)
    samel::SAMELDecoder decoder;       // SAME-L latents → audio (fp32)
    mx::Dtype           dit_dtype;     // dtype the DiT was loaded at — drives noise/sample dtype

    // Generate audio.
    //
    // Three modes selected by parameter combination:
    //   text-to-audio:   init_audio = nullopt, inpaint_range_seconds = nullopt
    //   audio-to-audio:  init_audio = some,   inpaint_range_seconds = nullopt
    //                    → init latents mixed with noise: lat*(1-σ) + noise*σ
    //   inpaint:         init_audio = some,   inpaint_range_seconds = some
    //                    → DiT sees inpaint mask + masked latents as local_add_cond;
    //                      unmasked latents are paste-back-exact every step.
    //
    // Returns (1, 2, T_audio) fp32 with T_audio = round(seconds * SAMPLE_RATE).
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
        std::optional<std::pair<float, float>> inpaint_range_seconds = std::nullopt) const;
};

// Load the full pipeline from four safetensors paths. The DiT safetensors
// also contains the conditioner under "cond.*" keys (extracted here).
Pipeline load_pipeline(
    const std::string& t5gemma_path,
    const std::string& dit_path,
    const std::string& samel_encoder_path,
    const std::string& samel_decoder_path,
    mx::Dtype dit_dtype = mx::float16);

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
