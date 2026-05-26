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

#include "dit.h"
#include "sa3_pipeline.h"
#include "samel_decoder.h"
#include "t5gemma.h"

namespace sa3 {
namespace orch {

namespace mx = mlx::core;

constexpr int SAMPLE_RATE        = 44100;
constexpr int SAMPLES_PER_LATENT = 4096;   // PatchedPretransform 256 × SAME 16× expansion

struct Pipeline {
    t5g::T5Gemma      t5;            // text encoder (fp16) + SentencePiece
    LoadedConditioner conditioner;   // padding_embedding + seconds_embedder (loaded with the DiT)
    dit::DiT          dit;           // velocity-prediction model (fp16 by default)
    samel::SAMELDecoder decoder;     // SAME-L latent → audio decoder (fp32)
    mx::Dtype         dit_dtype;     // dtype the DiT was loaded at — drives noise/sample dtype

    // Generate audio from a prompt.
    // Returns (1, 2, T_audio) fp32 with T_audio = round(seconds * SAMPLE_RATE).
    mx::array generate(
        const std::string& prompt,
        float seconds,
        int   steps             = 8,
        uint64_t seed           = 42,
        float cfg               = 1.0f,
        float apg               = 1.0f,
        const std::string& negative_prompt = "",
        float sigma_max         = 1.0f) const;
};

// Load the full pipeline from three safetensors paths. The DiT safetensors
// also contains the conditioner under "cond.*" keys (extracted here).
Pipeline load_pipeline(
    const std::string& t5gemma_path,
    const std::string& dit_path,
    const std::string& samel_decoder_path,
    mx::Dtype dit_dtype = mx::float16);

// Write 16-bit PCM stereo WAV at the given sample rate.
//   audio: (channels=2, T) fp32 in [-1, 1] — clipped on write.
void save_wav_pcm16(
    const std::string& path,
    const mx::array&   audio,
    int                sample_rate = SAMPLE_RATE);

}  // namespace orch
}  // namespace sa3
