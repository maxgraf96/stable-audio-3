// End-to-end sm-music + SAME-S pipeline test (C++).
//
// Loads the sm-music family via load_pipeline(..., Family::SmMusic), runs a full
// text-to-audio generate (T5Gemma -> conditioner -> dit_sm -> pingpong -> SAME-S
// decode -> unpatch), and writes a WAV. Compare against the Python sm-music
// pipeline (scripts/sa3_mlx.py --dit sm-music --decoder same-s) at the same
// prompt/seed/seconds/steps.
//
// Usage:
//   ./test_sm_e2e <models_dir> <out.wav> "<prompt>" <seconds> <steps> <seed>
#include <cstdio>
#include <string>

#include <mlx/mlx.h>

#include "sa3_orchestrator.h"

namespace mx = mlx::core;
using namespace sa3;

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage: %s <models_dir> <out.wav> <prompt> <seconds> <steps> <seed>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string out = argv[2];
    const std::string prompt = argv[3];
    const float seconds = std::stof(argv[4]);
    const int   steps   = std::stoi(argv[5]);
    const uint64_t seed = std::stoull(argv[6]);

    auto p = [&](const char* n) { return dir + "/" + n; };
    try {
        orch::Pipeline pipe = orch::load_pipeline(
            p("t5gemma_f16.safetensors"),
            p("dit_sm-music_f16.safetensors"),
            p("same_s_encoder_f32.safetensors"),
            p("same_s_decoder_f32.safetensors"),
            orch::Family::SmMusic);

        mx::array audio = pipe.generate(
            prompt, seconds, steps, seed,
            /*cfg=*/1.0f, /*apg=*/1.0f, /*neg=*/"", /*sigma_max=*/1.0f,
            /*init_audio=*/std::nullopt, /*inpaint=*/std::nullopt);

        mx::array a32 = mx::astype(audio, mx::float32);
        mx::eval(a32);
        mx::array mean = mx::mean(a32);
        mx::array amax = mx::max(mx::abs(a32));
        mx::eval(mean, amax);
        std::printf("audio shape=[%d, %d] mean=%.6f absmax=%.6f\n",
                    a32.shape()[0], a32.shape()[1], mean.item<float>(), amax.item<float>());

        orch::save_wav_pcm16(out, audio, orch::SAMPLE_RATE);
        std::fprintf(stderr, "wrote %s\n", out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
