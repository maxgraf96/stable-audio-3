// SA3 end-to-end CLI in C++.
//
// Mirrors the surface of optimized/mlx/scripts/sa3_mlx.py for text-to-audio
// and audio-to-audio (single-shot init_audio + optional inpaint).
//
// usage:
//   sa3_cli --prompt "lofi house loop" --seconds 5 --seed 42 \
//           --t5gemma <path>.safetensors --dit <path>.safetensors \
//           --encoder <path>.safetensors --decoder <path>.safetensors \
//           --out out.wav
//
// a2a:    add `--init-audio in.wav` (uses --init-noise-level for σmax)
// inpaint: add `--init-audio in.wav --inpaint-range "S,E"` (seconds)

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include "sa3_orchestrator.h"

namespace mx = mlx::core;

namespace {

struct Args {
    std::string prompt;
    std::string negative_prompt;
    float       seconds          = 5.0f;
    int         steps            = 8;
    uint64_t    seed             = 42;
    float       cfg              = 1.0f;
    float       apg              = 1.0f;
    float       sigma_max        = 1.0f;
    std::string t5gemma_path;
    std::string dit_path;
    std::string encoder_path;
    std::string decoder_path;
    std::string out_path         = "out.wav";
    std::string dit_dtype        = "fp16";
    std::string init_audio_path;
    std::string inpaint_range_str;
    // --init-noise-level is an alias for --sigma-max when --init-audio is
    // set (matches sa3_mlx.py terminology); ignored otherwise.
    std::optional<float> init_noise_level;
};

void usage(const char* argv0) {
    std::cerr <<
        "usage: " << argv0 << " --prompt P [options]\n"
        "  --prompt P              text prompt (required; \"\" allowed = unconditional)\n"
        "  --negative-prompt P     CFG uncond branch (only used when --cfg != 1.0)\n"
        "  --seconds S             output length (default 5)\n"
        "  --steps N               pingpong sampler steps (default 8)\n"
        "  --seed N                random seed (default 42)\n"
        "  --cfg F                 classifier-free guidance scale (default 1.0 = off)\n"
        "  --apg F                 adaptive projected guidance [0..1] (default 1.0)\n"
        "  --sigma-max F           starting noise level σmax (default 1.0)\n"
        "  --init-audio PATH       16-bit PCM stereo WAV at 44.1 kHz — enables a2a\n"
        "  --init-noise-level F    σmax when --init-audio is set (alias for --sigma-max)\n"
        "  --inpaint-range S,E     inpaint seconds range (requires --init-audio)\n"
        "  --dit-dtype fp16|fp32   DiT compute dtype (default fp16)\n"
        "  --t5gemma PATH          path to t5gemma_f16.safetensors\n"
        "  --dit PATH              path to dit_medium_f16.safetensors\n"
        "  --encoder PATH          path to same_l_encoder_f32.safetensors\n"
        "  --decoder PATH          path to same_l_decoder_f32.safetensors\n"
        "  --out PATH              output WAV path (default out.wav)\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: " << a << " requires a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--prompt")           args.prompt          = next();
        else if (a == "--negative-prompt")  args.negative_prompt = next();
        else if (a == "--seconds")          args.seconds         = std::stof(next());
        else if (a == "--steps")            args.steps           = std::stoi(next());
        else if (a == "--seed")             args.seed            = static_cast<uint64_t>(std::stoull(next()));
        else if (a == "--cfg")              args.cfg             = std::stof(next());
        else if (a == "--apg")              args.apg             = std::stof(next());
        else if (a == "--sigma-max")        args.sigma_max       = std::stof(next());
        else if (a == "--init-noise-level") args.init_noise_level = std::stof(next());
        else if (a == "--init-audio")       args.init_audio_path = next();
        else if (a == "--inpaint-range")    args.inpaint_range_str = next();
        else if (a == "--dit-dtype")        args.dit_dtype       = next();
        else if (a == "--t5gemma")          args.t5gemma_path    = next();
        else if (a == "--dit")              args.dit_path        = next();
        else if (a == "--encoder")          args.encoder_path    = next();
        else if (a == "--decoder")          args.decoder_path    = next();
        else if (a == "--out")              args.out_path        = next();
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "error: unknown argument: " << a << "\n"; usage(argv[0]); return false; }
    }
    if (args.t5gemma_path.empty() || args.dit_path.empty() ||
        args.encoder_path.empty() || args.decoder_path.empty()) {
        std::cerr << "error: --t5gemma, --dit, --encoder, --decoder are required\n";
        return false;
    }
    if (!args.inpaint_range_str.empty() && args.init_audio_path.empty()) {
        std::cerr << "error: --inpaint-range requires --init-audio\n";
        return false;
    }
    // --init-noise-level overrides --sigma-max when both relate to a2a/inpaint.
    if (args.init_noise_level.has_value()) args.sigma_max = *args.init_noise_level;
    return true;
}

std::optional<std::pair<float, float>> parse_inpaint_range(const std::string& s) {
    if (s.empty()) return std::nullopt;
    const auto comma = s.find(',');
    if (comma == std::string::npos) {
        std::cerr << "error: --inpaint-range expects 'START,END' (got '" << s << "')\n";
        std::exit(2);
    }
    return std::make_pair(std::stof(s.substr(0, comma)), std::stof(s.substr(comma + 1)));
}

mx::Dtype parse_dtype(const std::string& s) {
    if (s == "fp16") return mx::float16;
    if (s == "fp32") return mx::float32;
    std::cerr << "error: --dit-dtype must be fp16 or fp32 (got " << s << ")\n";
    std::exit(2);
}

double elapsed_seconds(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double>(steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    const mx::Dtype dit_dtype = parse_dtype(args.dit_dtype);

    auto t_load_0 = std::chrono::steady_clock::now();
    auto pipe = sa3::orch::load_pipeline(
        args.t5gemma_path, args.dit_path, args.encoder_path, args.decoder_path,
        dit_dtype);
    std::cerr << "[sa3] models loaded in " << elapsed_seconds(t_load_0) << "s\n";

    // Optionally read init audio. read_wav_pcm16 returns planar (channels,
    // samples) fp32 in [-1, 1]; the InitAudio struct holds a non-owning view
    // into this buffer, which must outlive the generate() call.
    std::vector<float> init_buf;
    std::optional<sa3::orch::InitAudio> init_audio;
    if (!args.init_audio_path.empty()) {
        int sr = 0, ch = 0, n = 0;
        init_buf = sa3::orch::read_wav_pcm16(args.init_audio_path, sr, ch, n);
        if (sr != sa3::orch::SAMPLE_RATE) {
            std::cerr << "warning: init_audio sample rate " << sr
                      << " != " << sa3::orch::SAMPLE_RATE
                      << " — pipeline assumes 44.1 kHz; resample first.\n";
        }
        init_audio = sa3::orch::InitAudio{init_buf.data(), ch, n};
        std::cerr << "[sa3] init audio: " << args.init_audio_path
                  << " (" << ch << " ch × " << n << " samples @ " << sr << " Hz)\n";
    }

    const auto inpaint = parse_inpaint_range(args.inpaint_range_str);

    auto t_gen_0 = std::chrono::steady_clock::now();
    mx::array audio = pipe.generate(
        args.prompt, args.seconds, args.steps, args.seed,
        args.cfg, args.apg, args.negative_prompt, args.sigma_max,
        init_audio, inpaint);
    const double gen_s = elapsed_seconds(t_gen_0);
    std::cerr << "[sa3] generate in " << gen_s << "s ("
              << (args.seconds / gen_s) << "× realtime)\n";

    sa3::orch::save_wav_pcm16(args.out_path, audio, sa3::orch::SAMPLE_RATE);
    std::cerr << "[sa3] saved " << args.out_path << "\n";
    return 0;
}
