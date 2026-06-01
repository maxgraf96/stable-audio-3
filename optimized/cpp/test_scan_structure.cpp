// Structural test for the forward-streaming playthrough (scan mode).
//
// Pushes a SRC_S-second linear chirp through a WIN_S-second window (so scanning_
// is on) and captures the output stream at 1x real-time. The chirp lets us map an
// output time back to the SOURCE time it came from (by instantaneous frequency),
// so a Python pass can see whether the output actually advances through the track
// or repeats / stalls / gaps. Saves /tmp/scan_source.wav and /tmp/scan_output.wav.
//
//   ./test_scan_structure <models_dir> [amount=0] [prompt] [src_s=60] [win_s=10] [quality=0]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <mlx/mlx.h>

#include "audio_ring.h"
#include "stream_generator.h"
#include "sa3_orchestrator.h"

using namespace sa3;
namespace mx = mlx::core;

static mx::array to_mx(const std::vector<std::vector<float>>& planar) {
    const int C = (int)planar.size();
    const int N = (int)planar[0].size();
    std::vector<float> flat((size_t)C * N);
    for (int c = 0; c < C; ++c)
        std::copy(planar[c].begin(), planar[c].end(), flat.begin() + (size_t)c * N);
    return mx::array(flat.data(), mx::Shape{C, N}, mx::float32);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <models_dir> [amount=0] [prompt] [src_s=60] [win_s=10] [quality=0]\n", argv[0]);
        return 2;
    }
    const std::string models_dir = argv[1];
    const float amount = argc > 2 ? std::stof(argv[2]) : 0.0f;
    const std::string prompt = argc > 3 ? std::string(argv[3]) : "";
    const float src_s = argc > 4 ? std::stof(argv[4]) : 60.0f;
    const float win_s = argc > 5 ? std::stof(argv[5]) : 10.0f;
    const bool quality = argc > 6 ? (std::stoi(argv[6]) != 0) : false;
    const bool rich = argc > 7 ? (std::stoi(argv[7]) != 0) : false;  // add rhythm/texture
    const float cfgv = argc > 8 ? std::stof(argv[8]) : 2.0f;         // CFG strength

    const int SR = orch::SAMPLE_RATE;
    const int n = (int)(src_s * SR);

    // Stereo linear chirp f0->f1 over the whole source (so output freq -> source time)
    // + slow 1 Hz tremolo. `rich` adds a 2 Hz kick + steady bass + hi-hat noise so a
    // style prompt has rhythm/texture to grab (the chirp alone is a poor style target).
    std::vector<std::vector<float>> source(2, std::vector<float>(n));
    const double f0 = 120.0, f1 = 3000.0;
    unsigned int rng = 12345;
    auto frand = [&]() { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 9) / 8388608.0f - 1.0f; };
    for (int i = 0; i < n; ++i) {
        double t = (double)i / SR;
        double phase = 2.0 * M_PI * (f0 * t + 0.5 * (f1 - f0) / src_s * t * t);
        double amp = 0.35 * (0.85 + 0.15 * std::sin(2.0 * M_PI * 1.0 * t));
        double s = amp * std::sin(phase);
        if (rich) {
            double tb = std::fmod(t, 0.5);                       // 2 Hz grid
            double kick = std::exp(-tb * 30.0) * std::sin(2.0 * M_PI * 60.0 * tb);  // kick
            double bass = 0.18 * std::sin(2.0 * M_PI * 90.0 * t);                   // bass
            double hat = (std::fmod(t, 0.25) < 0.02) ? 0.15 * frand() : 0.0;        // hat
            s = 0.5 * s + 0.4 * kick + bass + hat;
        }
        source[0][i] = (float)s;
        source[1][i] = (float)s;
    }
    orch::save_wav_pcm16("/tmp/scan_source.wav", to_mx(source), SR);

    auto even_ceil = [](float s) {
        int f = std::max(2, (int)std::ceil(s * orch::SAMPLE_RATE / orch::SAMPLES_PER_LATENT));
        return f + (f % 2);
    };
    const int L = even_ceil(win_s) * orch::SAMPLES_PER_LATENT;

    rt::GenConfig cfg;
    cfg.seconds = win_s;
    cfg.depth = 2;
    cfg.steps = 8;
    cfg.family = quality ? orch::Family::Medium : orch::Family::SmMusic;
    const int loop_xfade = (int)(cfg.loop_xfade_s * SR);
    rt::AudioRing ring(L, 2, 2048, loop_xfade);
    rt::StreamGenerator gen(ring, models_dir, source, cfg);
    if (!prompt.empty()) gen.set_prompt(prompt);
    gen.set_morph_amount(amount);
    gen.set_cfg(cfgv);

    std::printf("=== scan structure test ===\n");
    std::printf("  src %.0fs  window %.0fs  amount %.2f  cfg %.1f  prompt='%s'  model=%s\n",
                src_s, win_s, amount, cfgv, prompt.c_str(), quality ? "Quality" : "Live");

    auto t0 = std::chrono::steady_clock::now();
    gen.start();
    if (!gen.wait_ready(180.0)) { std::printf("  FAIL: not ready in 180s\n"); return 1; }
    std::printf("  warmup ready in %.1fs\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    const double run_s = src_s + 4.0;
    const int block = 1024;
    const double period = (double)block / SR;
    const int nblocks = (int)(run_s / period);
    std::vector<std::vector<float>> out(2, std::vector<float>(block));
    std::vector<std::vector<float>> cap(2);
    cap[0].reserve((size_t)nblocks * block);
    cap[1].reserve((size_t)nblocks * block);

    auto next = std::chrono::steady_clock::now();
    for (int i = 0; i < nblocks; ++i) {
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(period));
        ring.read(block, out);
        for (int c = 0; c < 2; ++c)
            cap[c].insert(cap[c].end(), out[c].begin(), out[c].begin() + block);
        std::this_thread::sleep_until(next);
    }
    rt::GenStats st = gen.stats();
    gen.stop();

    // Raw level of the captured stream (pre-limiter — the plugin tanh-limits later).
    double pk = 0.0, sq = 0.0; long clip = 0, ns = 0;
    for (int c = 0; c < 2; ++c)
        for (float v : cap[c]) { pk = std::max(pk, (double)std::fabs(v)); sq += (double)v*v; ++ns; if (std::fabs(v) > 0.99f) ++clip; }
    std::printf("  RAW level: peak=%.3f  rms=%.3f  clipping(|x|>0.99)=%.2f%%\n",
                pk, std::sqrt(sq / std::max(1L, ns)), 100.0 * clip / std::max(1L, ns));

    orch::save_wav_pcm16("/tmp/scan_output.wav", to_mx(cap), SR);
    std::printf("  captured %.1fs  ticks=%ld completions=%ld decodes=%ld  tick=%.1fms decode=%.1fms\n",
                (double)cap[0].size() / SR, st.ticks, st.completions, st.decodes,
                st.tick_ms, st.decode_ms);
    std::printf("  saved /tmp/scan_source.wav  /tmp/scan_output.wav\n");
    return 0;
}
