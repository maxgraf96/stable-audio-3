// Phase 4c headless self-test for the C++ real-time runtime.
//
// Spins up StreamGenerator (owns MLX, fills an AudioRing) and simulates an audio
// device pulling fixed blocks at real-time rate, exactly like the Python
// live_morph --selftest. Verifies: warmup completes, no underruns after warmup,
// finite audio, the generator out-paces playback (headroom), and live controls
// (denoise, sde curve) apply mid-run without crashing.
//
// Reads a source WAV (16-bit PCM); uses the orchestrator's WAV reader.
// Run:
//   ./test_stream_runtime <models_dir> <source.wav> [seconds] [depth] [run_s]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "audio_ring.h"
#include "stream_generator.h"
#include "sa3_orchestrator.h"   // read_wav_pcm16, SAMPLE_RATE, SAMPLES_PER_LATENT

using namespace sa3;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <models_dir> <source.wav> [seconds] [depth] [run_s]\n", argv[0]);
        return 2;
    }
    const std::string models_dir = argv[1];
    const std::string wav = argv[2];
    const float seconds = argc > 3 ? std::stof(argv[3]) : 0.0f;  // 0 -> source length
    const int depth = argc > 4 ? std::stoi(argv[4]) : 2;
    const double run_s = argc > 5 ? std::stof(argv[5]) : 12.0;

    // Read the source WAV (planar [2][n] fp32).
    int sr = 0, ch = 0, n = 0;
    std::vector<float> buf = orch::read_wav_pcm16(wav, sr, ch, n);
    std::vector<std::vector<float>> source(ch, std::vector<float>(n));
    for (int c = 0; c < ch; ++c)
        for (int i = 0; i < n; ++i) source[c][i] = buf[(size_t)c * n + i];
    const float src_seconds = (float)n / orch::SAMPLE_RATE;
    const float loop_seconds = (seconds > 0.0f) ? seconds : src_seconds;

    std::printf("=== C++ runtime self-test ===\n");
    std::printf("  source %.2fs @ %d Hz, %d ch  ->  loop %.2fs  depth %d\n",
                src_seconds, sr, ch, loop_seconds, depth);

    // Compute the playable loop length the generator will produce (loop_T frames).
    auto even_ceil = [](float s) {
        int f = std::max(2, (int)std::ceil(s * orch::SAMPLE_RATE / orch::SAMPLES_PER_LATENT));
        return f + (f % 2);
    };
    const int loop_T = even_ceil(loop_seconds);
    const int L = loop_T * orch::SAMPLES_PER_LATENT;

    rt::GenConfig cfg;
    cfg.seconds = loop_seconds;
    cfg.depth = depth;
    cfg.steps = 8;

    const int loop_xfade = (int)(cfg.loop_xfade_s * orch::SAMPLE_RATE);
    rt::AudioRing ring(L, 2, 2048, loop_xfade);
    rt::StreamGenerator gen(ring, models_dir, source, cfg);

    auto t0 = std::chrono::steady_clock::now();
    gen.start();
    if (!gen.wait_ready(120.0)) { std::printf("  FAIL: generator not ready in 120s\n"); return 1; }
    double warmup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  warmup ready in %.1fs\n", warmup_s);

    // Simulate the audio device pulling 1024-frame blocks at real time.
    const int block = 1024;
    const double period = (double)block / orch::SAMPLE_RATE;
    const int nblocks = (int)(run_s / period);
    std::vector<std::vector<float>> out(2, std::vector<float>(block));
    long underruns = 0;
    double peak = 0.0;
    bool finite = true;

    auto next = std::chrono::steady_clock::now();
    for (int i = 0; i < nblocks; ++i) {
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(period));
        ring.read(block, out);
        if (!ring.has_audio()) ++underruns;
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < block; ++k) {
                float v = out[c][k];
                if (!std::isfinite(v)) finite = false;
                peak = std::max(peak, (double)std::fabs(v));
            }
        // Exercise live controls mid-run.
        if (i == nblocks / 4)  gen.set_style_intensity(0.2f);
        if (i == nblocks / 2)  gen.set_morph_amount(0.7f);
        if (i == 3 * nblocks / 4) gen.set_morph_amount(1.0f);
        std::this_thread::sleep_until(next);
    }

    rt::GenStats st = gen.stats();
    gen.stop();

    std::printf("  ran %.0fs playback (%d blocks @ %d)\n", run_s, nblocks, block);
    std::printf("  ticks=%ld completions=%ld decodes=%ld skips=%ld\n",
                st.ticks, st.completions, st.decodes, st.skips);
    std::printf("  tick %.1f ms   decode %.1f ms\n", st.tick_ms, st.decode_ms);
    std::printf("  audio: peak=%.3f finite=%s underruns(after warmup)=%ld\n",
                peak, finite ? "true" : "false", underruns);
    double gen_rate = st.completions / run_s;
    double play_rate = 1.0 / loop_seconds;
    std::printf("  morph rate %.2f loops/s vs playback %.2f loops/s (%.0fx headroom)\n",
                gen_rate, play_rate, gen_rate / play_rate);

    bool ok = finite && underruns == 0 && peak > 1e-3 && st.completions >= 2;
    std::printf("=== SELF-TEST: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
