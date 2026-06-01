// StreamGenerator — owns all MLX work for the live morph instrument (C++).
//
// MLX binds compute streams to the creating thread, so EVERY MLX call (model
// loads, ticks, decode) happens on this one worker thread. The audio callback
// never touches MLX; it only reads the AudioRing this thread fills.
//
// Loop each iteration:
//   1. apply pending control changes (denoise, shared curves, prompt) at a tick
//      boundary, so mutations serialize with the tick.
//   2. keep the ring full: submit a fresh morph slot for each free slot.
//   3. tick() one denoising step across all in-flight slots.
//   4. on a finished latent: MSE-skip check, else cyclic-decode + interior crop
//      -> write_loop into AudioRing.
//
// Port of realtime/runtime/generator.py (sm-music + SAME-S only). Loads its own
// orch::Pipeline (SmMusic family) for T5Gemma + conditioner + dit_sm + SAME-S,
// then drives a stream::StreamPipeline.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "audio_ring.h"
#include "sa3_orchestrator.h"
#include "stream.h"
#include "stream_ode.h"

namespace sa3 {
namespace rt {

namespace mx = mlx::core;

struct GenConfig {
    float    seconds      = 8.0f;
    int      steps        = 8;
    int      depth        = 2;
    float    denoise      = 0.5f;
    float    denoise_slew = 0.04f;   // max change in effective denoise per submission
    std::string prompt;
    uint64_t seed_base    = 1000;
    float    margin_s     = 0.0f;    // interior-windowing margin per side (0 = cyclic decode)
    float    loop_xfade_s = 0.15f;   // tail->head fold for the interior seam
    bool     evolve       = false;   // false = fixed seed (coherent morph)
    float    mse_skip     = 5e-4f;   // skip decode when consecutive latents barely move
    bool     cyclic       = true;    // seamless-loop decode (wrap-around context)
    mx::Dtype dtype       = mx::float16;
};

struct GenStats {
    long ticks = 0, completions = 0, decodes = 0, skips = 0;
    double tick_ms = 0.0, decode_ms = 0.0;
};

class StreamGenerator {
public:
    // models_dir holds the sm-music + SAME-S safetensors (and t5gemma).
    // source: planar [ch][samples] fp32 at SAMPLE_RATE, in [-1, 1].
    StreamGenerator(AudioRing& ring, std::string models_dir,
                    std::vector<std::vector<float>> source, GenConfig cfg);
    ~StreamGenerator();

    void start();                          // spawns the worker thread
    void stop();                           // signals + joins
    bool wait_ready(double timeout_s);     // blocks until first loop is audible

    // Control API (any thread).
    void set_denoise(float v);
    void set_shared_curve(const std::string& name, float value);   // scalar curve
    void clear_shared_curve(const std::string& name);
    void set_prompt(const std::string& text);
    void set_evolve(bool on);

    GenStats stats() const;

private:
    void run();                            // worker thread entry
    void setup();                          // model load (on worker thread)
    void encode_prompt(const std::string& text);
    void submit_morph();
    void apply_shared_curves();
    std::vector<std::vector<float>> decode_crop(const mx::array& lat);

    AudioRing& ring_;
    std::string models_dir_;
    std::vector<std::vector<float>> src_;
    GenConfig cfg_;

    std::thread thread_;
    std::atomic<bool> running_{true};
    std::atomic<bool> ready_{false};

    // Control state (guarded by ctrl_mutex_).
    mutable std::mutex ctrl_mutex_;
    float denoise_target_;
    std::optional<float> sde_curve_;        // nullopt = inactive
    std::optional<float> velocity_scale_;
    std::optional<std::string> new_prompt_;
    bool evolve_;
    GenStats stats_;

    // MLX/model state (worker thread only).
    std::unique_ptr<orch::Pipeline> pipe_models_;
    std::unique_ptr<stream::StreamPipeline> pipe_;
    int gen_T_ = 0;
    int crop_start_ = 0;       // interior-window start (samples)
    int loop_samples_ = 0;     // interior length played back (samples)
    int chunk_ = 8, ovl_ = 2;  // SAME-S decode params
    uint64_t seed_ = 0;
    float denoise_eff_ = 0.5f;
    std::optional<mx::array> init_latents_;
    std::optional<mx::array> cross_;
    std::optional<mx::array> gcond_;
    std::optional<mx::array> secs_e_;
    std::optional<mx::array> last_latent_;
};

}  // namespace rt
}  // namespace sa3
