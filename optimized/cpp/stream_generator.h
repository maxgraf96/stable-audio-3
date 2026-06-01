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
#include <functional>
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
    // Live = sm-music DiT + SAME-S codec (fast ~30ms/step, weaker text-following);
    // Quality = medium DiT + SAME-L codec (~270ms/step, much stronger style
    // transfer — dubstep-vs-rock spectral cos 0.26 vs 0.46 for sm-music, and
    // cleaner peaks). Both share the streaming pipeline; only the model set + the
    // codec decode params differ.
    orch::Family family   = orch::Family::SmMusic;
    float    seconds      = 8.0f;
    int      steps        = 8;
    int      depth        = 2;
    float    denoise      = 0.3f;    // (unused) legacy base sigma; Amount->sigma now maps
                                     // the img2img init directly (see sigma_for_amount).
    float    scan_denoise = 0.85f;   // (unused) legacy playthrough sigma; superseded by
                                     // sigma_for_amount(Amount, Intensity).
    std::string prompt;
    uint64_t seed_base    = 1000;
    float    margin_s     = 0.0f;    // interior-windowing margin per side (0 = cyclic decode)
    float    loop_xfade_s = 0.15f;   // tail->head fold for the interior seam
    bool     evolve       = false;   // false = fixed seed (coherent morph)
    float    mse_skip     = 5e-4f;   // skip decode when consecutive latents barely move
    bool     cyclic       = true;    // seamless-loop decode (wrap-around context)
    // Classifier-free guidance strength applied WHEN a non-empty style prompt is
    // set. cfg=1 (no prompt) is source-preserving free variation; cfg>1 transfers
    // style. Measured sweet spot for this streaming SDE path is ~2.0: dubstep-vs-
    // rock spectral cos 0.46 (strong, distinct), peaks ~1.2 (the output limiter
    // in MorphEngine handles the residual). cfg=5 over-drove to peak ~2.5 =
    // clipped distortion where every prompt sounded the same. Only effective with
    // Character toward "explore" (the SDE re-noise path); Character="preserve"
    // re-anchors to source and ignores the prompt by design. Doubles the per-tick
    // forward (cond + uncond) only while a prompt is active.
    float    cfg_styled   = 2.0f;
    float    apg          = 1.0f;    // adaptive projected guidance (1=full)
    mx::Dtype dtype       = mx::float16;
};

struct GenStats {
    long ticks = 0, completions = 0, decodes = 0, skips = 0;
    double tick_ms = 0.0, decode_ms = 0.0;
    // Scan telemetry (sliding window over a track longer than the window). The
    // playhead's track position = scan_start + (ring.total_read() - chunk_base).
    long scan_start = 0;        // current chunk's source start (samples into the full track)
    long chunk_base = 0;        // current chunk's OUTPUT base (play position where its interior starts)
    int  window_samples = 0;    // window length (samples)
    long full_samples = 0;      // full track length (samples); <=window => not scanning
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
    void set_morph_amount(float v);     // Amount: 0=original loop, 1=styled target
    void set_style_intensity(float v);  // Character: how far the styled endpoint departs
    void set_cfg(float v);              // CFG strength used to build the styled target
    void set_shared_curve(const std::string& name, float value);   // velocity_scale
    void clear_shared_curve(const std::string& name);
    void set_prompt(const std::string& text);
    void set_evolve(bool on);

    GenStats stats() const;

private:
    void run();                            // worker thread entry
    void setup();                          // model load (on worker thread)
    void encode_prompt(const std::string& text);
    void encode_window(long start_sample);  // encode init_latents_ from track[start:start+window]
    mx::array encode_window_latent(long start_sample);   // ...returns the latent (scan: per-chunk source)
    void static_tick();                     // one iteration: looping morph (source <= one window)
    void scan_tick();                       // one iteration: forward-streaming playthrough
    void apply_scan_controls();             // re-encode prompt + velocity (Amount = per-window sigma)
    // Submit one clean windowed img2img (init-mix source to sigma, then text trajectory).
    void submit_window(const mx::array& source, float sigma);
    // Windowed decode of a sub-range [off, off+win] (latent frames) -> planar audio.
    std::vector<std::vector<float>> decode_window(const mx::array& lat, int off_frames, int win_frames);
    void submit_morph();                    // short-loop: clean img2img of the loop at sigma(Amount)
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
    float morph_amount_   = 0.5f;           // Amount (0..1): img2img sigma = original->styled
    float style_intensity_ = 0.6f;          // Intensity (0..1): caps Amount=1's sigma (departure)
    float cfg_styled_     = 2.0f;           // CFG strength (text guidance) when a prompt is set
    std::optional<float> velocity_scale_;   // nullopt = inactive (Motion)
    std::optional<std::string> new_prompt_;
    bool evolve_;
    GenStats stats_;

    // MLX/model state (worker thread only).
    std::unique_ptr<orch::Pipeline> pipe_models_;
    std::unique_ptr<stream::StreamPipeline> pipe_;
    stream::DiTForward dit_fwd_;             // bound DiT forward (model-agnostic)
    // Codec decode bound to the active family: (latents[1,256,T], chunk, ovl,
    // cyclic) -> patches [1,512,T*16]. Lets decode_crop stay family-agnostic.
    std::function<mx::array(const mx::array&, int, int, bool)> codec_decode_;
    int gen_T_ = 0;
    int crop_start_ = 0;       // interior-window start (samples)
    int loop_samples_ = 0;     // interior length played back (samples) == window length

    // Traveling window: when the source is longer than one window, slide the
    // window forward through the track as playback advances (a 1x restyled
    // playthrough). scanning_ gates it; scan_start_ is the current window's start
    // sample into the full src_; last_boundary_pos_ is the ring play cursor at the
    // last advance (a window of playback == one boundary).
    bool scanning_ = false;
    long scan_start_ = 0;
    long last_boundary_pos_ = 0;
    int chunk_ = 8, ovl_ = 2;  // codec decode params (SAME-S 8/2, SAME-L 128/2)
    uint64_t seed_ = 0;
    std::optional<mx::array> init_latents_;
    std::optional<mx::array> cross_;
    std::optional<mx::array> gcond_;
    std::optional<mx::array> neg_cross_;    // null (empty-prompt) conditioning for CFG
    std::optional<mx::array> secs_e_;
    std::optional<mx::array> last_latent_;   // static-loop mode: last finished latent
    bool prompt_active_ = false;            // non-empty style prompt -> enable CFG

    // Playthrough = streaming img2img. Each window is a clean img2img generation
    // (init-mix at sigma(Amount), text-conditioned trajectory) banked as a FINISHED
    // latent and windowed-decoded at the playhead. The pipe builds the NEXT window one
    // window ahead (next_pending_); at the overlap we crossfade cur->next windows.
    std::optional<mx::array> cur_latent_;    // current window: finished img2img latent
    std::optional<mx::array> next_latent_;   // next window: finished img2img latent
    bool next_pending_ = false;              // a next window is generating / ready
    long next_base_ = 0, next_scan_ = 0;     // incoming window's play-base / source-start
    long xfade_start_ = -1;                  // playhead where the cur->next crossfade began
};

}  // namespace rt
}  // namespace sa3
