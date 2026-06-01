// AudioRing — fixed-length looping playback buffer for the morph instrument.
//
// Pure C++/std (no MLX): a [ch, L] float ring. The audio callback (real-time
// thread) READS a contiguous block per call, wrapping at the loop boundary, and
// never blocks. The generator thread WRITES freshly-decoded loops in, equal-
// power-crossfaded against what's already playing at the current play cursor so
// a new morph fades in without a click.
//
// Single-writer (generator) / single-reader (audio callback). A short mutex
// guards the buffer swap + cursor; the read path holds it only to snapshot.
//
// Channels-first convention: planar [ch][L] float32 in [-1, 1]. Port of
// realtime/runtime/audio_ring.py.
#pragma once

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace sa3 {
namespace rt {

// Equal-power (constant-energy) crossfade ramps of length n.
std::pair<std::vector<float>, std::vector<float>> equal_power(int n);

// Fold a finite clip's tail back over its head to make it loop seamlessly.
// in: planar [ch][L]; out: planar [ch][L-xfade]. If xfade<=0 or L<=2*xfade,
// returns a copy of the input unchanged (and out_L = L).
std::vector<std::vector<float>> make_loopable(
    const std::vector<std::vector<float>>& audio, int xfade);

class AudioRing {
public:
    // `length` is the DECODED clip length L. loop_xfade folds the tail over the
    // head, so the PLAYABLE loop is L - loop_xfade.
    AudioRing(int length, int channels = 2, int xfade = 1024, int loop_xfade = 0);

    int  length() const { return L_; }
    int  channels() const { return ch_; }
    long pos() const;
    bool has_audio() const;

    // Read the next n frames into `out` (planar [ch][n]), wrapping once at the
    // loop boundary. Real-time safe: one short lock to snapshot buffer + cursor.
    // `out` must be sized [ch][>=n]; only the first n per channel are written.
    void read(int n, std::vector<std::vector<float>>& out);

    // Install a new loop, equal-power-crossfaded against the current one over the
    // xfade frames starting at the current play cursor. `audio` is the full
    // decoded clip [ch][L]; folded to a seamless loop of length L_ first.
    // Generator thread only.
    void write_loop(const std::vector<std::vector<float>>& audio);

private:
    std::vector<std::vector<float>> fit(const std::vector<std::vector<float>>& audio) const;

    int loop_xfade_;
    int L_;
    int ch_;
    int xfade_;
    std::vector<std::vector<float>> buf_;   // [ch][L]
    long pos_ = 0;                          // play cursor (reader-owned)
    bool has_audio_ = false;
    std::vector<float> fade_in_, fade_out_;
    mutable std::mutex mutex_;
};

}  // namespace rt
}  // namespace sa3
