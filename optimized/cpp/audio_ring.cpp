// AudioRing — implementation. Port of realtime/runtime/audio_ring.py.
#include "audio_ring.h"

#include <algorithm>
#include <cmath>

namespace sa3 {
namespace rt {

static constexpr float kHalfPi = 1.57079632679489661923f;

std::pair<std::vector<float>, std::vector<float>> equal_power(int n) {
    std::vector<float> fi(std::max(0, n)), fo(std::max(0, n));
    for (int i = 0; i < n; ++i) {
        // linspace(0,1,n,endpoint=True): t = i/(n-1) (n==1 -> 0, matching numpy)
        float t = (n == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(n - 1);
        fi[i] = std::sin(0.5f * kHalfPi * 2.0f * t);   // sin(0.5*pi*t)
        fo[i] = std::cos(0.5f * kHalfPi * 2.0f * t);   // cos(0.5*pi*t)
    }
    return {std::move(fi), std::move(fo)};
}

std::vector<std::vector<float>> make_loopable(
    const std::vector<std::vector<float>>& audio, int xfade) {
    const int ch = static_cast<int>(audio.size());
    const int L = ch ? static_cast<int>(audio[0].size()) : 0;
    if (xfade <= 0 || L <= 2 * xfade) {
        return audio;   // copy unchanged
    }
    const int M = L - xfade;
    auto [fi, fo] = equal_power(xfade);
    std::vector<std::vector<float>> out(ch, std::vector<float>(M));
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < M; ++i) out[c][i] = audio[c][i];
        // out[:, :xfade] = head*fade_in + tail*fade_out  (tail = audio[M:L])
        for (int i = 0; i < xfade; ++i) {
            out[c][i] = audio[c][i] * fi[i] + audio[c][M + i] * fo[i];
        }
    }
    return out;
}

AudioRing::AudioRing(int length, int channels, int xfade, int loop_xfade)
    : loop_xfade_(loop_xfade),
      L_(length - loop_xfade),
      ch_(channels),
      xfade_(std::min(xfade, (length - loop_xfade) / 4)),
      buf_(channels, std::vector<float>(length - loop_xfade, 0.0f)) {
    auto ramps = equal_power(xfade_);
    fade_in_ = std::move(ramps.first);
    fade_out_ = std::move(ramps.second);
}

long AudioRing::pos() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return pos_;
}

long AudioRing::total_read() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return total_read_;
}

bool AudioRing::has_audio() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return has_audio_;
}

// Copy n frames from `pos` into out (wrapping once). Caller holds the lock.
void AudioRing::copy_from(long pos, int n, std::vector<std::vector<float>>& out) const {
    if (static_cast<int>(out.size()) < ch_) out.resize(ch_);
    for (int c = 0; c < ch_; ++c) {
        if (static_cast<int>(out[c].size()) < n) out[c].resize(n);
        const long end = pos + n;
        if (end <= L_) {
            std::copy(buf_[c].begin() + pos, buf_[c].begin() + end, out[c].begin());
        } else {
            const long first = L_ - pos;
            std::copy(buf_[c].begin() + pos, buf_[c].end(), out[c].begin());
            std::copy(buf_[c].begin(), buf_[c].begin() + (n - first),
                      out[c].begin() + first);
        }
    }
}

void AudioRing::read(int n, std::vector<std::vector<float>>& out) {
    std::lock_guard<std::mutex> lk(mutex_);
    const long pos = pos_;
    pos_ = (pos + n) % L_;
    total_read_ += n;
    copy_from(pos, n, out);
}

void AudioRing::read_peek(int n, std::vector<std::vector<float>>& out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    copy_from(pos_, n, out);
}

void AudioRing::advance(int n) {
    std::lock_guard<std::mutex> lk(mutex_);
    pos_ = (pos_ + n) % L_;
    total_read_ += n;
}

void AudioRing::write_loop(const std::vector<std::vector<float>>& audio_in) {
    // Fold to a seamless loop, then fit to L_.
    std::vector<std::vector<float>> folded =
        (loop_xfade_ > 0) ? make_loopable(audio_in, loop_xfade_) : audio_in;
    std::vector<std::vector<float>> a = fit(folded);

    std::lock_guard<std::mutex> lk(mutex_);
    if (has_audio_ && xfade_ > 0) {
        // Crossfade the xfade window STARTING AT THE CURRENT CURSOR (loops are
        // phase-aligned, so blending old[i]->new[i] at the cursor is seamless).
        for (int k = 0; k < xfade_; ++k) {
            const long idx = (pos_ + k) % L_;
            for (int c = 0; c < ch_; ++c) {
                a[c][idx] = a[c][idx] * fade_in_[k] + buf_[c][idx] * fade_out_[k];
            }
        }
    }
    buf_ = std::move(a);
    has_audio_ = true;
}

long AudioRing::write_head() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return write_head_;
}

void AudioRing::write_forward(long start, const std::vector<std::vector<float>>& audio,
                              int xfade) {
    const int len = (audio.empty() || audio[0].empty()) ? 0 : (int)audio[0].size();
    if (len <= 0) return;
    auto [fi, fo] = equal_power(std::min({xfade, len, L_}));
    const int xf = (int)fi.size();

    std::lock_guard<std::mutex> lk(mutex_);
    for (int c = 0; c < ch_; ++c) {
        const std::vector<float>& src = audio[std::min(c, (int)audio.size() - 1)];
        for (int k = 0; k < len; ++k) {
            long idx = (start + k) % L_;
            if (idx < 0) idx += L_;
            // Equal-power blend the leading xfade frames into what's already there
            // (a fresher decode of the same region), then hard-write the remainder.
            if (k < xf && has_audio_) {
                buf_[c][idx] = src[k] * fi[k] + buf_[c][idx] * fo[k];
            } else {
                buf_[c][idx] = src[k];
            }
        }
    }
    const long endp = start + len;
    if (endp > write_head_) write_head_ = endp;
    has_audio_ = true;
}

std::vector<std::vector<float>> AudioRing::fit(
    const std::vector<std::vector<float>>& audio) const {
    std::vector<std::vector<float>> src = audio;
    // mono -> stereo, or trim extra channels
    if (static_cast<int>(src.size()) != ch_) {
        if (src.size() == 1) {
            src.assign(ch_, audio[0]);
        } else {
            src.resize(ch_);
        }
    }
    const int n = src.empty() ? 0 : static_cast<int>(src[0].size());
    if (n == L_) return src;
    std::vector<std::vector<float>> out(ch_, std::vector<float>(L_, 0.0f));
    const int m = std::min(n, L_);
    for (int c = 0; c < ch_; ++c) {
        std::copy(src[c].begin(), src[c].begin() + m, out[c].begin());
    }
    return out;
}

}  // namespace rt
}  // namespace sa3
