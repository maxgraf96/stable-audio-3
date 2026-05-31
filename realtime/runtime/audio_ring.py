"""AudioRing — a fixed-length looping playback buffer for the morph instrument.

The audio callback (PortAudio, real-time thread) READS a contiguous block per
call, wrapping at the loop boundary, and must never block. The generator thread
WRITES freshly-decoded loops in, equal-power-crossfaded against what's already
playing so a new morph fades in without a click.

Design (single-writer / single-reader, matches DEMON's audio_engine):
  - The buffer holds exactly one loop (L samples, stereo). Playback wraps
    modulo L forever, so a fixed-length SA3 generation plays as a seamless loop.
  - read() advances a play cursor and copies out, wrapping once. No allocation,
    no lock on the steady path (the writer swaps a whole buffer under a lock the
    reader briefly takes — sub-microsecond).
  - write_loop() equal-power-crossfades the incoming loop against the current
    one over a short head window, so morph updates are inaudible at the seam.

Channels-first audio convention everywhere: [2, L] float32 in [-1, 1].
"""

from __future__ import annotations

import threading

import numpy as np


def equal_power(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power (constant-energy) crossfade ramps of length n: (fade_in, fade_out)."""
    t = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    fade_in = np.sin(0.5 * np.pi * t)
    fade_out = np.cos(0.5 * np.pi * t)
    return fade_in, fade_out


def make_loopable(audio: np.ndarray, xfade: int) -> np.ndarray:
    """Turn a finite clip [ch, L] into a seamless loop of length M = L - xfade by
    overlapping its tail back over its head.

    SA3 generates a piece with an intro/outro (it "opens and closes" — the
    DEMON-paper limitation), so a raw clip looped jumps loud-start <- quiet-end.
    We fold the trailing `xfade` samples (faded out) into the leading `xfade`
    (faded in). Result: at the wrap M-1 -> 0 the samples are consecutive in the
    original (continuous), AND the tail's energy is mixed into the head so the
    loop point's level matches. Equal-power ramps keep perceived loudness flat.

      out[0:xfade] = head*fade_in + tail*fade_out
      out[xfade:M] = audio[xfade:M]
    """
    ch, L = audio.shape
    if xfade <= 0 or L <= 2 * xfade:
        return audio.astype(np.float32, copy=True)
    M = L - xfade
    fi, fo = equal_power(xfade)
    out = audio[:, :M].astype(np.float32, copy=True)
    out[:, :xfade] = audio[:, :xfade] * fi + audio[:, M:L] * fo
    return out


class AudioRing:
    def __init__(self, length: int, channels: int = 2, xfade: int = 1024,
                 loop_xfade: int = 0):
        # `length` is the DECODED clip length L. loop_xfade folds the tail over
        # the head to make it loop, so the PLAYABLE loop is L - loop_xfade.
        self.loop_xfade = int(loop_xfade)
        self.L = int(length) - self.loop_xfade
        self.ch = int(channels)
        self.xfade = min(int(xfade), self.L // 4)
        self._buf = np.zeros((self.ch, self.L), dtype=np.float32)
        self._pos = 0                       # play cursor (reader-owned)
        self._lock = threading.Lock()
        self._fade_in, self._fade_out = equal_power(self.xfade)
        self._has_audio = False

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def has_audio(self) -> bool:
        return self._has_audio

    def read(self, n: int) -> np.ndarray:
        """Return the next n samples [ch, n], wrapping at the loop boundary.
        Real-time safe: one short lock to snapshot the buffer ref + cursor."""
        with self._lock:
            buf = self._buf
            pos = self._pos
            self._pos = (pos + n) % self.L
        out = np.empty((self.ch, n), dtype=np.float32)
        end = pos + n
        if end <= self.L:
            out[:] = buf[:, pos:end]
        else:                               # wrap once
            first = self.L - pos
            out[:, :first] = buf[:, pos:]
            out[:, first:] = buf[:, :n - first]
        return out

    def write_loop(self, audio: np.ndarray) -> None:
        """Install a new loop, equal-power-crossfaded against the current one over
        the `xfade` samples STARTING AT THE CURRENT PLAY CURSOR, so the swap is
        click-free wherever the playhead happens to be (not just at the seam).
        Both loops are phase-aligned (index i == same musical position), so
        blending old[i]->new[i] across a window at the cursor is seamless. `audio`
        is the full decoded clip [ch, L]; we first fold it into a seamless loop of
        length self.L via make_loopable. Called from the generator thread only."""
        if self.loop_xfade > 0:
            audio = make_loopable(audio, self.loop_xfade)
        a = self._fit(audio)
        with self._lock:
            if self._has_audio and self.xfade > 0:
                idx = (self._pos + np.arange(self.xfade)) % self.L   # window at cursor
                old_seg = self._buf[:, idx]
                new_seg = a[:, idx]
                a[:, idx] = new_seg * self._fade_in + old_seg * self._fade_out
            self._buf = a
            self._has_audio = True

    def _fit(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] != self.ch:       # mono -> stereo, or trim extra ch
            audio = (np.repeat(audio, self.ch, axis=0) if audio.shape[0] == 1
                     else audio[:self.ch])
        n = audio.shape[1]
        if n == self.L:
            return audio.astype(np.float32, copy=True)
        out = np.zeros((self.ch, self.L), dtype=np.float32)
        m = min(n, self.L)
        out[:, :m] = audio[:, :m]
        return out
