"""Real-time runtime for the SA3 streaming engine (Phase 3).

Wraps the parity-gated StreamPipeline in a continuous, playable loop:
  - AudioRing      : a looping playback buffer; lock-free read for the audio
                     callback, crossfaded patch-in write from the generator.
  - GeneratorThread: owns ALL MLX work (thread-affinity) — submit/tick the ring
                     buffer, windowed-decode finished latents, write to AudioRing.
  - LiveMorph      : ties them together + the live control surface (denoise,
                     shared curves, prompt) for a "living-loop morph instrument".
"""
