# SA3 Realtime

A real-time, **conditional audio-generation engine** for Stable Audio 3 on Apple Silicon (MLX / Metal), plus a JUCE plugin (**SA3 Realtime**) built on it.

> **This is not a loop tool.** The core is a streaming generator that runs over a **sliding window of an input signal**: at every moment it generates output conditioned on (a) a rolling window of the *recent* input audio and (b) a control input, and streams the result out in real time as the input advances. The window is the model's working **context** — "what it's listening to right now" — not a loop to be played back.

## The general shape

What is **fixed** in the architecture:

- a **rolling input window** (the context / the model's working memory),
- a **streaming diffusion engine** (StreamDiffusion / DEMON-style ring buffer) that generates conditioned on that window, and
- a **forward-streamed output** at 1× that advances as the input does.

What is **task-specific** and swappable on top:

- the **conditioning** (the control input), and
- the **input → output relationship**.

### Tasks are "heads" on the same engine

| Task | The input window is… | Output is… | Routing |
|---|---|---|---|
| **Style transfer** (today) | the thing being **transformed** | a restyled version of the input | **replaces** the input (dry → wet) |
| **Accompaniment** (future) | **context to respond to** | **new, complementary** material (drums, bass, harmony…) | **mixes with** the input |

The axis that changes between tasks is **transform-in-place vs. generate-alongside**, and it reaches beyond the model:

- **Routing.** Style transfer mutes the dry input and plays the wet; accompaniment must play **input + generated** together, in time.
- **Input source.** Both ultimately want a **live input** (a musician playing into a rolling buffer), not just a dropped file.
- **Sync / latency.** Generate-alongside raises the bar: the generated part has to land *rhythmically aligned* with the live input.

The streaming-window engine described below is the **reusable substrate**. Style transfer is one head; accompaniment is another that needs a model/conditioning capable of *generate-alongside* plus the input+output mix routing.

## Today's app: real-time text-prompt style transfer

Drop a piece of audio, type a **Style**, and the output transforms in real time toward that style, based on a recent **Window** of the input. A short source is treated as a single window; a longer track **plays through** while the window slides forward, restyling the recent context continuously.

The controls (plugin):

| Control | Meaning |
|---|---|
| **Amount** | the headline morph: 0 = the input as-is, 1 = fully styled (a latent morph toward the styled target) |
| **Style** | the text prompt that defines the styled target |
| **Intensity** | how far the styled endpoint departs from the source |
| **CFG** | prompt-guidance strength for the styled target |
| **Window** | the length of the rolling context the morph is based on (default 10 s) |
| **Motion** | per-frame texture (velocity) |
| **Mode** | Coherent (fixed seed) vs Evolve (walking seed) — meaningful in single-window mode |

## Engine architecture

Built bottom-up (Phases 0–5; see the engineering notes for parity gates and measurements):

- **`StreamPipeline`** (`optimized/cpp/stream.{h,cpp}`, Python ref `realtime/engine/stream.py`) — a StreamDiffusion/DEMON-style **ring buffer of in-flight generations**, each at a different denoising step, advanced together by one batched DiT forward per `tick()`. After warmup, finished latents stream out at `depth/steps` per tick. Per-frame control **curves** (`velocity_scale`, `sde_denoise_curve`, `x0_target_strength`) are shared-mutable → **~1-tick latency**; structural changes (new prompt / new source window) are per-request → `steps`-tick latency. Bit-exact streaming-vs-batch.

- **`StreamGenerator`** (`optimized/cpp/stream_generator.{h,cpp}`) — owns all MLX work on one worker thread (model load, ticks, decode), fills an audio ring. Two run modes:
  - **Single-window** (source ≤ window): a banked-target **x0-morph** — a source-anchored base generation morphed toward a once-computed styled target by `x0_target_strength` (= Amount, 1-tick).
  - **Playthrough** (source > window): the **forward-streaming** path —
    - a **forward FIFO** (`AudioRing::write_forward`): the producer writes ahead of the play cursor, the consumer plays forward; no looping.
    - **windowed decode at the advancing playhead**: each tick decodes a small slice of the current latent at the play position and writes it forward.
    - a **per-chunk banked styled target**, generated **once** and held (stable — no per-completion stochastic silence); the output is a latent morph `lerp(source, target, Amount)`, so Amount stays a ~1-tick lerp.
    - **interior margins**: each chunk is generated with margins so the playhead only ever decodes the chunk **interior**, never SA3's intro/outro fade.
    - a **one-chunk look-ahead** + **equal-power crossfade** between adjacent chunks (decoding the identical source position in both interiors) for seam-free boundaries.

- **Models / quality tiers.** *Live* = sm-music DiT + SAME-S codec (fast, lighter text-following). *Quality* = medium DiT + SAME-L codec (stronger transfer, slower per tick). Both share the streaming engine.

- **Plugin** (`plugin_morph/`) — JUCE 8 WebView plugin. `MorphEngine` (a `juce::AudioSource`) owns the `AudioRing` + `StreamGenerator` (which owns the MLX thread) and resamples the 44.1 kHz ring into the host block. UI ↔ engine over the `juce://` bridge; the engine pushes live telemetry (playhead / meter) back at ~30 fps.

## The constraint that shapes everything

No TensorRT on Apple Silicon — the substrate is **MLX/Metal, eager** (`mx.compile` disabled). So:

- **Per-frame curves are ~1-tick**; anything that requires a new generation is `steps`-tick.
- DiT cost scales with the window length, so windows stay modest (≈10 s), and the **window / overlap / margin / look-ahead** are the knobs that trade boundary smoothness against latency.
- DEMON gets seamless playthrough from end-to-end TRT (~10× faster); our equivalent is the look-ahead + interior-margin + crossfade machinery above.

## Files

- `optimized/cpp/stream.{h,cpp}`, `stream_ode.{h,cpp}` — the streaming engine + step primitives (C++ / MLX).
- `optimized/cpp/stream_generator.{h,cpp}`, `audio_ring.{h,cpp}` — the in-process runtime (MLX worker, ring buffer, playthrough).
- `optimized/cpp/test_scan_structure.cpp` — headless harness: pushes a source through the scan engine, captures output for structural analysis (advancement, dropouts, styling strength).
- `realtime/engine/`, `realtime/runtime/`, `realtime/bench/` — Python-MLX prototype + parity gates the C++ port was built against.
- `plugin_morph/` — the SA3 Realtime JUCE plugin (Source + WebView Resources).

## Status & roadmap

**Working:** real-time text-prompt style transfer on a dropped file — single-window morph and long-track playthrough, on both Live and Quality, with a stable forward stream and Amount morphing input ↔ styled.

**Next, to grow into the general vision:**

1. **Live audio input** — a rolling input buffer instead of file-only, so the window is genuinely "the last N seconds of what's coming in."
2. **Pluggable task heads** — treat the conditioning + input/output relationship as a swappable "task" rather than hardcoded style transfer; **accompaniment** (generate-alongside, input+output mix) is the first target beyond style.
3. **Transport / tempo sync** — required once the output must align rhythmically with a live input.
