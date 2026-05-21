# Remix — Stable Audio 3 web app

A small single-page app for **remixing** existing tracks: drop a song,
describe a transformation, get a remix. Built on top of `stable_audio_3`'s
audio-to-audio (init_audio) mode, with optional in-browser Demucs source
separation so vocals are preserved through the remix.

## Architecture

- **Server (`web/server.py`)** — FastAPI + uvicorn. Loads `StableAudioModel`
  once at startup. Exposes `POST /api/remix` (multipart: `audio` file +
  `prompt`, `init_noise_level`, `steps`, `cfg_scale`, `seed`) and serves the
  static frontend.
- **Frontend (`web/static/`)** — vanilla HTML/CSS/JS, no build step.
  - `js/app.js` — drop-zone, controls, remix orchestration, WAV encoding.
  - `js/demucs.js` — in-browser source separation (WebGPU when available,
    WASM fallback). Port of the reference at
    https://github.com/sevagh/demucs.onnx (`src_wasm/demo.js`).

## Remix flow

1. Drop audio → decode → resample to 44.1 kHz stereo client-side.
2. *(Optional)* Demucs runs in the browser → `{drums, bass, other, vocals}`.
3. The instrumental sum (or full track) is encoded as WAV and POSTed to
   `/api/remix`.
4. Server runs `StableAudioModel.generate(init_audio=..., ...)` and returns a
   WAV.
5. If we separated, the original vocals are mixed back in client-side
   (sample-aligned, resampled if needed, with a "Vocal level" gain).

## Setup

### 1. Install Python deps

```bash
uv sync --extra web
```

### 2. Drop in the Demucs assets

The browser needs the Demucs ONNX model and the compiled WASM artifacts. They
are **not** vendored here — copy them in:

```
web/static/models/htdemucs.onnx          # Demucs ONNX
web/static/wasm/demucs_wasm.js           # Emscripten glue
web/static/wasm/demucs_wasm.wasm
web/static/wasm/demucs_wasm.worker.js    # (if your build emits threading worker)
```

If you build them from `/root/projects/demucs.onnx`, that's:
- model: `onnx-models/htdemucs.onnx`
- wasm: `build/build-wasm/demucs_wasm.{js,wasm,worker.js}`

If these aren't present, the app still works **with the vocal-separation
toggle off** — it just remixes the full track directly.

### 3. Run

```bash
uv run --extra web python web/server.py
```

Defaults to `http://0.0.0.0:9000`. Set the model via env var:

```bash
SAO_MODEL=medium uv run --extra web python web/server.py
```

The first request after startup waits for the model to finish loading (~few
seconds on a warm GPU). After that, generation is ~real-time-ish at 8 steps.

## Controls

- **Prompt** — free text. AudioSparx-style tags help (`TrackType: Music,
  VocalType: Instrumental, …`). See `docs/guides/prompting.md`.
- **Remix amount** (init_noise_level, 0.01–1.0) — low keeps melody/rhythm
  close to the input; high lets the model deviate more (timbre and tonality
  transfer).
- **Steps** (1–50) — more = slower, possibly cleaner. 8 is the medium
  default.
- **CFG scale** (0–25) — prompt adherence; 1.5 is a reasonable default.
- **Vocal level** — only shown when separation is on; gain applied to the
  vocals stem before mixing back.

## Notes / limits

- No inpainting/continuation UI in v1 — just init-audio remix.
- Track length is capped at the model's max (380 s for `medium`).
- Cross-origin isolation headers (`COOP: same-origin`, `COEP: require-corp`)
  are set globally by middleware so `SharedArrayBuffer` works for the
  threaded ONNX-Runtime-web WASM backend.
- Vocal-presence detection is intentionally **not** implemented yet — the
  toggle is exposed unconditionally with a sensible default of ON. Add a
  pre-flight RMS heuristic on the vocals stem if you want auto-disable.
