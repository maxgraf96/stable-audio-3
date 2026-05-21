# Remix — Design Brief

A working app exists. This document explains what it does, why it exists,
who it's for, and how it currently behaves — so a designer can reimagine
the interface with full context.

---

## 1. At a glance

**Remix** is a single-page web app that takes an existing song, lets you
describe a transformation in plain English ("turn this into a bossa nova
jazz piece with brushed drums and brazilian violins"), and produces a
remixed version that keeps the original's structure but changes its
character.

It's a thin, focused UI on top of three pieces:

- **Stable Audio 3** (Stability AI's latest audio diffusion model) — does
  the actual remix on a GPU server.
- **Demucs** (Meta's source separation model, running entirely in the
  browser via WebAssembly/WebGPU) — pulls vocals out and back in cleanly.
- **A music-intelligence pipeline** (internal service) — auto-detects BPM
  and musical key from the dropped track and feeds them to the model.

Everything is one route: drop → describe → remix → compare.

---

## 2. The idea

Generative audio models are amazing at making music from nothing, but
musicians and producers rarely want "something from nothing." They want
*"this thing, but different."* Take this song, give it a different mood;
take this loop, swap out the instruments; take this sketch, render it as
if a real band played it.

Stable Audio 3 supports this via **init audio** mode: instead of starting
from noise, the diffusion process starts from your existing track and
walks away from it by a controllable amount. Low amount = subtle
variation; high amount = drastic style transfer. This is the core
operation the app exposes.

There are two complications the app exists to solve:

1. **Stable Audio mangles vocals.** When the input has singing, the model
   warps it into garbled vocalisations because it was trained mostly on
   instrumental data and doesn't preserve intelligible lyrics. So we need
   to pull vocals *out* before remixing, then layer them *back on top*
   afterwards. That's what the in-browser Demucs is for.
2. **BPM and key drift.** Without telling the model the tempo and key,
   the remix often shifts both, which makes a "remix" feel disconnected
   from the source. The intelligence pipeline detects both on drop and
   appends them to the prompt automatically.

Everything else in the UI is in service of those two ideas — plus the
ability to iterate fast, because the right "remix amount" and prompt
combo always needs a few tries.

---

## 3. Who it's for

Three concentric audiences:

**Primary — musicians, producers, sound designers** who already think in
terms of stems, BPM, key, and style references. They want a tool that
respects their vocabulary, doesn't hide knobs, and produces results fast
enough to iterate. Power users — they will tinker.

**Secondary — DJs and remixers** looking for raw material. They drop a
vocal track, describe a backing they want, and use the result as a stem
in a larger project. Don't care about the science, just the output.

**Tertiary — curious people** who want to hear what their favourite song
sounds like as bossa nova. They'll drop a YouTube rip and play. The UI
shouldn't require musical literacy to be approachable, but shouldn't dumb
itself down for the primary audience either.

What ties all three together: **the click that matters is "Remix."**
Everything before it should feel like minimal setup; everything after
should feel like instant comparison.

---

## 4. How Stable Audio 3 works in one minute

(Useful context — the designer doesn't need to surface this, but
understanding it helps shape the metaphors.)

- The model is a **diffusion** model. It generates audio by starting
  from noise and iteratively denoising it across N steps, guided by a
  text prompt.
- In **init audio** mode, instead of pure noise, it starts from
  *your audio + some noise*. The `init_noise_level` (0.01–1.0) controls
  how much noise to add. Low noise = the model only nudges the input;
  high noise = it mostly starts from scratch with a faint shadow of
  the input.
- The model has a **maximum duration** baked in (~6:20 for the `medium`
  model we use). The remix output is the same length as the input.
- The model is trained on AudioSparx data which uses **structured tags**
  like `TrackType: Music, VocalType: Instrumental, Genre: Funk,
  Instruments: Guitar, Saxophone`. Prompts in this style perform much
  better than pure natural language.
- It produces **stereo, 44.1 kHz** audio.

The model takes 10–40 seconds to remix a 3-minute track on a warm GPU.
That latency is the dominant UX constraint.

---

## 5. The remix flow (user journey)

Today the flow is linear and stateful — one track, one set of controls,
one result at a time. Iteration happens by changing controls and
clicking Remix again.

1. **Land.** Empty page with a single big drop zone. Hero text:
   "Remix — drop a track, describe a transformation, get a remix."
2. **Drop.** User drops a file (or clicks to browse). Page decodes it
   client-side, shows the original in an HTML audio player, reveals the
   controls panel below. **In the background**, the file is also POSTed
   to the analysis pipeline — BPM/Key fields show a tiny spinner.
3. **(Optional) Tweak.** User edits the prompt, BPM, Key, the two
   vocal-handling toggles, the remix amount, the CFG scale, the vocal
   level. Sane defaults are pre-filled so most users can skip this.
4. **Remix.** Click. If vocal separation is enabled and not yet done,
   Demucs runs in the browser (10–30s, shown as progress text). Then the
   audio is uploaded and the model generates. Then the result decodes,
   and if "mix vocals back" is on, the vocals are layered on top.
5. **Compare.** Result panel appears with a custom A/B player. User can
   instantly switch between original and remix while playing. Download
   button is a sibling.
6. **Iterate.** Change a control, click Remix again. Separation is
   cached so re-runs only pay the GPU cost (~10–40s).
7. **Replace.** Click "Replace" to drop a new track. Everything resets
   except the prompt and slider values.

---

## 6. What's on screen today

This is the current layout, top-to-bottom. Each section is one block.

### Hero
- App title "Remix" in a purple-blue gradient.
- Tagline: "Drop a track, describe a transformation, get a remix."

### Drop zone (until a track is loaded)
- Large dashed-border panel.
- Centered up-arrow icon, "Drop a track here", "or choose a file" link,
  format hint ("MP3, WAV, FLAC, OGG — up to ~6 minutes").
- Hover-highlights when dragging a file over it.

### Track panel (once loaded)
- File name + duration on the left, "Replace" button on the right.
- HTML `<audio controls>` element to preview the original.
- A boxed area with two toggles, each with a title and a one-sentence
  explanation:
  - **Separate vocals before remixing** (default: on). Strips vocals from
    the upload so the model only remixes the instrumental.
  - **Mix vocals back on top of the remix** (default: on). Layers the
    original vocals over the model output.

  Both can be independently on/off. The matrix:

  | Strip | Add back | Result |
  |---|---|---|
  | on  | on  | Cleanest — instrumental in, instrumental out, vocals re-added |
  | on  | off | Pure instrumental remix |
  | off | on  | Full track in, model mangles vocals, original vocals masked on top |
  | off | off | No Demucs, model does whatever it does |

### Controls panel
- **Prompt** — multi-line textarea, prefilled with an AudioSparx-style
  example.
- **BPM** and **Key** — two text inputs side by side, auto-filled by the
  analysis pipeline on drop. Both show a tiny spinner + "analyzing…"
  label while in-flight, then clear. User can manually override (manual
  edits are respected and never overwritten).
- **Sliders** (2-column grid):
  - **Remix amount** (0.01–1.0, default 0.54) — `init_noise_level`. The
    most important creative knob.
  - **CFG scale** (0–25, default 4.0) — prompt adherence.
  - **Vocal level** (0–2, default 1.0) — gain for the re-added vocals.
    Only visible when "mix vocals back" is on.
- **Remix** button — full-width gradient, primary CTA.

### Status (transient)
- Spinner + plain text status while working ("Separating vocals…",
  "Remixing on the GPU…", etc.).

### Result panel (after a successful remix)
- Title "Result" + Download button (downloads the result WAV).
- Custom A/B player:
  - Round play/pause button.
  - Current time `0:00`, seek bar, total time `0:00`.
  - Segmented control "A · Original | B · Remix" (pill style, active
    pill has the gradient fill).
  - Hint: "Space = play/pause · X = swap A/B".

### Admin modal
- Hidden by default. Toggled with **Shift + F12**, dismissed with **Esc**
  or backdrop click. Holds the "Steps" slider (diffusion steps, default 8).
  This is an escape hatch for power users; the average user never sees it.

### Error box (only on failure)
- Red-tinted band at the bottom with the error message.

---

## 7. Controls — what each one actually does

A designer should know what each knob *means* so they can decide
hierarchy, labels, and tooltip copy.

- **Prompt** — free text, but works best with AudioSparx tag syntax.
  Examples:
  - `TrackType: Music, VocalType: Instrumental, Genre: Funk, Instruments:
    Bass, Drums, Electric Piano. Warm 70s soul groove, 90 BPM.`
  - `Bossa nova jazz, Brazilian violins, light acoustic guitar.`
  We auto-append BPM and Key to whatever the user types.

- **BPM** — beats per minute (integer). Auto-detected.

- **Key** — musical key. Auto-detected as e.g. "A Major", "F# Minor".

- **Remix amount** (init noise level, 0.01–1.0) — the **defining creative
  knob.** Low (0.2–0.4): subtle variation, melody and rhythm mostly
  preserved. Medium (0.4–0.7): timbre and mood change but you can still
  hear the original shape. High (0.7–1.0): strong style transfer, the
  source becomes a faint suggestion.

- **CFG scale** (0–25) — how strictly the model follows the prompt.
  Low values let the model wander; high values pin it down hard but can
  produce artifacts. Default 4.0 is a sensible middle ground.

- **Steps** (1–50, in admin modal) — diffusion denoising steps. More
  = slower, sometimes cleaner. 8 is fine for the medium model. Power
  users only.

- **Vocal level** (0–2) — linear gain on the re-added vocals stem
  before summing. 1.0 = same level as Demucs output. >1.0 = louder
  than original. Useful when the remixed backing has higher energy
  than the original instrumental and would otherwise drown the
  vocals.

---

## 8. Technical architecture (for context, not for screen)

A designer doesn't need to render this, but knowing where compute
happens shapes the UX (what's instant vs. slow, what needs progress UI).

```
┌─────────── Browser ────────────┐         ┌─────── GPU Server ────────┐
│                                │         │                            │
│  Drop file                     │         │  FastAPI                   │
│   ↓                            │         │   ├─ POST /api/remix       │
│  Decode + resample to 44.1k    │         │   │   → StableAudioModel   │
│   ↓                            │         │   │     .generate()        │
│  ┌──────────────────────────┐  │         │   │   → WAV               │
│  │ POST /api/analyze   ─────┼──┼─proxy──▶│   │                        │
│  │   ← {bpm, key}           │  │   to    │   ├─ POST /api/analyze     │
│  └──────────────────────────┘  │ pipeline│   │   → forwards file with │
│                                │         │   │     bearer token       │
│  Show controls                 │         │   │   ← {bpm, key}         │
│   ↓                            │         │   │                        │
│  (Click Remix)                 │         │   └─ GET /…  → static SPA  │
│   ↓                            │         │                            │
│  Demucs WASM separation        │         │  Stable Audio model        │
│  (WebGPU when available)       │         │  stays loaded in VRAM      │
│   ↓                            │         │  across all requests.      │
│  POST /api/remix       ───────┼──upload──▶                            │
│   ← remix WAV                  │         │                            │
│   ↓                            │         └────────────────────────────┘
│  Decode, optionally mix        │
│  vocals back on top            │
│   ↓                            │         External service (proxied):
│  Custom A/B player             │           ai.pipeline.corpus.music
└────────────────────────────────┘
```

**Where work happens:**

- **In the browser** — file decoding, resampling, source separation
  (Demucs, WebGPU/WASM), the entire A/B playback engine (Web Audio API
  with two `GainNode`s being crossfaded).
- **On the GPU server** — only the Stable Audio inference. The server
  also proxies the analysis call so the bearer token never leaves the
  backend.
- **Externally** — the music-intelligence pipeline (separate service).

This split means a 5-minute remix involves a ~30-second WAV upload from
the browser, ~10–40s of GPU work, then a ~30s WAV download. The Demucs
separation runs entirely client-side and takes ~10–30s on first run, then
is cached for the same track so subsequent remixes skip it.

---

## 9. Performance & timing — what the designer should expect to surface

This is what the user is waiting on at each step. Knowing the cost
helps decide what deserves a progress UI vs. a spinner vs. nothing.

| Action | Typical time | Already shown how |
|---|---|---|
| Decode audio after drop | <1 s | "Decoding audio…" status |
| Analyze BPM/Key | 3–10 s | Spinner next to label |
| Demucs separation (first time, per track) | 10–30 s | "Separating N/M…" status |
| Demucs separation (cached) | 0 | "Reusing cached separation…" |
| Upload WAV | 1–5 s | "Encoding upload…" / network |
| Stable Audio inference | 10–40 s | "Remixing on the GPU…" |
| Decode result | <1 s | "Decoding remix…" |
| Mix vocals back in | <1 s | "Mixing vocals back in…" |

The big two are **Demucs** (first time only, in browser) and **the model
inference itself** (server). Both deserve visible progress.

The download of model weights also matters:
- **Stable Audio model** loads once at server startup (one-time, the
  user never sees it after the server is up).
- **Demucs model (~150MB)** downloads to the browser on first remix
  with separation enabled. We cache it in the browser's Cache Storage,
  so subsequent visits are instant. First-time experience is "Downloading
  separation model… (cached for next time)".

---

## 10. Interaction details worth preserving

These are the behaviours users will miss if a redesign drops them.

- **Drop-and-go.** Dropping a file triggers decoding *and* analysis
  immediately, in parallel — the user can be editing the prompt while
  BPM/Key fill in.
- **Cached separation.** Changing the prompt or sliders and re-clicking
  Remix on the same track does not re-run Demucs. The vocal-toggle
  combinations all share one cached separation per track.
- **Manual override of auto values.** If the user types in BPM or Key
  before analysis returns (or before they've heard the auto result),
  their typed value is never overwritten.
- **A/B is sample-aligned and clickless.** Both buffers play in parallel
  through gain nodes; switching A↔B crossfades over ~12 ms. There's no
  reload, no seek snap, no perceptible glitch. Toggling rapidly while
  playing is the killer feature — that's how producers compare takes.
- **Space = play/pause, X = swap A/B.** Keyboard-first. Only active
  when the result panel is visible. Disabled while typing.
- **Shift + F12 opens the admin modal.** Esc closes it.
- **Vocal-level slider only appears when "mix vocals back" is on.**
  Less surface area when it's irrelevant.
- **"Replace" preserves controls.** New track, same prompt/sliders. So
  the user can A/B the same recipe across different sources.

---

## 11. Look & feel today

Dark UI. Background is `#0b0b0e` with two large soft radial gradients
in purple-blue. Panels are slightly lighter (`#15151a`) with `#26262f`
borders, 12px radius, soft drop-shadow. Accent gradient is
`#7c8cff → #b070ff` (used on the hero text, the primary button, the
active A/B pill, slider thumbs, focused borders).

Font: system UI stack (San Francisco / Segoe / Inter / system-ui).

Width: capped at 760px, centered, mobile-friendly down to a single
column when the viewport narrows.

This is "functional but unconsidered." A starting point, not a
destination. The visual language could go almost anywhere — playful,
clinical, brutalist, music-app-glossy. The structure should hold up.

---

## 12. Constraints to respect

- **Single-track flow.** Today the app does one remix at a time. It's
  not a project workspace, not a multi-track DAW, not a versioned
  history. Feel free to redesign for that scope. (A "Recent results"
  shelf would be a great addition; a full project sidebar would be
  scope creep.)
- **Long waits are unavoidable.** Remix is fundamentally a 30–60-second
  operation. Don't pretend it's instant; design for the wait
  (interesting progress, motion, something to look at).
- **Inputs and outputs are audio.** The hero element is *sound*, not
  pixels. A redesign that doesn't give audio visual presence
  (waveforms, spectrograms, motion) is wasting an opportunity.
- **No build step.** The frontend is vanilla HTML/CSS/JS, served
  directly. A redesign can introduce a build pipeline if needed, but
  not required — keeping it dependency-free is a real virtue.
- **WebGPU isolation.** The page requires `COOP: same-origin` and
  `COEP: require-corp` headers so Demucs can use SharedArrayBuffer.
  External assets (CDN scripts, fonts, images) need to either be
  same-origin or have `Cross-Origin-Resource-Policy: cross-origin`. In
  practice this means: prefer locally-hosted assets; CDN-hosted things
  that don't set the right header will fail to load.
- **Mobile is a stretch goal.** Demucs in the browser is heavy.
  Mobile-first isn't realistic for now; mobile-doesn't-look-broken is
  the target.

---

## 13. Open design questions

A non-exhaustive list of things worth thinking about:

- **What's the metaphor?** Mixing console? Tape machine? Notebook?
  Studio whiteboard? The current UI is "settings page." It could be a
  much stronger metaphor.
- **Hierarchy of controls.** Today the prompt is at the top, sliders
  in a grid below, toggles in their own box above. Is that the right
  emphasis? The remix amount is arguably the most important knob and
  it's currently the same size as everything else.
- **A/B as the centrepiece.** Right now A/B only appears after the
  first remix. What if it's always there from the moment you drop —
  showing the dropped track on both sides until a remix exists, then
  filling B in?
- **Progress as content.** The "Remixing on the GPU…" status is a
  tiny spinner and a line of text. With a 10–40s wait, this is dead
  weight. What if it showed a live waveform of the original being
  dissolved into the remix? A spectrogram morph? A countdown?
- **Result history.** Right now each remix overwrites the previous.
  Iteration would benefit from a shelf of the last N results to
  A/B/C/D between.
- **Per-toggle visualization.** Should the vocal separation toggles
  show *which* vocals were found (a tiny waveform of the vocals stem)?
  Show "vocals detected: yes/no" automatically?
- **Mobile.** What's the right collapsed view for a phone? Or do we
  just say "use a desktop"?
- **Onboarding.** New users have no idea what "Remix amount" or "CFG
  scale" means. Tooltips exist (the `?` next to each slider label) but
  they're easily missed. Should the first run feel like a guided
  example? Should there be preset chips ("Subtle", "Strong", "Wild")?

---

## 14. Deliverables we'd love from a designer

Order of priority:

1. **High-fidelity comps** of the empty state, the dropped-but-not-yet-
   remixed state, the remixing state (with progress), and the
   compare-result state. These are the four moments of the app.
2. **An interaction spec for the A/B player.** This is the
   differentiated piece — it should feel premium.
3. **Treatment of progress / latency.** What does the user look at
   for 40 seconds while a remix renders?
4. **Type and color system** — a coherent palette and type scale that
   we can implement.
5. **Mobile breakpoint** (stretch).
6. **Optional: a metaphor or visual language proposal** — even if it's
   "let's do a tape machine," that decision drives everything else.

---

## 15. What's intentionally out of scope (today)

So the designer knows what *not* to design for:

- Multi-track / DAW workflows.
- User accounts, saved sessions, sharing.
- Inpainting (regenerate a section in the middle) and continuation
  (extend a track). These are Stable Audio features we deliberately
  don't expose yet.
- Multiple LoRAs / fine-tuning controls.
- Batch generation (multiple variants from one prompt).
- Lyrics, song structure detection, stem-level editing.

Any of these could land later; none of them need to be designed for now.

---

## 16. One-line summary, if it helps

**Remix is an A/B comparison tool with a remix engine attached.**
Everything that isn't "drop, describe, click, compare" is supporting
infrastructure. Design accordingly.
