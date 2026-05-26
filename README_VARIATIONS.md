# SA3 Sample Variations

Inference-only variation harness for Stable Audio 3 on Apple Silicon. Wraps the official `optimized/mlx/sa3` runner with a candidate-grid sweeper, two 5-shot product presets, a diagnostic preset, and a local drag-and-drop web UI.

Two scripts:

- `sa3_variations.py` — CLI harness
- `sa3_studio.py` — local web UI

## Two product targets

The variation problem has two valid framings that lead to very different parameter regimes. The harness supports both as distinct presets.

| | Free variation (`--preset free`, **default**) | Preserve sound (`--preset app`) |
|---|---|---|
| Goal | Different sounds, same musical slot | Same sample, varied |
| Sound identity | varies freely | preserved (Splice-like) |
| Timbre | can drift | must survive |
| Instrument | may change | must stay |
| Harmonic context | preserved | varies subtly |
| Noise level | a2a 0.45 (slider-controllable in studio) | a2a 0.45 (slider-controllable), inpaint 0.85 |
| Guidance | cfg=1.0 (unconditional) | cfg=4 for a2a, cfg=4 for inpaint |
| Per-candidate steers | none, empty prompt | yes (musical directions) |

Both produce 5 candidates per run from the same input. Neither is "wrong" — they solve different problems and a real product probably offers both. `free` is the default because it consistently produces a nice neighbourhood around the original with no prompt engineering on the user's part; `app` is for when you specifically need the timbre to stay put.

## Quick start

### Studio (recommended)

```bash
optimized/mlx/.venv/bin/python sa3_studio.py
```

Opens `http://localhost:8765`. Drop a WAV and click Generate — **Free variation** is the default-on preset; toggle to **Preserve sound** if you need to lock timbre instead. ~7.7s for 5 candidates on M4 Max once the pipeline is warm (10s loop, medium + same-l, 8 steps), ~9.8s on the first click while the pipeline loads. The pipeline stays resident across clicks for the same audio duration; a new duration triggers a ~1.3s rebuild.

The studio auto-detects BPM and key from the dropped filename using the common `<bpm>_<key>_<name>` convention (Noiiz / Loopmasters / older Splice). Examples:

- `128_A_FifthsPluckSynth_718.wav` → BPM=128, Key=A
- `loop_124bpm_Am_drums.wav` → BPM=124, Key=A minor

Outputs land in `runs/studio_<timestamp>_<preset>/` with the converted source, 5 variations, a `manifest.jsonl`, and a `review.html`.

### CLI

```bash
optimized/mlx/.venv/bin/python sa3_variations.py \
  --sa3-root . \
  --input <some.wav> \
  --kind melodic \
  --outdir ./runs/<name>
```

Defaults to `--preset free` (unconditional a2a at n=0.45, varied seeds). Add `--preset app` for the prompt-anchored "preserve sound" regime. Use `--noise-a2a` to override the noise level (mirrors the studio slider). Filename-based BPM/key auto-detection runs by default for `app` (which uses BPM for bar-aware inpaint masks); `--bpm` / `--key` override. `free` ignores both since it sends an empty prompt and has no inpaint candidates.

## What we learned about SA3 for variation

These findings drove the defaults in the presets. They are validated on **melodic loops**; drums / one-shots / SFX use sensible parameters but have not been listening-tested at the same depth.

### 1. Inpainting has structural timbre preservation; a2a does not.

In **inpaint mode**, the unmasked region is paste-back exact at every step. Timbre survives by construction — the model only generates the masked region, and that small section is surrounded by exact-copy context from the source. You can push inpaint noise to 1.0 and the loop still "sounds like the source" because most of it literally is.

In **a2a mode**, every sample is regenerated. The init_audio is mixed into the starting latent once (`latent * (1-σ) + noise * σ`) and from there the denoising trajectory is shaped *entirely* by the text conditioning. There is no mechanism that re-anchors the trajectory to init_audio along the way. Init_audio is a starting bias, not an anchor.

This asymmetry is the single most important architectural fact when designing a variation product on SA3.

### 2. The productive a2a noise band for tonal melodic content is 0.25–0.35.

The MLX runner's help says "0.4–0.8 typical for variation." That's correct for text-to-audio. For init_audio-driven variation of a tonal melodic loop *without an instrument-naming prompt*, the band collapses:

| σmax (cfg=2, no prompt) | Result |
|---|---|
| 0.25 | timbre almost identical, very small variation |
| 0.30 | timbre preserved, light variation |
| 0.35 | timbre preserved, real musical flourish — sweet spot |
| 0.45 | timbre starts drifting |
| 0.50+ | drifts to neighbouring tonal instruments (guitar / vibraphone / etc) |

**Why:** SA3 was trained as text→audio. At σ=0.5 the denoising trajectory has too much room and lands somewhere in the tonal-instrument neighbourhood of latent space. Without a text anchor naming the instrument, the trajectory is undisambiguated. Piano, electric piano, guitar, and vibraphone are all close cousins in this space.

With a strong text anchor (e.g. `"vintage electric piano riff"`), σ=0.50 + cfg=4 produces beautiful variations — but the text was doing the anchoring. It's easy to mis-attribute that to init_audio.

### 3. CFG is mode-specific.

- **a2a wants cfg=2.0.** Higher cfg amplifies generic text guidance (when no instrument is named, the text *is* generic), pulling the trajectory away from init_audio's timbre. cfg=1.0 (no guidance) actually drifts *more* because there's nothing keeping the trajectory inside the "musical loop" region of the conditional manifold.
- **Inpaint wants cfg=4.0.** Paste-back protects timbre structurally, so amplified text guidance affects only the masked region — where it usefully steers the rewrite toward the per-candidate musical direction.

Note: cfg=1.0 silently disables the unconditional branch — the negative prompt is ignored. Easy to miss.

### 4. End-of-loop inpaints drift; start/middle inpaints succeed.

For melodic loops, every `inpaint_last_bar` and `inpaint_last_2_bars` candidate failed in listening tests. Every `inpaint_bar_2`, `inpaint_middle_bar`, `inpaint_first_2_bars`, `inpaint_middle_2_bars` worked.

**Mechanism:** end-of-loop masks have only *left* context to anchor to. The model has to compose a "landing" with no future to resolve into, and the variation drifts. Start/middle masks have bilateral context — the rewrite is interpolating between two known regions instead of extrapolating into nothing.

Drums are exempt: end-of-loop *is* the right place for fills and turnarounds.

### 5. Inpaint masks need a full bar of room.

1-bar masks (≈2.67s at 90 BPM) consistently produced coherent musical variations. Half-bar masks (1.33s) produced "stitched transitions" — too short for the model to compose anything. 2-bar masks (5.33s) gave the most global musical change while still preserving the rest of the loop exactly.

Drums are again exempt — half-bar fills are valid for them.

### 6. Inpaint noise should cap at 0.95 for melodic.

At n=1.00 (full regen in the mask) the masked region has no echo of the original to anchor on, only prompt + neighbouring bars. With cfg=4 this consistently went feral. At n=0.85–0.95 the faint trace of the original keeps the rewrite *related*. n=0.85 was preferred over n=0.95 in direct comparison.

Drums tolerate the chaos at n=1.00.

### 7. Steers that worked, steers that didn't.

Per-candidate musical-direction nudges work when they describe operations SA3 has internalised:

| Steer | Verdict |
|---|---|
| `subtly different chord progression, same key` | gold — worked on both a2a and inpaint |
| `alternate chord voicing throughout, same key` | strong |
| `extra passing notes and ornaments` | strong on a2a |
| `small reharmonization while preserving the melodic contour` | strong on inpaint |
| `with a small fill at the end` | weak — SA3 doesn't seem to have a structural concept of "fill" |
| Anything referring to intro / outro / drop / build | untested but expected to be weak |

### 8. The friend's setup: cfg=1.0, n=0.52, unconditional.

A friend posted their setup on X / Twitter: empty prompt, `init_noise_level=0.52`, seeds 1–24, output duration matched to input. Their evaluation: *"keeps harmonic context and feel, varies on structure, timbre, instrumentation."*

This is literally the `diag_000` candidate in our diagnostic preset — the one we called "terrible" under the timbre-preservation goal. Under the *free variation* goal it's exactly correct. Same regime, opposite evaluation criterion.

That's the realisation that motivated the two-preset design. `--preset free` ships exactly the friend's setup.

The "matched to the duration" phrasing in their post likely just means they set `--seconds` equal to the input length. There's a weaker mechanism by which longer clips might tolerate higher noise (more `T_lat` positions → more attention context), but it's a secondary effect; the 0.52 value isn't derived from the duration by any formula we know of.

## Filename heuristic

Both the CLI and the studio derive BPM and key from the filename when not provided explicitly. Recognised patterns:

- `<bpm>_<key>_<rest>` (Noiiz / Loopmasters dominant) — `128_A_FifthsPluckSynth_718.wav`
- `<n>bpm` / `bpm<n>` tagged — `loop_124bpm_Am_drums.wav`, `120bpm_Em_pad.wav`
- Various separators (`_`, `-`, space, `.`)
- Keys: `A`–`G` with optional `#`/`b` and optional `m`/`min`/`minor`/`maj`/`major`

BPM is constrained to the 40–240 range to avoid false matches like `kick_440Hz.wav`.

## Presets reference

| Preset | Slots | Mode mix | Noise | CFG | Prompt | Use |
|---|---|---|---|---|---|---|
| `app` | 5 | 3 a2a + 2 inpaint | a2a 0.45*, inpaint 0.85 | a2a 4.0, inpaint 4.0 | per-candidate steer | "preserve sound" product |
| `free` | 5 | 5 a2a | 0.45* | 1.0 | empty | "free variation" product (default) |

\* a2a noise is `--noise-a2a` on the CLI / the slider in the studio. 0.45 is the default; the original friend's setup used 0.52 (see §8).
| `diagnose` | 4 | 4 a2a | 0.50 / 0.25 / 0.35 / 0.45 | 1.0 / 2.0 / 2.0 / 2.0 | neutral steer | isolate cfg vs noise as drift source |
| `grid` | `--count` | inpaint-dominant | melodic a2a 0.25–0.35, inpaint 0.85 | per-mode | per-candidate steer | exploration sweep |

## Files

```
sa3_variations.py        CLI harness
sa3_studio.py            Local web UI (drop WAV → 5 variations)
README_VARIATIONS.md     This file
runs/<name>/
  source_44100_stereo_s16.wav    converted input
  var_*.wav                      generated candidates
  manifest.jsonl                 full settings, seeds, paths, metrics
  review.html                    sortable listening page
```

## Limitations

- Inference-only. No fine-tuning / LoRA.
- The MLX runner accepts one inpaint range per command, so multi-region inpaints in a single pass are not used.
- Empirical findings are validated on melodic loops; drums / one-shots / SFX use sensible defaults but have not been deeply listening-tested.
- Triage metrics in `manifest.jsonl` (spectral cosine, envelope correlation, identity score) are heuristics, not perceptual quality measures. The ear is the final judge.
- BPM / key for the prompt are derived from the filename or `--bpm` / `--key`. No audio-based detection yet — a librosa one-liner would close this gap.
- `--preset app` is currently melodic-only. Drums / one-shots / SFX still go through `--preset grid`.
