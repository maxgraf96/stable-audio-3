#!/usr/bin/env python3
"""
SA3 variation harness for loops, drum loops, one-shots, and SFX.

This is intentionally inference-only. It drives Stability AI's Stable Audio 3
MLX runner on Apple Silicon, generates a grid of candidates, writes a JSONL
manifest, computes lightweight audio sanity/similarity metrics, and creates a
simple HTML review page.

Expected layout:
  stable-audio-3/
    optimized/mlx/sa3
  sa3_variations.py

Example:
  python sa3_variations.py \
    --sa3-root ~/code/stable-audio-3 \
    --input ~/samples/guitar_loop.wav \
    --kind melodic \
    --bpm 124 \
    --key "D minor" \
    --prompt "muted electric guitar riff, same amp and pedal tone" \
    --count 24 \
    --outdir ./runs/guitar_variations

Notes:
  - Requires ffmpeg for robust input conversion to the MLX runner's expected
    44.1 kHz / stereo / 16-bit PCM WAV format.
  - Uses the optimized/mlx CLI by default because it is the most practical path
    on a MacBook M4 Max.
  - Does not fine-tune or modify weights.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SAMPLE_RATE = 44100


@dataclass
class CandidateSpec:
    index: int
    mode: str
    kind: str
    seed: int
    prompt: str
    negative_prompt: str
    steer: str
    model_dit: str
    decoder: str
    seconds: float
    steps: int
    init_noise_level: float
    cfg: float
    apg: float
    inpaint_range: Optional[tuple[float, float]]
    out_path: str


@dataclass
class AudioMetrics:
    peak: float
    rms: float
    crest: float
    envelope_corr: float
    spectral_cosine: float
    centroid_ratio: float
    identity_score: float


_KEY_TOKEN_RE = re.compile(r"^([A-G])([#b]?)(m|min|minor|maj|major)?$", re.IGNORECASE)


def _match_key_token(part: str) -> Optional[str]:
    m = _KEY_TOKEN_RE.match(part)
    if not m:
        return None
    note = m.group(1).upper() + (m.group(2) or "")
    qual = (m.group(3) or "").lower()
    if qual in ("m", "min", "minor"):
        return f"{note} minor"
    if qual in ("maj", "major"):
        return f"{note} major"
    return note


def parse_bpm_key_from_filename(name: str) -> tuple[Optional[float], Optional[str]]:
    """Best-effort BPM/key extraction from common sample-library naming.

    Handles the dominant convention `<bpm>_<key>_<rest>` (Noiiz / Loopmasters /
    Splice older), and tagged variants (`<n>bpm`, `bpm<n>`). Key tokens are
    A–G with optional #/b and optional m/min/minor/maj/major qualifier.
    Returns (None, None) when nothing plausible is found.

    Examples:
      128_A_FifthsPluckSynth_718.wav       → (128, "A")
      90_C_BriarPatchVintagePiano_01.wav   → (90, "C")
      loop_124bpm_Am_drums.wav             → (124, "A minor")
      120bpm_Em_pad.wav                    → (120, "E minor")"""
    stem = Path(name).stem
    parts = re.split(r"[_\-\s.]+", stem)
    parts = [p for p in parts if p]

    bpm: Optional[float] = None
    bpm_idx = -1

    for i, p in enumerate(parts):
        m = re.fullmatch(r"(\d{2,3})bpm", p, re.IGNORECASE) or re.fullmatch(r"bpm(\d{2,3})", p, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 40 <= v <= 240:
                bpm, bpm_idx = float(v), i
                break
    if bpm is None:
        for i, p in enumerate(parts):
            if p.isdigit():
                v = int(p)
                if 40 <= v <= 240:
                    bpm, bpm_idx = float(v), i
                    break

    # Key: prefer the token immediately after the BPM token (the dominant
    # convention). Fall back to the token before it, then a global scan.
    key: Optional[str] = None
    if bpm_idx >= 0 and bpm_idx + 1 < len(parts):
        key = _match_key_token(parts[bpm_idx + 1])
    if key is None and bpm_idx >= 1:
        key = _match_key_token(parts[bpm_idx - 1])
    if key is None:
        for p in parts:
            key = _match_key_token(p)
            if key:
                break

    return bpm, key


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def run(cmd: list[str], cwd: Optional[Path] = None, dry_run: bool = False) -> subprocess.CompletedProcess:
    print("$", " ".join(shlex_quote(x) for x in cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_mlx_runner(sa3_root: Path) -> Path:
    runner = sa3_root / "optimized" / "mlx" / "sa3"
    require_file(runner, "SA3 MLX runner")
    return runner


def convert_to_mlx_wav(src: Path, dst: Path, dry_run: bool = False) -> Path:
    """Convert any ffmpeg-readable audio to 44.1k stereo 16-bit PCM WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Allow already-compatible WAVs when ffmpeg is missing.
        try:
            with wave.open(str(src), "rb") as w:
                ok = (
                    w.getframerate() == SAMPLE_RATE
                    and w.getnchannels() in (1, 2)
                    and w.getsampwidth() == 2
                )
        except wave.Error as exc:
            raise RuntimeError(
                "ffmpeg is required unless the input is already 44.1kHz 16-bit PCM WAV"
            ) from exc
        if not ok:
            raise RuntimeError(
                "ffmpeg is required to convert input to 44.1kHz stereo 16-bit PCM WAV"
            )
        if src.resolve() != dst.resolve() and not dry_run:
            shutil.copyfile(src, dst)
        return dst

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    run(cmd, dry_run=dry_run)
    return dst


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    """Read PCM WAV into float32 array with shape (channels, samples)."""
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported for metrics: {path}")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nch == 1:
        audio = data[None, :]
    else:
        audio = data.reshape(-1, nch).T
    return audio, sr


def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0)


def rms_envelope(x: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    x = mono(x)
    if len(x) < frame:
        return np.array([float(np.sqrt(np.mean(x * x) + 1e-12))], dtype=np.float32)
    n = 1 + (len(x) - frame) // hop
    env = np.empty(n, dtype=np.float32)
    win = np.hanning(frame).astype(np.float32)
    for i in range(n):
        seg = x[i * hop : i * hop + frame] * win
        env[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    return env


def mean_log_spectrum(x: np.ndarray, frame: int = 4096, hop: int = 2048) -> tuple[np.ndarray, float]:
    x = mono(x)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    win = np.hanning(frame).astype(np.float32)
    mags = []
    centroids = []
    freqs = np.fft.rfftfreq(frame, d=1.0 / SAMPLE_RATE)
    for start in range(0, max(1, len(x) - frame + 1), hop):
        seg = x[start : start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win)).astype(np.float32)
        mags.append(np.log1p(mag))
        denom = float(np.sum(mag) + 1e-9)
        centroids.append(float(np.sum(freqs * mag) / denom))
    spec = np.mean(np.stack(mags, axis=0), axis=0)
    return spec, float(np.mean(centroids))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = a[:n] - float(np.mean(a[:n]))
    b = b[:n] - float(np.mean(b[:n]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)


def compute_metrics(source_wav: Path, candidate_wav: Path) -> AudioMetrics:
    src, sr_src = read_wav_float(source_wav)
    cand, sr_cand = read_wav_float(candidate_wav)
    if sr_src != sr_cand:
        raise ValueError("source and candidate sample rates differ")

    cand_m = mono(cand)
    peak = float(np.max(np.abs(cand_m))) if cand_m.size else 0.0
    rms = float(np.sqrt(np.mean(cand_m * cand_m) + 1e-12)) if cand_m.size else 0.0
    crest = float(peak / max(rms, 1e-9))

    env_src = rms_envelope(src)
    env_cand = rms_envelope(cand)
    envelope_corr = max(0.0, corr(env_src, env_cand))

    spec_src, cen_src = mean_log_spectrum(src)
    spec_cand, cen_cand = mean_log_spectrum(cand)
    spectral_cosine = max(0.0, cosine(spec_src, spec_cand))
    centroid_ratio = float(min(cen_src, cen_cand) / max(cen_src, cen_cand, 1e-9))

    # First-pass identity estimate: spectral similarity dominates, with envelope
    # and centroid sanity checks. This is not a perceptual metric; it is a cheap
    # triage score to sort a review page.
    identity_score = float(
        0.60 * spectral_cosine + 0.25 * envelope_corr + 0.15 * max(0.0, centroid_ratio)
    )
    return AudioMetrics(
        peak=peak,
        rms=rms,
        crest=crest,
        envelope_corr=float(envelope_corr),
        spectral_cosine=float(spectral_cosine),
        centroid_ratio=float(centroid_ratio),
        identity_score=identity_score,
    )


def steer_prompts(kind: str) -> list[str]:
    """Per-candidate musical-direction nudges. Cycled across the candidate grid."""
    if kind == "drum":
        return [
            "subtle ghost notes added between hits",
            "syncopated hi-hat variation, same kit",
            "rolled snare fill in the last beat",
            "alternate kick pattern, same kit",
            "open hi-hat accent on the offbeat",
            "tom fill toward the end",
            "tighter groove with subtle swing",
            "ghost snare hits added",
        ]
    if kind == "oneshot":
        return [
            "alternate take with slightly different attack",
            "same source with subtle pitch inflection",
            "slightly longer decay, same body",
            "softer transient, same character",
            "harder hit variant of the same sample",
        ]
    if kind == "sfx":
        return [
            "alternate take of the same source",
            "subtle texture variation, same space",
            "different intensity, same source",
            "longer tail variation, same room",
            "different angle on the same source",
        ]
    # melodic default — biased toward harmonic restatement, which is what SA3
    # actually does well empirically. "Fill at the end" was dropped because SA3
    # doesn't seem to have a strong notion of fills as a structural concept.
    return [
        "subtly different chord progression, same key",
        "alternate chord voicing throughout, same key",
        "small reharmonization while preserving the melodic contour",
        "small melodic variation with different note choices",
        "alternate phrasing, same harmonic feel",
        "extra passing notes and ornaments",
        "different chord voicing, same progression",
        "syncopated rhythm with same melodic contour",
    ]


def prompt_for_kind(kind: str, user_prompt: str, bpm: Optional[float], key: Optional[str], steer: str) -> str:
    bpm_txt = f", {bpm:g} BPM" if bpm else ""
    key_txt = f", in {key}" if key else ""
    user = user_prompt.strip()

    if kind == "drum":
        base = (
            "TrackType: Instrument, drum loop, same drum kit, same room, "
            "same microphones, same recording chain"
        )
    elif kind == "oneshot":
        base = (
            "TrackType: SFX, one-shot sample, same source sound, same timbre, "
            "same transient character, same recording chain"
        )
    elif kind == "sfx":
        base = (
            "TrackType: SFX, same source material, same texture, "
            "same space, same microphone perspective, same processing"
        )
    else:
        base = (
            "TrackType: Instrument, musical loop, same instrument tone, "
            "same performance style, same recording chain"
        )

    parts = [base + bpm_txt + key_txt]
    if user:
        parts.append(user)
    if steer:
        parts.append(steer)
    return ", ".join(parts)


def negative_prompt_for_kind(kind: str) -> str:
    common = (
        "poor quality, distorted, clipping, noisy, vocals, speech, silence, "
        "abrupt cutoff, completely different instrument, different timbre, different tone"
    )
    if kind == "drum":
        return common + ", changed drum kit, different room, weak transient"
    if kind == "oneshot":
        return common + ", long musical phrase, background music, reverb wash"
    if kind == "sfx":
        return common + ", background music, unrelated object, different source material"
    return common + ", different genre, unrelated melody, sudden style change"


def quality_defaults(quality: str) -> tuple[str, str, int]:
    """Return (default_dit, default_decoder, default_steps) for a quality preset."""
    if quality == "fast":
        return ("sm-music", "same-s", 8)
    if quality == "best":
        return ("medium", "same-l", 16)
    # "good" — recommended on M4 Max
    return ("medium", "same-l", 12)


def choose_model(kind: str, requested: str, quality: str) -> tuple[str, str]:
    if requested == "auto":
        if quality == "fast":
            requested = "sm-sfx" if kind in ("oneshot", "sfx") else "sm-music"
        else:
            requested = "medium"
    if requested in ("sm-music", "sm-sfx"):
        return requested, "same-s"
    if requested == "medium":
        return "medium", "same-l"
    raise ValueError(f"unknown --dit {requested}")


def noise_grid_a2a(kind: str) -> list[float]:
    """Audio-to-audio noise band. The melodic band depends critically on
    whether the prompt names the instrument. With a strong text anchor
    (e.g. "vintage electric piano"), n=0.50 at cfg=4 stays timbre-stable
    (the "014 regime"). Without an instrument name, the productive band
    collapses to n=0.25–0.35 at cfg=2: above 0.40 the denoising trajectory
    has too much room and drifts to neighbouring tonal-instrument classes
    (guitar / vibraphone / etc). Drums/SFX live in tighter latent
    neighbourhoods and tolerate higher noise."""
    if kind == "oneshot":
        return [0.20, 0.30, 0.45, 0.60]
    if kind == "sfx":
        return [0.30, 0.45, 0.60, 0.75]
    if kind == "drum":
        return [0.45, 0.55, 0.65, 0.75]
    return [0.25, 0.30, 0.35]


def noise_grid_inpaint(kind: str) -> list[float]:
    """Inpaint noise band. Paste-back preserves the unmasked region, so timbre
    survives by construction. For melodic, n=0.95 went feral consistently
    (014, 001, 006 all bad) — capped at 0.85. Drums tolerate the chaos."""
    if kind == "oneshot":
        return [0.50, 0.70, 0.90]
    if kind == "sfx":
        return [0.70, 0.85, 1.00]
    if kind == "drum":
        return [0.85, 0.95, 1.00]
    return [0.85]


def clamp_range(start: float, end: float, duration: float) -> Optional[tuple[float, float]]:
    start = max(0.0, min(float(start), duration))
    end = max(0.0, min(float(end), duration))
    if end - start < 0.08:
        return None
    # Don't round here. The runner validates end <= args.seconds, and any
    # rounding that nudges end above the (often non-decimal) duration causes
    # the runner to reject the range. mlx_command formats with :.6f.
    return (start, end)


def bar_len_seconds(bpm: Optional[float], beats_per_bar: int) -> Optional[float]:
    if not bpm or bpm <= 0:
        return None
    return 60.0 / bpm * beats_per_bar


def mask_recipes(kind: str, duration: float, bpm: Optional[float], beats_per_bar: int) -> list[tuple[str, Optional[tuple[float, float]]]]:
    bar = bar_len_seconds(bpm, beats_per_bar)
    recipes: list[tuple[str, Optional[tuple[float, float]]]] = []

    # Full-loop audio-to-audio.
    recipes.append(("a2a", None))

    if kind == "oneshot":
        # One-shot: keep attack where possible, vary body/tail.
        recipes.append(("inpaint_tail", clamp_range(duration * 0.35, duration, duration)))
        recipes.append(("inpaint_decay", clamp_range(duration * 0.55, duration, duration)))
        return recipes

    if kind == "sfx":
        recipes.append(("inpaint_middle", clamp_range(duration * 0.25, duration * 0.75, duration)))
        recipes.append(("inpaint_tail", clamp_range(duration * 0.60, duration, duration)))
        return recipes

    if bar and duration >= bar * 1.5:
        # Bar-aware loop edits.
        if duration >= bar * 4:
            recipes.append(("inpaint_bar_2", clamp_range(bar, bar * 2, duration)))
            recipes.append(("inpaint_middle_bar", clamp_range(duration * 0.5 - bar * 0.5, duration * 0.5 + bar * 0.5, duration)))
        if kind == "drum":
            # Drums: end-of-loop fills work — turnarounds and last-bar fills
            # are an important variation surface.
            recipes.append(("inpaint_last_bar", clamp_range(duration - bar, duration, duration)))
            recipes.append(("inpaint_last_half_bar", clamp_range(duration - bar * 0.5, duration, duration)))
            if duration >= bar * 2:
                recipes.append(("inpaint_turnaround", clamp_range(duration - bar * 1.25, duration - bar * 0.25, duration)))
        else:
            # Melodic: end-of-loop edits drift consistently — the model has no
            # right-side context to resolve into. Stick to start/middle masks,
            # including 2-bar windows for longer composition room.
            if duration >= bar * 4:
                recipes.append(("inpaint_first_2_bars", clamp_range(0.0, bar * 2, duration)))
                recipes.append(("inpaint_middle_2_bars", clamp_range(duration * 0.5 - bar, duration * 0.5 + bar, duration)))
    else:
        # Fallback for unknown BPM.
        recipes.append(("inpaint_middle", clamp_range(duration * 0.35, duration * 0.65, duration)))
        if kind == "drum":
            recipes.append(("inpaint_last_quarter", clamp_range(duration * 0.75, duration, duration)))
            recipes.append(("inpaint_second_half", clamp_range(duration * 0.50, duration, duration)))
        else:
            recipes.append(("inpaint_first_half", clamp_range(0.0, duration * 0.5, duration)))

    return [(name, rng) for name, rng in recipes if rng is not None or name == "a2a"]


def build_candidates(
    kind: str,
    duration: float,
    count: int,
    seed: int,
    base_prompt_builder,
    negative_prompt: str,
    bpm: Optional[float],
    beats_per_bar: int,
    dit: str,
    decoder: str,
    outdir: Path,
    steps: int,
    cfg_a2a: float,
    cfg_inpaint: float,
    apg: float,
) -> list[CandidateSpec]:
    a2a_noises = noise_grid_a2a(kind)
    inpaint_noises = noise_grid_inpaint(kind)
    masks = mask_recipes(kind, duration, bpm, beats_per_bar)
    steers = steer_prompts(kind)

    # Recipe pool: (mode, mask_or_None, noise).
    a2a_pool: list[tuple[str, Optional[tuple[float, float]], float]] = []
    inpaint_pool: list[tuple[str, Optional[tuple[float, float]], float]] = []
    for mode, mask in masks:
        if mode == "a2a":
            for n in a2a_noises:
                a2a_pool.append((mode, None, n))
        else:
            for n in inpaint_noises:
                inpaint_pool.append((mode, mask, n))

    rng = random.Random(seed)
    rng.shuffle(a2a_pool)
    rng.shuffle(inpaint_pool)

    # For melodic: a2a at the right noise (~0.50) produces some of the best
    # variations — gentle global reharmonizations rather than per-bar rewrites.
    # Interleave 2:1 (inpaint:a2a) so the user gets a mix of "rewrite this bar"
    # and "subtly restate the whole loop" early in the grid.
    # For drum: inpaint-dominant (fills + groove variants are the win).
    # For oneshot/sfx: a2a-first since bar-aware masks don't apply.
    if kind == "melodic" and a2a_pool and inpaint_pool:
        pool: list[tuple[str, Optional[tuple[float, float]], float]] = []
        i_idx = a_idx = 0
        while i_idx < len(inpaint_pool) or a_idx < len(a2a_pool):
            for _ in range(2):
                if i_idx < len(inpaint_pool):
                    pool.append(inpaint_pool[i_idx])
                    i_idx += 1
            if a_idx < len(a2a_pool):
                pool.append(a2a_pool[a_idx])
                a_idx += 1
    elif kind == "drum" and inpaint_pool:
        pool = inpaint_pool + a2a_pool
    else:
        pool = a2a_pool + inpaint_pool

    candidates: list[CandidateSpec] = []
    for i in range(count):
        mode, mask, noise = pool[i % len(pool)]
        steer = steers[i % len(steers)]
        cand_seed = seed + i * 101 + rng.randint(0, 100)
        prompt = base_prompt_builder(steer)
        safe_mode = mode.replace("/", "_")
        out_path = outdir / f"var_{i:03d}_{kind}_{safe_mode}_n{noise:.2f}_s{cand_seed}.wav"
        cand_cfg = cfg_a2a if mode == "a2a" else cfg_inpaint
        candidates.append(
            CandidateSpec(
                index=i,
                mode=mode,
                kind=kind,
                seed=cand_seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steer=steer,
                model_dit=dit,
                decoder=decoder,
                seconds=duration,
                steps=steps,
                init_noise_level=float(noise),
                cfg=float(cand_cfg),
                apg=float(apg),
                inpaint_range=mask,
                out_path=str(out_path),
            )
        )
    return candidates


def build_app_preset(
    kind: str,
    duration: float,
    seed: int,
    base_prompt_builder,
    negative_prompt: str,
    bpm: Optional[float],
    beats_per_bar: int,
    dit: str,
    decoder: str,
    outdir: Path,
    steps: int,
    cfg_a2a: float,
    cfg_inpaint: float,
    apg: float,
) -> list[CandidateSpec]:
    """Five-shot preset matching the product shape:
      3 × a2a (global reharmonization) at n=0.50/0.52/0.55
      2 × inpaint (sectional rewrite) at n=0.85, middle-bar and first-2-bars

    All five configurations sit in empirically validated sweet spots from the
    earlier exploration sweeps. Melodic only for now — drum/oneshot/sfx have
    not been validated with the same depth."""
    if kind != "melodic":
        raise SystemExit(
            "error: --preset app is currently tuned for --kind melodic only. "
            "Use --preset grid for drum / oneshot / sfx."
        )

    bar = bar_len_seconds(bpm, beats_per_bar)
    if bar and duration >= bar * 4:
        middle_bar = clamp_range(duration * 0.5 - bar * 0.5, duration * 0.5 + bar * 0.5, duration)
        first_2_bars = clamp_range(0.0, bar * 2, duration)
    else:
        # Short loop or unknown BPM — fall back to fractional masks so the
        # preset still produces 5 candidates.
        middle_bar = clamp_range(duration * 0.35, duration * 0.65, duration)
        first_2_bars = clamp_range(0.0, duration * 0.5, duration)

    slots: list[tuple[str, Optional[tuple[float, float]], float, str]] = [
        # a2a noise band tightened to 0.27–0.35 after the diagnose preset
        # confirmed n>=0.45 drifts timbre on tonal content without a
        # named-instrument prompt. n=0.35 was the empirical sweet spot
        # ("small flourish, timbre intact").
        ("a2a", None, 0.27, "extra passing notes and ornaments"),
        ("a2a", None, 0.31, "alternate phrasing, same harmonic feel"),
        ("a2a", None, 0.35, "small melodic variation with different note choices"),
        ("inpaint_middle_bar", middle_bar, 0.85, "subtly different chord progression, same key"),
        ("inpaint_first_2_bars", first_2_bars, 0.85, "small reharmonization while preserving the melodic contour"),
    ]

    candidates: list[CandidateSpec] = []
    for i, (mode, mask, noise, steer) in enumerate(slots):
        cand_seed = seed + i * 1009
        prompt = base_prompt_builder(steer)
        safe_mode = mode.replace("/", "_")
        out_path = outdir / f"var_{i:03d}_{kind}_{safe_mode}_n{noise:.2f}_s{cand_seed}.wav"
        cand_cfg = cfg_a2a if mode == "a2a" else cfg_inpaint
        candidates.append(
            CandidateSpec(
                index=i,
                mode=mode,
                kind=kind,
                seed=cand_seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steer=steer,
                model_dit=dit,
                decoder=decoder,
                seconds=duration,
                steps=steps,
                init_noise_level=float(noise),
                cfg=float(cand_cfg),
                apg=float(apg),
                inpaint_range=mask,
                out_path=str(out_path),
            )
        )
    return candidates


def build_free_preset(
    kind: str,
    duration: float,
    seed: int,
    dit: str,
    decoder: str,
    outdir: Path,
    steps: int,
) -> list[CandidateSpec]:
    """Five-shot 'free variation' preset — the unconditional / no-prompt regime.

    All a2a, empty prompt, cfg=1.0, n=0.52, only seeds vary. This is the same
    setup as diag_000 in the diagnose preset (which scored 'terrible' under
    the timbre-preservation target) — but evaluated under a different goal:
    keep the harmonic context and feel, let timbre / instrumentation drift.
    Complement to --preset app for when the user wants 'fits in this slot'
    rather than 'same sample, varied'."""
    candidates: list[CandidateSpec] = []
    for i in range(5):
        cand_seed = seed + i * 1009
        out_path = outdir / f"var_{i:03d}_{kind}_free_n0.52_s{cand_seed}.wav"
        candidates.append(
            CandidateSpec(
                index=i,
                mode="a2a",
                kind=kind,
                seed=cand_seed,
                prompt="",
                negative_prompt="",
                steer="free variation (unconditional)",
                model_dit=dit,
                decoder=decoder,
                seconds=duration,
                steps=steps,
                init_noise_level=0.52,
                cfg=1.0,
                apg=1.0,
                inpaint_range=None,
                out_path=str(out_path),
            )
        )
    return candidates


def build_diagnose_preset(
    kind: str,
    duration: float,
    seed: int,
    base_prompt_builder,
    negative_prompt: str,
    dit: str,
    decoder: str,
    outdir: Path,
    steps: int,
    apg: float,
) -> list[CandidateSpec]:
    """Four-shot diagnostic to isolate the cause of a2a timbre drift on tonal
    melodic content. All four are a2a with the same steer — only cfg and noise
    vary, so listening-test results map cleanly to one hypothesis:

      0: cfg=1.0, n=0.50 — no CFG amplification. If timbre drifts here, σ=0.50
         alone is too denatured for tonal init_audio; if stable, text pull is
         the drift source and we need auto-tagging or a named instrument.
      1: cfg=2.0, n=0.25 — low noise, light guidance. Closer to original latent
         leaves less room for the trajectory to wander.
      2: cfg=2.0, n=0.35 — same idea, more variation room.
      3: cfg=2.0, n=0.45 — back near the band we know was previously musical."""
    if kind != "melodic":
        raise SystemExit(
            "error: --preset diagnose targets melodic-only timbre drift; "
            "use --preset grid for other kinds."
        )

    neutral_steer = "small melodic variation with different note choices"
    slots: list[tuple[str, Optional[tuple[float, float]], float, float, str]] = [
        ("a2a", None, 0.50, 1.0, neutral_steer),
        ("a2a", None, 0.25, 2.0, neutral_steer),
        ("a2a", None, 0.35, 2.0, neutral_steer),
        ("a2a", None, 0.45, 2.0, neutral_steer),
    ]

    candidates: list[CandidateSpec] = []
    for i, (mode, mask, noise, cfg, steer) in enumerate(slots):
        cand_seed = seed + i * 1009
        prompt = base_prompt_builder(steer)
        safe_mode = mode.replace("/", "_")
        out_path = outdir / f"diag_{i:03d}_{kind}_{safe_mode}_n{noise:.2f}_cfg{cfg:.1f}_s{cand_seed}.wav"
        candidates.append(
            CandidateSpec(
                index=i,
                mode=mode,
                kind=kind,
                seed=cand_seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steer=steer,
                model_dit=dit,
                decoder=decoder,
                seconds=duration,
                steps=steps,
                init_noise_level=float(noise),
                cfg=float(cfg),
                apg=float(apg),
                inpaint_range=mask,
                out_path=str(out_path),
            )
        )
    return candidates


def mlx_command(runner: Path, input_wav: Path, spec: CandidateSpec) -> list[str]:
    cmd = [
        str(runner),
        "--prompt",
        spec.prompt,
        "--negative-prompt",
        spec.negative_prompt,
        "--dit",
        spec.model_dit,
        "--decoder",
        spec.decoder,
        "--seconds",
        f"{spec.seconds:.6f}",
        "--steps",
        str(spec.steps),
        "--seed",
        str(spec.seed),
        "--cfg",
        f"{spec.cfg:.3f}",
        "--apg",
        f"{spec.apg:.3f}",
        "--init-audio",
        str(input_wav),
        "--init-noise-level",
        f"{spec.init_noise_level:.6f}",
        "--out",
        str(Path(spec.out_path).resolve()),
    ]
    if spec.inpaint_range is not None:
        start, end = spec.inpaint_range
        # Match the runner's --seconds precision (:.6f). Using :.3f rounds the
        # end value up past --seconds and the runner rejects the range.
        cmd.extend(["--inpaint-range", f"{start:.6f},{end:.6f}"])
    return cmd


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def relative_url(target: Path, base_dir: Path) -> str:
    try:
        rel = os.path.relpath(target, start=base_dir)
    except ValueError:
        rel = str(target)
    return Path(rel).as_posix()


def write_review_html(outdir: Path, source_wav: Path, rows: list[dict]) -> Path:
    html_path = outdir / "review.html"
    sorted_rows = sorted(rows, key=lambda r: r.get("metrics", {}).get("identity_score", -1), reverse=True)
    trs = []
    for row in sorted_rows:
        spec = row["spec"]
        metrics = row.get("metrics", {})
        wav_path = Path(spec["out_path"])
        audio_src = html.escape(relative_url(wav_path, outdir))
        mask = spec.get("inpaint_range")
        mask_txt = "-" if mask is None else f"{mask[0]:.2f}–{mask[1]:.2f}s"
        steer = spec.get("steer", "")
        cfg = spec.get("cfg", 1.0)
        trs.append(
            "<tr>"
            f"<td>{spec['index']:03d}</td>"
            f"<td>{html.escape(spec['mode'])}</td>"
            f"<td>{spec['init_noise_level']:.2f}</td>"
            f"<td>{cfg:.1f}</td>"
            f"<td>{mask_txt}</td>"
            f"<td>{html.escape(steer)}</td>"
            f"<td>{spec['seed']}</td>"
            f"<td>{metrics.get('identity_score', 0):.3f}</td>"
            f"<td>{metrics.get('spectral_cosine', 0):.3f}</td>"
            f"<td>{metrics.get('envelope_corr', 0):.3f}</td>"
            f"<td>{metrics.get('peak', 0):.3f}</td>"
            f"<td><audio controls preload='none' src='{audio_src}'></audio></td>"
            "</tr>"
        )
    source_rel = html.escape(relative_url(source_wav, outdir))
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SA3 variation review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: white; }}
    audio {{ width: 260px; }}
    code {{ background: #f5f5f5; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>SA3 variation review</h1>
  <h2>Source</h2>
  <p><code>{html.escape(str(source_wav))}</code></p>
  <audio controls src="{source_rel}"></audio>
  <h2>Candidates</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>mode</th><th>noise</th><th>cfg</th><th>mask</th><th>steer</th><th>seed</th>
        <th>identity</th><th>spectral</th><th>envelope</th><th>peak</th><th>audio</th>
      </tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")
    return html_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate Stable Audio 3 sample variations on Apple Silicon via MLX")
    ap.add_argument("--sa3-root", type=Path, required=True, help="Path to cloned Stability-AI/stable-audio-3 repo")
    ap.add_argument("--input", type=Path, required=True, help="Input sample/loop/one-shot")
    ap.add_argument("--outdir", type=Path, required=True, help="Output run directory")
    ap.add_argument(
        "--kind",
        choices=["melodic", "drum", "oneshot", "sfx"],
        default="melodic",
        help="Input type; affects prompts, model choice, noise grid, masks",
    )
    ap.add_argument("--prompt", default="", help="Extra source description to append to the generated prompt")
    ap.add_argument("--negative-prompt", default=None, help="Override negative prompt")
    ap.add_argument("--bpm", type=float, default=None, help="BPM for bar-aware masks")
    ap.add_argument("--key", default=None, help="Musical key text, e.g. 'D minor'")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument(
        "--preset",
        choices=["grid", "app", "free", "diagnose"],
        default="grid",
        help="grid (default): exploration sweep of --count candidates. "
             "app: 5-shot 'preserve sound' preset (3 a2a + 2 inpaint, melodic only). "
             "free: 5-shot 'free variation' preset (a2a unconditional at n=0.52, seeds vary; "
             "preserves harmonic context, lets timbre/instrument drift). "
             "diagnose: 4-shot a2a-only set isolating cfg vs noise. "
             "app/free/diagnose ignore --count.",
    )
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--quality",
        choices=["fast", "good", "best"],
        default="good",
        help="fast = sm-music + same-s, 8 steps. good = medium + same-l, 12 steps (default, M4 Max-friendly). best = medium + same-l, 16 steps.",
    )
    ap.add_argument("--steps", type=int, default=None, help="Override sampler steps; default comes from --quality")
    ap.add_argument("--dit", default="auto", choices=["auto", "sm-music", "sm-sfx", "medium"], help="Override the DiT picked by --quality")
    ap.add_argument(
        "--cfg-a2a",
        type=float,
        default=2.0,
        help="CFG scale for audio-to-audio candidates. Lower than --cfg-inpaint "
             "because a2a has no paste-back: every sample is regenerated, so "
             "the init_audio latent has to anchor timbre on its own. High CFG "
             "(~4) amplifies generic text guidance and pulls timbre away from "
             "the source when no instrument-specific prompt is provided.",
    )
    ap.add_argument(
        "--cfg-inpaint",
        type=float,
        default=4.0,
        help="CFG scale for inpaint candidates. Higher than --cfg-a2a because "
             "paste-back preserves the unmasked region exactly, so timbre is "
             "anchored structurally — high CFG just makes the masked region "
             "follow the steer more strongly without risking the whole loop.",
    )
    ap.add_argument(
        "--apg",
        type=float,
        default=1.0,
        help="Adaptive Projected Guidance [0..1]. Only matters when --cfg != 1.0. 1.0 = full APG (anti-saturation at high CFG).",
    )
    ap.add_argument("--duration", type=float, default=None, help="Override output duration; default=input duration")
    ap.add_argument("--dry-run", action="store_true", help="Print commands and write manifest without running SA3")
    ap.add_argument("--skip-existing", action="store_true", help="Do not regenerate WAVs that already exist")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.sa3_root = args.sa3_root.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    require_file(args.input, "input audio")
    runner = find_mlx_runner(args.sa3_root)
    mlx_dir = runner.parent

    if args.bpm is None or args.key is None:
        fn_bpm, fn_key = parse_bpm_key_from_filename(args.input.name)
        if args.bpm is None and fn_bpm is not None:
            print(f"[bpm] inferred {fn_bpm:g} from filename {args.input.name!r}", file=sys.stderr)
            args.bpm = fn_bpm
        if args.key is None and fn_key is not None:
            print(f"[key] inferred {fn_key!r} from filename {args.input.name!r}", file=sys.stderr)
            args.key = fn_key

    prepared = args.outdir / "source_44100_stereo_s16.wav"
    convert_to_mlx_wav(args.input, prepared, dry_run=args.dry_run)
    if args.dry_run and not prepared.exists():
        # For dry run without conversion, use a sane placeholder duration.
        duration = args.duration or 8.0
    else:
        duration = float(args.duration) if args.duration else wav_duration_seconds(prepared)

    _, _, q_steps = quality_defaults(args.quality)
    dit, decoder = choose_model(args.kind, args.dit, args.quality)
    # When --dit is forced to a small model, fall back to its matching decoder.
    if args.dit != "auto" and args.dit != "medium":
        decoder = "same-s"
    steps = args.steps if args.steps is not None else q_steps
    negative_prompt = args.negative_prompt or negative_prompt_for_kind(args.kind)

    def base_prompt_builder(steer: str) -> str:
        return prompt_for_kind(args.kind, args.prompt, args.bpm, args.key, steer)

    if args.cfg_a2a == 1.0 or args.cfg_inpaint == 1.0:
        print(
            "warning: cfg=1.0 disables the unconditional branch, so the negative prompt is "
            "ignored and the model has no guidance for the affected mode.",
            file=sys.stderr,
        )

    if args.preset == "free":
        candidates = build_free_preset(
            kind=args.kind,
            duration=duration,
            seed=args.seed,
            dit=dit,
            decoder=decoder,
            outdir=args.outdir,
            steps=steps,
        )
    elif args.preset == "diagnose":
        candidates = build_diagnose_preset(
            kind=args.kind,
            duration=duration,
            seed=args.seed,
            base_prompt_builder=base_prompt_builder,
            negative_prompt=negative_prompt,
            dit=dit,
            decoder=decoder,
            outdir=args.outdir,
            steps=steps,
            apg=args.apg,
        )
    elif args.preset == "app":
        candidates = build_app_preset(
            kind=args.kind,
            duration=duration,
            seed=args.seed,
            base_prompt_builder=base_prompt_builder,
            negative_prompt=negative_prompt,
            bpm=args.bpm,
            beats_per_bar=args.beats_per_bar,
            dit=dit,
            decoder=decoder,
            outdir=args.outdir,
            steps=steps,
            cfg_a2a=args.cfg_a2a,
            cfg_inpaint=args.cfg_inpaint,
            apg=args.apg,
        )
    else:
        candidates = build_candidates(
            kind=args.kind,
            duration=duration,
            count=args.count,
            seed=args.seed,
            base_prompt_builder=base_prompt_builder,
            negative_prompt=negative_prompt,
            bpm=args.bpm,
            beats_per_bar=args.beats_per_bar,
            dit=dit,
            decoder=decoder,
            outdir=args.outdir,
            steps=steps,
            cfg_a2a=args.cfg_a2a,
            cfg_inpaint=args.cfg_inpaint,
            apg=args.apg,
        )

    rows: list[dict] = []
    t0 = time.time()
    for spec in candidates:
        out_path = Path(spec.out_path)
        if args.skip_existing and out_path.exists():
            print(f"Skipping existing {out_path}")
        else:
            cmd = mlx_command(runner, prepared, spec)
            run(cmd, cwd=mlx_dir, dry_run=args.dry_run)

        row = {"spec": asdict(spec), "command": mlx_command(runner, prepared, spec)}
        if out_path.exists() and not args.dry_run:
            try:
                row["metrics"] = asdict(compute_metrics(prepared, out_path))
            except Exception as exc:
                row["metrics_error"] = str(exc)
        rows.append(row)
        write_jsonl(args.outdir / "manifest.jsonl", rows)

    if not args.dry_run:
        review = write_review_html(args.outdir, prepared, rows)
    else:
        review = args.outdir / "review.html"

    elapsed = time.time() - t0
    print()
    print(f"Wrote manifest: {args.outdir / 'manifest.jsonl'}")
    if not args.dry_run:
        print(f"Wrote review:   {review}")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
