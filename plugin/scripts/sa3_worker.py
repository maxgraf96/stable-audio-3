#!/usr/bin/env python3
"""Resident inference worker for the JUCE plugin's WorkerBackend.

On Apple Silicon the plugin runs SA3 in-process through the C++ MLX
orchestrator. Off Apple there is no such path — MLX is Metal-only — so the
plugin spawns this instead and keeps it warm for the life of the app.

Protocol: newline-delimited JSON, requests on stdin, events on stdout. One
object per line, no framing, no partial writes (every emit flushes). The
plugin reads events until it sees a terminal one, so every command must end in
exactly one of `ready`, `memory`, `done`, or `error` — a command that can fail
silently would hang the UI thread waiting for a line that never comes.

    -> {"cmd":"probe"}
    <- {"event":"memory","totalGB":12.8,...,"defaultKind":0}

    -> {"cmd":"load","model":"medium","seconds":8.0}
    <- {"event":"ready","loadSeconds":16.4}

    -> {"cmd":"generate","src":"C:/.../source.wav","preset":"free",...}
    <- {"event":"status","message":"Loading SA3 Medium (10.7s loop)..."}   (optional)
    <- {"event":"candidate","idx":0,"path":"C:/.../var_000.wav","steer":"...","mode":"a2a"}
    <- ... one per candidate, emitted as each finishes ...
    <- {"event":"done","count":5,"elapsed":4.9}

`status` is progress, not a reply: it may appear any number of times before a
terminal event and the plugin forwards it to the UI's status line.

    -> {"cmd":"quit"}

Candidates stream so the plugin fills its slots progressively, matching what
run_variations' per-candidate callback gives the Apple path. The preset specs
come from sa3_variations' own builders rather than being restated here, so the
two backends stay in step by construction.

stdout is the protocol channel and nothing else may touch it: the pipeline and
its dependencies log to stderr, and sys.stdout is redirected below to make an
accidental print() from a library harmless rather than protocol corruption.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# Two layouts have to import sa3_variations / sa3_pipeline*:
#
#   dev       <repo>/plugin/scripts/sa3_worker.py  — siblings live at parents[1]
#   installed <app>/app/sa3_worker.py              — siblings live right here
#
# Adding both covers each without having to detect which one we're in. The
# entry that doesn't apply is simply a directory with nothing to import.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Bind the real stdout before anything else can grab it, then point sys.stdout
# at stderr. Every protocol write goes through _emit and the fd it captured.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

# Mirrors sa3plugin::ModelKind in plugin/Source/InferenceBackend.h.
KIND_MEDIUM, KIND_SMALL_MUSIC, KIND_SMALL_SFX = 0, 1, 2

_DISPLAY_NAME = {
    "medium": "SA3 Medium",
    "sm-music": "SA3 Small Music",
    "sm-sfx": "SA3 Small SFX",
}

_MODEL_TO_DIT = {
    "medium": ("medium", "same-l"),
    "sm-music": ("sm-music", "same-s"),
    "sm-sfx": ("sm-sfx", "same-s"),
}

# SA3 Medium measured at ~9.3 GB peak VRAM for an 8 s loop on a 12 GB card
# (fp16 weights plus decode working set). 10.5 GB is that peak with enough
# margin to survive a desktop compositor and a browser also holding VRAM;
# below it the small models are the honest recommendation. This is deliberately
# not the Mac's 11 GB unified-memory number — that budget covers weights *and*
# working set in one pool, which is a different question.
MEDIUM_MIN_GB = 10.5
MEDIUM_COMFORTABLE_GB = 14.0


def _emit(obj: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(obj) + "\n")
    _PROTOCOL_OUT.flush()


def _log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


class Worker:
    def __init__(self) -> None:
        self.pipeline = None
        self.pipeline_key: Optional[tuple] = None
        self.model_name = "medium"

    # ── probe ────────────────────────────────────────────────────────
    def probe(self) -> None:
        """Report device memory. Never raises — a probe failure must still
        produce a usable profile or the UI has nothing to show."""
        total_gb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            else:
                total_gb = self._system_ram_gb()
        except Exception as exc:  # noqa: BLE001
            _log(f"probe fell back to system RAM: {exc}")
            total_gb = self._system_ram_gb()

        supported = total_gb >= MEDIUM_MIN_GB
        _emit({
            "event": "memory",
            "totalGB": total_gb,
            "workingSetGB": total_gb,
            "mediumSupported": supported,
            "mediumComfortable": total_gb >= MEDIUM_COMFORTABLE_GB,
            "simulated": False,
            "defaultKind": KIND_MEDIUM if supported else KIND_SMALL_MUSIC,
        })

    @staticmethod
    def _system_ram_gb() -> float:
        try:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1e9
        except Exception:  # noqa: BLE001
            return 0.0

    # ── load ─────────────────────────────────────────────────────────
    def load(self, req: dict) -> None:
        model = str(req.get("model", "medium"))
        if model not in _MODEL_TO_DIT:
            raise ValueError(f"unknown model {model!r}; expected one of {list(_MODEL_TO_DIT)}")
        seconds = float(req.get("seconds", 8.0))
        self.model_name = model
        self._ensure_pipeline(model, seconds)
        _emit({"event": "ready", "loadSeconds": self._last_load_seconds})

    def _retarget(self, seconds: float) -> bool:
        """True if the resident pipeline can serve `seconds` without a rebuild."""
        p = self.pipeline
        if abs(float(p.seconds) - float(seconds)) < 1e-3:
            return True
        setter = getattr(p, "set_duration", None)
        if setter is None:
            return False        # MLX sizes its DiT buffers to T_lat — must rebuild.
        setter(seconds)
        return True

    def _ensure_pipeline(self, model: str, seconds: float) -> None:
        """Build the pipeline, reusing the resident one when it already matches.

        Keyed on (dit, decoder) only. The duration is deliberately *not* part
        of the key: on torch it costs nothing to retarget (see
        Pipeline.set_duration), and keying on it meant the plugin — which loads
        at a default 8s before any source exists — rebuilt the whole model
        (~23s) the first time it saw a loop of any other length.
        """
        from sa3_pipeline import Pipeline

        dit, decoder = _MODEL_TO_DIT[model]
        key = (dit, decoder)
        if self.pipeline is not None and self.pipeline_key == key and self._retarget(seconds):
            self._last_load_seconds = 0.0
            return
        # Announce before building, not after. Reaching here costs ~13 s, and
        # a model switch does it mid-session; silence reads as a very slow
        # generation. (A new loop length no longer lands here — see _retarget.)
        _emit({
            "event": "status",
            "message": "Loading %s..." % _DISPLAY_NAME[model],
        })
        t0 = time.time()
        # Drop the old one first: loading is the memory-heaviest moment and
        # holding two sets of weights at once is what pushes a 12 GB card over.
        self.pipeline = None
        self.pipeline_key = None
        self.pipeline = Pipeline(
            dit=dit, decoder=decoder, seconds=float(seconds),
            dit_dtype="fp16", load_encoder_now=True, verbose=True,
        )
        self.pipeline_key = key
        self._last_load_seconds = time.time() - t0
        _log(f"pipeline ready: {key} in {self._last_load_seconds:.1f}s")

    _last_load_seconds = 0.0

    # ── generate ─────────────────────────────────────────────────────
    def generate(self, req: dict) -> None:
        import sa3_variations as sv
        from sa3_pipeline import read_wav_44k_stereo_s16

        src = Path(str(req["src"]))
        if not src.is_file():
            raise FileNotFoundError(f"source wav not found: {src}")

        outdir = Path(str(req.get("outdir") or src.parent / "variations"))
        outdir.mkdir(parents=True, exist_ok=True)

        preset = str(req.get("preset", "free"))
        kind = str(req.get("kind", "melodic"))
        seconds = float(req.get("seconds") or sv.wav_duration_seconds(src))
        steps = int(req.get("steps", 8))
        seed = int(req.get("seed", 1234))
        noise = float(req.get("noise", 0.45))
        dit, decoder = _MODEL_TO_DIT[self.model_name]

        if preset == "free":
            specs = sv.build_free_preset(
                kind=kind, duration=seconds, seed=seed, dit=dit, decoder=decoder,
                outdir=outdir, steps=steps, noise_a2a=noise,
            )
        elif preset == "app":
            bpm = req.get("bpm")
            bpm = float(bpm) if bpm else None
            user_prompt = str(req.get("userPrompt", ""))
            key = str(req.get("key", ""))
            negative_prompt = sv.negative_prompt_for_kind(kind)

            def base_prompt_builder(steer: str) -> str:
                return sv.prompt_for_kind(kind, user_prompt, bpm, key, steer)

            specs = sv.build_app_preset(
                kind=kind, duration=seconds, seed=seed,
                base_prompt_builder=base_prompt_builder,
                negative_prompt=negative_prompt, bpm=bpm,
                beats_per_bar=int(req.get("beatsPerBar", 4)),
                dit=dit, decoder=decoder, outdir=outdir, steps=steps,
                cfg_a2a=float(req.get("cfgA2a", 4.0)),
                cfg_inpaint=float(req.get("cfgInpaint", 4.0)),
                apg=float(req.get("apg", 1.0)),
                noise_a2a=noise,
            )
        else:
            raise ValueError(f"unknown preset {preset!r} (expected free|app)")

        # The pipeline is sized to the loop length, so a new duration rebuilds
        # it. Do that before the first candidate rather than mid-batch.
        self._ensure_pipeline(self.model_name, seconds)

        init_audio = read_wav_44k_stereo_s16(str(src))
        t0 = time.time()
        for i, spec in enumerate(specs):
            self.pipeline.generate(
                prompt=spec.prompt,
                negative_prompt=spec.negative_prompt or None,
                seed=spec.seed,
                steps=spec.steps,
                init_noise_level=spec.init_noise_level,
                cfg=spec.cfg,
                apg=spec.apg,
                init_audio_np=init_audio,
                inpaint_range_seconds=spec.inpaint_range,
                out_path=spec.out_path,
            )
            _emit({
                "event": "candidate",
                "idx": i,
                "path": str(Path(spec.out_path)),
                "steer": spec.steer,
                "mode": spec.mode,
            })
        _emit({"event": "done", "count": len(specs), "elapsed": time.time() - t0})

    # ── dispatch ─────────────────────────────────────────────────────
    def handle(self, req: dict) -> bool:
        """Returns False when the worker should exit."""
        cmd = req.get("cmd")
        if cmd == "quit":
            return False
        if cmd == "probe":
            self.probe()
        elif cmd == "load":
            self.load(req)
        elif cmd == "generate":
            self.generate(req)
        else:
            raise ValueError(f"unknown cmd {cmd!r}")
        return True


def main() -> int:
    worker = Worker()
    _log("started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"event": "error", "message": f"malformed request: {exc}"})
            continue
        try:
            if not worker.handle(req):
                break
        except Exception as exc:  # noqa: BLE001 - every failure must reach the UI
            _log(traceback.format_exc())
            _emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
    _log("exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
