"""Backend dispatcher for the SA3 variation runtime.

sa3_variations.py and sa3_studio.py import `Pipeline` and
`read_wav_44k_stereo_s16` from here and never learn which backend they got:

  - sa3_pipeline_mlx.py   — Apple Silicon, drives optimized/mlx/ directly.
  - sa3_pipeline_torch.py — CUDA / CPU, drives the upstream `stable_audio_3`
                            package.

MLX wins when it's importable because it is the faster path on Apple Silicon
and is what the presets in README_VARIATIONS were tuned against. Set
SA3_BACKEND=torch|mlx to override (useful for A/B-ing the two on a Mac).
"""
from __future__ import annotations

import os

_REQUESTED = os.environ.get("SA3_BACKEND", "auto").strip().lower()
if _REQUESTED not in ("auto", "mlx", "torch"):
    raise RuntimeError(f"SA3_BACKEND must be auto, mlx, or torch — got {_REQUESTED!r}")


def _load_mlx():
    from sa3_pipeline_mlx import Pipeline, read_wav_44k_stereo_s16

    return "mlx", Pipeline, read_wav_44k_stereo_s16


def _load_torch():
    from sa3_pipeline_torch import Pipeline, read_wav_44k_stereo_s16

    return "torch", Pipeline, read_wav_44k_stereo_s16


if _REQUESTED == "mlx":
    BACKEND, Pipeline, read_wav_44k_stereo_s16 = _load_mlx()
elif _REQUESTED == "torch":
    BACKEND, Pipeline, read_wav_44k_stereo_s16 = _load_torch()
else:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        BACKEND, Pipeline, read_wav_44k_stereo_s16 = _load_torch()
    else:
        BACKEND, Pipeline, read_wav_44k_stereo_s16 = _load_mlx()

__all__ = ["BACKEND", "Pipeline", "read_wav_44k_stereo_s16"]
