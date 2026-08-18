#!/usr/bin/env python3
"""Publish the fp16 conversions to a public Hugging Face repo.

Run on the release machine after scripts/convert_weights_fp16.py. The repo it
creates is what plugin/scripts/install_models.ps1 downloads from, and it must
be **public and ungated** — the whole point is that colleagues install the app
without a Hugging Face account.

    python scripts/upload_weights.py --src dist/models
    python scripts/upload_weights.py --src dist/models --models medium
    python scripts/upload_weights.py --src dist/models --dry-run

Needs HF_TOKEN with write scope (the release machine keeps it in plugin/.env).

Attribution: these are Stability AI's weights, converted, not ours. The model
card written below names the upstream repos and points at their licences —
publishing a derivative without that would be sloppy at best.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "maxgraf/sa3-variations-torch"

# Upstream source for each converted directory, for the model card.
UPSTREAM = {
    "medium":      "stabilityai/stable-audio-3-medium",
    "small-music": "stabilityai/stable-audio-3-small-music",
    "small-sfx":   "stabilityai/stable-audio-3-small-sfx",
}

CARD = """---
license: other
license_name: stability-ai-community
license_link: https://huggingface.co/stabilityai/stable-audio-3-medium/blob/main/LICENSE.md
base_model:
{base_models}
tags:
  - audio
  - text-to-audio
  - stable-audio
library_name: stable-audio-3
---

# SA3 Variations — fp16 weights

Float16 conversions of Stability AI's [Stable Audio 3](https://huggingface.co/collections/stabilityai/stable-audio-3)
checkpoints, repackaged for the Windows build of **SA3 Variations**.

**These are Stability AI's models. This repo is a redistribution, not a new
model, and is not affiliated with or endorsed by Stability AI.**

## Why this exists

The SA3 Variations Windows app runs inference through PyTorch and needs its
weights on disk at install time. The upstream repos are gated, which would mean
every user creating a Hugging Face account and a token just to install a
desktop app. This repo removes that step.

## What was changed

Converted with [`scripts/convert_weights_fp16.py`](https://github.com/maxgraf96/stable-audio-3):
every float32/float64 tensor cast to float16. Integer indices, masks and bools
are untouched, and the bundled T5Gemma text encoder is left in **bfloat16** —
bf16 to fp16 narrows exponent range and risks overflow.

This is a storage change, not a quality one: the app loads with
`model_half=True` on CUDA, so these weights were being cast to fp16 at load
anyway. Verified by generating identical seeds from the original fp32 model and
this conversion:

    corr(fp32, fp16) = +1.000000   (3 seeds)

| | upstream fp32 | here |
|---|---|---|
| `medium/model.safetensors` | 9.22 GB | 4.3 GB |
| `medium/t5gemma-b-b-ul2/` | 1.18 GB | unchanged (bf16) |

## Layout

Each directory is self-contained — checkpoint, config, and its own copy of the
text encoder, so the app can load it with a local path and no network access:

```
{layout}
```

## Licence

Inherits the upstream terms, which you should read before using these files:

{licences}

The bundled T5Gemma encoder additionally carries Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Source

App and conversion script: <https://github.com/maxgraf96/stable-audio-3>
"""


def build_card(models: list[str]) -> str:
    base = "\n".join(f"  - {UPSTREAM[m]}" for m in models if m in UPSTREAM)
    layout = "\n".join(
        f"{m}/\n  model_config.json\n  model.safetensors\n  t5gemma-b-b-ul2/" for m in models
    )
    licences = "\n".join(
        f"- `{m}/` — [{UPSTREAM[m]}](https://huggingface.co/{UPSTREAM[m]})"
        for m in models if m in UPSTREAM
    )
    return CARD.format(base_models=base, layout=layout, licences=licences)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="output dir from convert_weights_fp16.py")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--models", nargs="+", help="subdirectories to upload (default: all present)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded, then stop")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("error: HF_TOKEN is not set (needs write scope)", file=sys.stderr)
        return 2

    models = args.models or sorted(d.name for d in args.src.iterdir() if d.is_dir())
    if not models:
        print(f"error: no model directories under {args.src}", file=sys.stderr)
        return 2

    total = 0
    print(f"repo:   {args.repo}  (public)")
    for m in models:
        d = args.src / m
        if not (d / "model.safetensors").is_file():
            print(f"error: {d} has no model.safetensors — convert it first", file=sys.stderr)
            return 2
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        total += size
        print(f"  {m:<14} {size / 1e9:6.2f} GB")
    print(f"  {'total':<14} {total / 1e9:6.2f} GB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    print(f"\nrepo ready: https://huggingface.co/{args.repo}")

    card = args.src / "README.md"
    card.write_text(build_card(models), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    print("model card uploaded")

    for m in models:
        print(f"\nuploading {m}/ ...")
        api.upload_folder(
            folder_path=str(args.src / m),
            path_in_repo=m,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"fp16 conversion of {UPSTREAM.get(m, m)}",
        )
        print(f"  done: https://huggingface.co/{args.repo}/tree/main/{m}")

    print("\nAll uploaded. install_models.ps1 pulls from this repo by default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
