# SA3 Variations — Windows Release Notes

How to produce a Windows installer and what end-users need on their machines.
The macOS twin of this document is [RELEASE.md](RELEASE.md); read that one for
the AU/VST3/notarisation story, none of which applies here.

Both platforms ship the same app, but not the same way, because inference runs
differently on each:

| | macOS | Windows |
|---|---|---|
| Inference | in-process C++ MLX orchestrator | Python worker (`sa3_worker.py`) driving PyTorch |
| Formats | AU + VST3 + Standalone | **Standalone only** (for now) |
| Runtime | none — dylibs vendored into the bundle | ~5.6 GB Python env, built on the target machine |
| Weights | ~8.6 GB converted safetensors | ~5.5 GB fp16 safetensors |
| Signing | Developer ID + notarisation | **unsigned** — SmartScreen warning on first run |

## Builder prerequisites

- Windows 10/11 x64 with an NVIDIA GPU, so you can test what you ship.
- Visual Studio 2022 **Build Tools** with the C++ workload and a Windows SDK:
  `winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
- CMake ≥ 3.22 — `winget install Kitware.CMake`
- Inno Setup 6 — `winget install JRSoftware.InnoSetup`
- `uv` on PATH. It is both the dev dependency manager and the binary bundled
  into the installer to build the end-user runtime.
- The `Microsoft.Web.WebView2` NuGet package unpacked where JUCE looks for it,
  `%LOCALAPPDATA%\PackageManagement\NuGet\Packages\Microsoft.Web.WebView2.<ver>\`.
  JUCE prints the exact `Install-Package` command when it is missing; the
  package is a plain zip, so downloading from nuget.org and extracting also
  works.

## End-user prerequisites

- Windows 10/11 x64, **NVIDIA GPU**, driver **570 or newer** — PyTorch is built
  against CUDA 12.8, and Blackwell (RTX 50-series) requires it.
- ~16 GB free disk: ~5.6 GB runtime, ~5.5 GB weights, plus working room.
- **~10.5 GB VRAM** for SA3 Medium. Below that the app defaults to SA3 Small
  Music, and `install_models.ps1` skips the medium weights rather than
  downloading gigabytes that could never load.
- No Python, no Hugging Face account, no repo checkout.

## Build

```powershell
# one-time, if the build tree doesn't exist yet
cmake -S plugin -B plugin/build -G "Visual Studio 17 2022" -A x64

powershell -ExecutionPolicy Bypass -File plugin\scripts\build_installer.ps1 -Version 0.1.0
#   -SkipBuild   reuse the current CMake output
```

Output: `plugin\build\release_assets\SA3-Variations-Setup-<version>.exe` (~16 MB).

What goes into the payload:

| Path | Contents |
|---|---|
| `SA3 Variations.exe` | the JUCE standalone |
| `app\` | `sa3_worker.py`, the variation harness, `stable_audio_3\`, `pyproject.toml`, `uv.lock` |
| `scripts\` | `setup_runtime.ps1`, `install_models.ps1` |
| `tools\uv.exe` | bootstraps the runtime on the target machine |

### Why the runtime isn't in the installer

It is ~5.6 GB installed and ~2.5 GB compressed, and GitHub caps release assets
at **2 GB**. So it is built on the target machine by `setup_runtime.ps1` — the
same approach upstream takes for its `optimized/tflite` Windows path. The cost
is a first install that downloads for several minutes.

### Why the lockfile ships, not a requirements.txt

`uv export` produces a flat requirements file that **cannot express
`[[tool.uv.index]]`**, so the pinned `torch==2.7.1+cu128` becomes unresolvable
on the target machine — PyPI only carries plain `2.7.1`. Shipping
`pyproject.toml` + `uv.lock` and running `uv sync` there keeps the CUDA index
routing intact and gives every colleague byte-for-byte the resolution that was
tested here.

`--no-install-project` skips building `stable-audio-3` itself: its source ships
in `app\`, and the worker puts that directory on `sys.path`.

### PowerShell scripts must stay ASCII-only

Windows PowerShell 5.1 — still the default on Windows 10/11 — reads `.ps1`
files as CP1252 unless they carry a UTF-8 BOM. A UTF-8 em dash (`E2 80 94`)
decodes as `â€"`, and that trailing `0x94` is a curly closing quote which
**terminates the enclosing string**, producing parse errors nowhere near the
real line. Keep `.ps1` files plain ASCII (`-` not `—`, `->` not `→`), which is
what upstream's `optimized/tflite/bootstrap.ps1` does.

## Publishing the weights

End-users must not need a Hugging Face token, so the weights are re-hosted
ungated, converted to fp16 first:

```powershell
python scripts\convert_weights_fp16.py --out dist\models --verify <some_loop.wav>
```

`--verify` generates the same seeds from the original fp32 hub model and the
converted local one, and refuses to pass below 0.999 correlation. It measured
**+1.000000** across three seeds. fp16 is what the app runs anyway
(`model_half=True` on CUDA), so this is a storage change, not a quality one.

| | fp32 (upstream) | fp16 (published) |
|---|---|---|
| `medium/model.safetensors` | 9.22 GB | **4.3 GB** |
| `medium/t5gemma-b-b-ul2/` | 1.18 GB | 1.18 GB — unchanged |

t5gemma ships as **bfloat16**, and bf16 → fp16 narrows exponent range, so the
converter casts only float32/float64 and leaves it — along with every integer
index, mask and bool — alone.

Upload to an ungated repo and point `install_models.ps1` at it via its `-Repo`
default:

```powershell
hf upload maxgraf/sa3-variations-torch dist\models . --repo-type model
```

> **Licence check before uploading.** Redistributing Stability's weights — even
> converted — is a licensing question, not just a hosting one. The existing
> `maxgraf/sa3-variations-models` repo is precedent for the MLX conversions, but
> confirm the SA3 and Gemma terms cover this artifact before making it public.

Anything the model dropdown offers must exist in that repo, or selecting it
fails at load. When you add a kind to `DIT_CHOICES` in `sa3_pipeline_torch.py`,
convert and upload its weights in the same change.

## End-user install

1. Download `SA3-Variations-Setup-<version>.exe` from the GitHub release.
2. Run it. **SmartScreen will warn** — the installer is unsigned. *More info* →
   *Run anyway*. This is the Windows equivalent of the un-notarised macOS
   right-click → Open path.
3. Leave both components ticked. The installer sets up the Python runtime
   (~2.5 GB) and downloads the weights (~5.5 GB), each in a visible window.
4. Launch from the Start Menu, drop a loop, hit Generate.

Both downloads are re-runnable from **Start → SA3 Variations** if either was
skipped or failed, so a bad network moment doesn't mean reinstalling.

## Where things live

```
%LOCALAPPDATA%\Programs\SA3 Variations\    application, runtime\, app\, scripts\
%LOCALAPPDATA%\SA3 Variations\models\      weights
%LOCALAPPDATA%\SA3 Variations\worker.log   model load progress + Python tracebacks
```

Model lookup (`models_root()` in `sa3_pipeline_torch.py`, mirroring macOS's
`resolveModelsDir()`):

1. `%SA3_MODELS_DIR%` — dev override.
2. `%LOCALAPPDATA%\SA3 Variations\models` — production default.
3. The Hugging Face cache — how a dev checkout works, and why this machine
   needs no local models dir.

Worker lookup (`resolveWorkerPaths()` in `plugin/Source/WorkerBackend.cpp`):
the installed layout (`<exe>\app\`, `<exe>\runtime\`) is preferred, with the dev
checkout as fallback; `SA3_REPO` / `SA3_PYTHON` override either.

Uninstalling removes the app and its runtime but **leaves the models**, so a
reinstall doesn't re-download 5.5 GB. Delete that folder by hand to reclaim the
space.

## Troubleshooting

| Symptom | Cause |
|---|---|
| "no Python runtime at ...\runtime\Scripts\python.exe" | runtime step skipped or failed — re-run *Set up Python runtime* |
| Generation fails immediately | read `worker.log`; it holds load progress and any Python traceback |
| First generate slow, later ones fast | expected — the first click pays the ~13 s model load |
| Everything inexplicably slow | something else is holding VRAM. Check `nvidia-smi`; see "Only one thing may hold the GPU at a time" in [README_VARIATIONS.md](../README_VARIATIONS.md) |
| App says CUDA unavailable | driver older than 570, or no NVIDIA GPU |

## Not in scope yet

- **VST3** — CMake builds Standalone only off Apple. Adding it is a one-line
  change to `SA3_FORMATS` plus DAW testing.
- **Code signing** — Azure Trusted Signing (~$10/month, no hardware token) is
  the cheapest route that works in CI.
- **CI packaging** — the installer is built locally today.
