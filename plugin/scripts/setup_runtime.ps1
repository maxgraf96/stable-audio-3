# setup_runtime.ps1 - build the Python runtime SA3 Variations needs.
#
# The app itself is a 6 MB executable; inference runs in a Python worker
# (app\sa3_worker.py) driving PyTorch. That runtime is ~5.6 GB installed, well
# past GitHub's 2 GB release-asset cap, so it is built here on the target
# machine rather than shipped. Downloads ~2.5 GB, mostly the CUDA-enabled torch
# wheel.
#
# Safe to re-run: uv reuses whatever is already installed, so a failed or
# interrupted run picks up where it left off.
#
# Usage (from the install folder):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1
#
#   -Force   recreate runtime\ from scratch
[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"

$Root    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runtime = Join-Path $Root "runtime"
$Uv      = Join-Path $Root "tools\uv.exe"
$AppDir  = Join-Path $Root "app"

function Step($m) { Write-Host "`n-> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "   OK $m" -ForegroundColor Green }
function Warn($m) { Write-Host "   !! $m" -ForegroundColor Yellow }

Write-Host "SA3 Variations - runtime setup" -ForegroundColor White
Write-Host "install folder: $Root"

# -- preflight --------------------------------------------------------
# Check the things that make this fail slowly and confusingly if unmet, and
# say so now rather than 2 GB into a download.

Step "Checking disk space"
$drive = (Get-Item $Root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
if ($freeGB -lt 16) {
    throw "only $freeGB GB free on $($drive.Name): - the runtime needs ~6 GB and the models ~5 GB. Free up space and re-run."
}
Ok "$freeGB GB free on $($drive.Name):"

Step "Checking for an NVIDIA GPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Warn "nvidia-smi not found. SA3 will fall back to CPU, which is far too slow to be usable."
    Warn "Install the current NVIDIA driver from https://www.nvidia.com/Download/index.aspx"
} else {
    $info = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    if ($info) {
        Ok $info
        # PyTorch is built against CUDA 12.8; the driver has to be new enough to
        # expose it. 570 is the floor, and Blackwell (RTX 50-series) needs it.
        $drv = ($info -split ",")[2].Trim()
        $major = [int](($drv -split "\.")[0])
        if ($major -lt 570) {
            Warn "driver $drv is older than 570 - CUDA 12.8 needs 570+. Update the NVIDIA driver before running the app."
        }
        $vram = [double](($info -split ",")[1] -replace "[^0-9.]", "")
        if ($vram -lt 10500) {
            Warn ("this GPU reports {0} MiB of VRAM; SA3 Medium needs ~10.5 GB. The app will default to the small models." -f $vram)
        }
    }
}

if (-not (Test-Path $Uv))  { throw "missing $Uv - the install looks incomplete, please reinstall." }
if (-not (Test-Path (Join-Path $AppDir "uv.lock"))) { throw "missing uv.lock in $AppDir - the install looks incomplete, please reinstall." }

# -- build the venv ---------------------------------------------------
if ($Force -and (Test-Path $Runtime)) {
    Step "Removing existing runtime\ (-Force)"
    Remove-Item $Runtime -Recurse -Force
}

Step "Creating runtime\ (Python 3.10)"
# --relocatable so the folder survives being moved, e.g. if someone relocates
# the install. --python-preference only-managed makes uv fetch its own Python
# rather than depending on whatever happens to be on PATH.
& $Uv venv --relocatable --python 3.10 --python-preference only-managed $Runtime
if ($LASTEXITCODE -ne 0) { throw "uv venv failed ($LASTEXITCODE)" }
Ok "runtime\ created"

Step "Installing PyTorch + dependencies (~2.5 GB download, several minutes)"
# `uv sync` against the shipped lock rather than a requirements file: the lock
# carries the [[tool.uv.index]] entry routing torch to the CUDA 12.8 index,
# which a flat requirements.txt cannot express - PyPI only has plain 2.7.1.
# UV_PROJECT_ENVIRONMENT points uv at runtime\ instead of creating app\.venv, and
# --no-install-project skips building stable-audio-3 itself: its source ships
# in app\ and the worker puts that directory on sys.path.
$env:UV_PROJECT_ENVIRONMENT = $Runtime
& $Uv sync --project $AppDir --no-dev --no-install-project
if ($LASTEXITCODE -ne 0) {
    throw "dependency install failed ($LASTEXITCODE). If this machine is behind a proxy, note that the CUDA wheels come from https://download.pytorch.org/whl/cu128 and the rest from https://pypi.org."
}
Ok "dependencies installed"

# -- prove it actually works ------------------------------------------
# Importing torch and asking for the device is the cheapest end-to-end check
# that the wheel matches the driver - a mismatch here is much easier to read
# than the same failure surfacing inside the audio app later.
Step "Verifying CUDA"
$py = Join-Path $Runtime "Scripts\python.exe"
& $py -c "import torch; print('torch', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
if ($LASTEXITCODE -ne 0) { throw "the runtime was installed but torch failed to import" }

Write-Host "`nRuntime ready." -ForegroundColor Green
Write-Host "Next: scripts\install_models.ps1 downloads the model weights." -ForegroundColor White
