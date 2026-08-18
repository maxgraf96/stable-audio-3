# build_installer.ps1 - package the Windows standalone into an installer.
#
# The macOS twin of this is prepare_release_assets.sh. Same shape: take what
# CMake built, stage it into a distributable layout, emit an artifact into
# plugin\build\release_assets\.
#
# Unlike macOS there is nothing to sign or notarise yet, so this is purely a
# staging + compile step. Colleagues will see a SmartScreen warning on first
# run; see RELEASE_WINDOWS.md.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File plugin\scripts\build_installer.ps1
#   ... -Version 0.2.0
#   ... -SkipBuild          reuse the existing CMake output
[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$Repo    = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$Plugin  = Join-Path $Repo "plugin"
$Build   = Join-Path $Plugin "build"
$Payload = Join-Path $Build "installer_payload"
$Assets  = Join-Path $Build "release_assets"
$ExeName = "SA3 Variations.exe"

function Step($m) { Write-Host "`n-> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "   OK $m" -ForegroundColor Green }

# -- build ------------------------------------------------------------
if (-not $SkipBuild) {
    Step "Building the standalone (Release)"
    $cmake = @("$env:ProgramFiles\CMake\bin\cmake.exe", "cmake") |
             Where-Object { $_ -eq "cmake" -or (Test-Path $_) } | Select-Object -First 1
    & $cmake --build $Build --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
}
$exe = Join-Path $Build "SA3_artefacts\Release\Standalone\$ExeName"
if (-not (Test-Path $exe)) { throw "missing $exe - build the Standalone target first" }
Ok $exe

# -- stage ------------------------------------------------------------
Step "Staging payload"
if (Test-Path $Payload) { Remove-Item $Payload -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Payload, "$Payload\app", "$Payload\scripts", "$Payload\tools" | Out-Null

Copy-Item $exe $Payload

# The Python side of the app. sa3_pipeline.py is the backend dispatcher, so
# sa3_pipeline_mlx.py comes along too - it is never imported on Windows (the
# dispatcher only loads it when `import mlx.core` succeeds) but shipping it
# keeps the file set identical to the repo and avoids a surprising ImportError
# if someone sets SA3_BACKEND=mlx.
foreach ($f in @("sa3_worker.py")) {
    Copy-Item (Join-Path $Plugin "scripts\$f") "$Payload\app\"
}
foreach ($f in @("sa3_variations.py", "sa3_pipeline.py", "sa3_pipeline_torch.py", "sa3_pipeline_mlx.py")) {
    Copy-Item (Join-Path $Repo $f) "$Payload\app\"
}
Copy-Item (Join-Path $Repo "stable_audio_3") "$Payload\app\stable_audio_3" -Recurse
Get-ChildItem "$Payload\app" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

Copy-Item (Join-Path $Plugin "scripts\setup_runtime.ps1")  "$Payload\scripts\"
Copy-Item (Join-Path $Plugin "scripts\install_models.ps1") "$Payload\scripts\"

# Runtime dependency set: ship the project's own pyproject.toml + uv.lock and
# let `uv sync` resolve on the target machine. An exported requirements.txt
# does NOT carry [[tool.uv.index]], so the pinned torch==2.7.1+cu128 would be
# unresolvable there - PyPI only has plain 2.7.1. Shipping the lock also means
# a colleague gets byte-for-byte the resolution that was tested here.
Step "Staging pyproject.toml + uv.lock"
Copy-Item (Join-Path $Repo "pyproject.toml") "$Payload\app\"
Copy-Item (Join-Path $Repo "uv.lock")        "$Payload\app\"
Ok "lockfile staged"

Step "Bundling uv.exe"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw "uv not found on PATH - needed to bootstrap the runtime on target machines" }
Copy-Item $uv "$Payload\tools\uv.exe"
Ok $uv

@"
SA3 Variations $Version

Generate five variations of any audio loop, locally, on your NVIDIA GPU.

FIRST RUN
  The installer sets up a Python runtime and downloads model weights. If you
  skipped either step, re-run them from the Start Menu:
    Start > SA3 Variations > Set up Python runtime
    Start > SA3 Variations > Download models

REQUIREMENTS
  Windows 10/11 x64, an NVIDIA GPU, driver 570 or newer.
  SA3 Medium needs about 10.5 GB of VRAM; below that the app uses the
  smaller models automatically.

WHERE THINGS LIVE
  Application   %LOCALAPPDATA%\Programs\SA3 Variations
  Models        %LOCALAPPDATA%\SA3 Variations\models
  Worker log    %LOCALAPPDATA%\SA3 Variations\worker.log

  Uninstalling leaves the models in place so a reinstall doesn't re-download
  them. Delete that folder by hand to reclaim the space.

TROUBLESHOOTING
  Generation fails, or the app says the runtime is missing -> re-run
  "Set up Python runtime" and read worker.log, which holds the model load
  progress and any Python traceback.
"@ | Set-Content "$Payload\README.txt" -Encoding utf8

$payloadMB = [math]::Round((Get-ChildItem $Payload -Recurse -File | Measure-Object Length -Sum).Sum/1MB, 1)
Ok "payload staged: $payloadMB MB"

# -- compile ----------------------------------------------------------
Step "Compiling the installer"
# winget's per-user install lands under %LOCALAPPDATA%\Programs, the
# machine-wide installer under Program Files. Check both.
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
          "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
          "$env:ProgramFiles\Inno Setup 6\ISCC.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found. Install with: winget install JRSoftware.InnoSetup" }

New-Item -ItemType Directory -Force -Path $Assets | Out-Null
& $iscc "/DAppVersion=$Version" "/DPayloadDir=$Payload" "/DOutputDir=$Assets" `
        (Join-Path $Plugin "installer\sa3_variations.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }

$out = Join-Path $Assets "SA3-Variations-Setup-$Version.exe"
$mb = [math]::Round((Get-Item $out).Length/1MB, 1)
Write-Host "`nInstaller ready: $out ($mb MB)" -ForegroundColor Green
