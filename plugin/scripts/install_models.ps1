# install_models.ps1 — download SA3 Variations model weights into the place
# the app looks at runtime:
#   %LOCALAPPDATA%\SA3 Variations\models\
#
# The PowerShell twin of install_models.sh, and deliberately the same shape:
# plain https resolve URLs (no HuggingFace client, no token), idempotent
# size-checked skips so an interrupted run resumes, and the same
# SA3_MODEL_SET switch.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_models.ps1
#   ... -Repo maxgraf/sa3-variations-torch -Revision main
#   $env:SA3_MODEL_SET="small"; ... \install_models.ps1
#
# Which models get downloaded (SA3_MODEL_SET):
#   auto    (default) everything this GPU can actually run — see below
#   all     every model
#   small   sa3-sm-music + sa3-sm-sfx only
#   medium  sa3-medium only
#
# `auto` exists because sa3-medium needs ~10.5 GB of VRAM and the app refuses
# to load it below that. On an 8 GB card the medium files are gigabytes that
# can never be used. The threshold mirrors kMediumMinGB in
# plugin/Source/WorkerBackend.cpp and MEDIUM_MIN_GB in sa3_worker.py — keep
# the three in step.
[CmdletBinding()]
param(
    [string]$Repo     = "maxgraf/sa3-variations-torch",
    [string]$Revision = "main"
)

$ErrorActionPreference = "Stop"
$MediumMinVramMB = 10500

$Base = if ($env:SA3_HF_BASE) { $env:SA3_HF_BASE } else { "https://huggingface.co/$Repo/resolve/$Revision" }
$Dest = Join-Path $env:LOCALAPPDATA "SA3 Variations\models"
$ModelSet = if ($env:SA3_MODEL_SET) { $env:SA3_MODEL_SET } else { "auto" }

function Step($m) { Write-Host "`n-> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "   OK $m" -ForegroundColor Green }

# Files that make up each model, relative to both the HF repo and $Dest.
# Every kind carries its own copy of the text encoder subfolder, because the
# app loads it with a local model_path (see load_local_model in
# sa3_pipeline_torch.py) rather than resolving it from a repo id.
function ModelFiles([string]$name) {
    @(
        "$name/model_config.json"
        "$name/model.safetensors"
        "$name/t5gemma-b-b-ul2/config.json"
        "$name/t5gemma-b-b-ul2/model.safetensors"
        "$name/t5gemma-b-b-ul2/special_tokens_map.json"
        "$name/t5gemma-b-b-ul2/tokenizer.json"
        "$name/t5gemma-b-b-ul2/tokenizer_config.json"
        "$name/t5gemma-b-b-ul2/tokenizer.model"
    )
}

# ── decide what to fetch ─────────────────────────────────────────────
$wantSmall = $false
$wantMedium = $false
switch ($ModelSet) {
    "all"    { $wantSmall = $true; $wantMedium = $true }
    "small"  { $wantSmall = $true }
    "medium" { $wantMedium = $true }
    "auto"   {
        $wantSmall = $true
        $vram = 0
        try {
            $q = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
            if ($q) { $vram = [int](($q -split "`n")[0].Trim()) }
        } catch { $vram = 0 }
        if ($vram -ge $MediumMinVramMB) {
            $wantMedium = $true
            Write-Host "GPU reports $vram MiB of VRAM — including SA3 Medium."
        } else {
            Write-Host "GPU reports $vram MiB of VRAM (< $MediumMinVramMB) — skipping SA3 Medium."
            Write-Host "Override with: `$env:SA3_MODEL_SET='all'"
        }
    }
    default  { throw "unknown SA3_MODEL_SET '$ModelSet' (expected auto|all|small|medium)" }
}

$files = @()
if ($wantSmall)  { $files += ModelFiles "small-music"; $files += ModelFiles "small-sfx" }
if ($wantMedium) { $files += ModelFiles "medium" }
if (-not $files) { throw "nothing selected to download" }

Write-Host "SA3 Variations — model download" -ForegroundColor White
Write-Host "repo:        $Repo@$Revision"
Write-Host "destination: $Dest"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# ── download ─────────────────────────────────────────────────────────
# Skip anything already present at the size the server reports, so a
# re-run after an interrupted download resumes rather than starting over.
$downloaded = 0
foreach ($rel in $files) {
    $url = "$Base/$rel"
    $out = Join-Path $Dest ($rel -replace "/", "\")
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null

    $remoteSize = 0
    try {
        $head = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -MaximumRedirection 5 -TimeoutSec 60
        $remoteSize = [int64]$head.Headers['Content-Length'][0]
    } catch {
        throw "cannot reach $url — check the network, or that the repo/revision exists.`n$($_.Exception.Message)"
    }

    if ((Test-Path $out) -and ((Get-Item $out).Length -eq $remoteSize)) {
        Write-Host ("   skip  {0}  ({1:N0} MB, already present)" -f $rel, ($remoteSize/1MB))
        continue
    }

    Step ("{0}  ({1:N0} MB)" -f $rel, ($remoteSize/1MB))
    # Invoke-WebRequest buffers whole responses in memory on Windows PowerShell
    # 5.1, which is fatal for a 4 GB file — BITS streams to disk and shows
    # native progress. Fall back to a direct stream copy if BITS is unavailable
    # (it is disabled in some managed environments).
    try {
        Start-BitsTransfer -Source $url -Destination $out -Description "SA3 model download" -ErrorAction Stop
    } catch {
        Write-Host "   (BITS unavailable, falling back to a direct download)"
        $wc = New-Object System.Net.WebClient
        try { $wc.DownloadFile($url, $out) } finally { $wc.Dispose() }
    }
    $downloaded++
    Ok ("{0:N0} MB" -f ((Get-Item $out).Length/1MB))
}

$total = (Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("`nModels ready — {0:N1} GB in {1}" -f ($total/1GB), $Dest) -ForegroundColor Green
if ($downloaded -eq 0) { Write-Host "(everything was already present)" }
