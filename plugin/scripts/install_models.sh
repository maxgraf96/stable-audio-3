#!/usr/bin/env bash
# install_models.sh — download SA3 Variations model weights from the
# HuggingFace Hub into the place the plugin looks at runtime:
#   ~/Library/Application Support/SA3 Variations/models/
#
# Usage:
#   ./install_models.sh                                   # default repo, main
#   ./install_models.sh <repo> [<revision>]               # override repo/rev
#   SA3_HF_BASE=<url> ./install_models.sh                 # full URL override
#   SA3_MODEL_SET=auto|all|small|medium ./install_models.sh
#
# The default repo is set below — bump it before cutting a release if you
# upload to a different HuggingFace account. Files are pulled via plain
# https resolve URLs so this script has zero dependencies beyond curl.
# Re-runs are idempotent: any model file already present with the right
# size is skipped.
#
# Which models get downloaded (SA3_MODEL_SET):
#   auto    (default) everything this Mac can actually run — see below
#   all     both bundles, ~8.6 GB
#   small   sa3-sm-music + sa3-sm-sfx only, ~2.7 GB
#   medium  sa3-medium only, ~6.4 GB
#
# `auto` exists because sa3-medium needs roughly 11 GB of unified memory to
# run without swapping, and the plugin refuses to load it below that. On an
# 8 GB Mac the medium bundle is 5.9 GB of files that can never be used, so
# auto skips it. The threshold matches MEDIUM_PEAK_BYTES + HOST_RESERVE_MIN_BYTES
# in optimized/cpp/sa3_orchestrator.h — keep the two in step.
set -euo pipefail

DEFAULT_REPO="maxgraf/sa3-variations-models"
DEFAULT_REV="main"
MEDIUM_MIN_BYTES=10600000000        # 10.6 GB — mirrors the plugin's own gate

REPO="${1:-$DEFAULT_REPO}"
REV="${2:-$DEFAULT_REV}"
BASE="${SA3_HF_BASE:-https://huggingface.co/${REPO}/resolve/${REV}}"
DEST="$HOME/Library/Application Support/SA3 Variations/models"
MODEL_SET="${SA3_MODEL_SET:-auto}"

# Shared text encoder — every model kind uses it.
FILES=("t5gemma_f16.safetensors")

want_small=0
want_medium=0
case "$MODEL_SET" in
    small)  want_small=1 ;;
    medium) want_medium=1 ;;
    all)    want_small=1; want_medium=1 ;;
    auto)
        want_small=1                    # runs on every Apple Silicon Mac
        RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        if [[ "$RAM_BYTES" -ge "$MEDIUM_MIN_BYTES" ]]; then
            want_medium=1
        else
            echo "Note: this Mac has $((RAM_BYTES / 1000000000)) GB of unified memory."
            echo "      SA3 Medium needs ~11 GB, so it is being skipped (the plugin"
            echo "      would refuse to load it anyway). Re-run with"
            echo "      SA3_MODEL_SET=all to download it regardless."
            echo
        fi
        ;;
    *)
        echo "error: SA3_MODEL_SET must be auto|all|small|medium (got '$MODEL_SET')" >&2
        exit 2
        ;;
esac

if [[ "$want_small" == 1 ]]; then
    FILES+=("dit_sm-music_f16.safetensors"
            "dit_sm-sfx_f16.safetensors"
            "same_s_encoder_f32.safetensors"
            "same_s_decoder_f32.safetensors")
fi
if [[ "$want_medium" == 1 ]]; then
    FILES+=("dit_medium_f16.safetensors"
            "same_l_encoder_f32.safetensors"
            "same_l_decoder_f32.safetensors")
fi

mkdir -p "$DEST"
echo "Installing SA3 Variations models →  $DEST"
echo "Source: $BASE"
echo "Set:    $MODEL_SET (${#FILES[@]} files)"
echo

# fetch <filename>
# Skips a redownload if the local file's size already matches the HEAD
# request's Content-Length on the Hub. Hugging Face responds to HEAD
# with the byte count of the underlying object (after LFS redirect).
fetch() {
    local name="$1"
    local url="$BASE/$name"
    local dst="$DEST/$name"
    # Follow redirects (LFS), look at the final Content-Length.
    local remote_size
    remote_size=$(curl -sI -L "$url" \
        | awk 'tolower($1)=="content-length:" {print $2}' \
        | tr -d '\r' \
        | tail -n1)
    if [[ -f "$dst" && -n "$remote_size" ]]; then
        local local_size; local_size=$(stat -f%z "$dst")
        if [[ "$local_size" == "$remote_size" ]]; then
            echo "  [skip] $name  ($local_size bytes)"
            return
        fi
        echo "  [redo] $name  (local $local_size ≠ remote $remote_size — re-downloading)"
        rm -f "$dst"
    fi
    echo "  [get ] $name  ($remote_size bytes)"
    curl -L --fail --progress-bar -o "$dst" "$url"
}

for f in "${FILES[@]}"; do
    fetch "$f"
done

echo
echo "Done. Contents:"
ls -lh "$DEST"
