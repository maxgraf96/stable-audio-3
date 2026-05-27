#!/usr/bin/env bash
# install_models.sh — download SA3 Variations model weights from the
# HuggingFace Hub into the place the plugin looks at runtime:
#   ~/Library/Application Support/SA3 Variations/models/
#
# Usage:
#   ./install_models.sh                                   # default repo, main
#   ./install_models.sh <repo> [<revision>]               # override repo/rev
#   SA3_HF_BASE=<url> ./install_models.sh                 # full URL override
#
# The default repo is set below — bump it before cutting a release if you
# upload to a different HuggingFace account. Files are pulled via plain
# https resolve URLs so this script has zero dependencies beyond curl.
# Re-runs are idempotent: any model file already present with the right
# size is skipped.
set -euo pipefail

DEFAULT_REPO="maxgraf/sa3-variations-models"
DEFAULT_REV="main"

REPO="${1:-$DEFAULT_REPO}"
REV="${2:-$DEFAULT_REV}"
BASE="${SA3_HF_BASE:-https://huggingface.co/${REPO}/resolve/${REV}}"
DEST="$HOME/Library/Application Support/SA3 Variations/models"

mkdir -p "$DEST"
echo "Installing SA3 Variations models →  $DEST"
echo "Source: $BASE"
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

fetch "t5gemma_f16.safetensors"
fetch "dit_medium_f16.safetensors"
fetch "same_l_encoder_f32.safetensors"
fetch "same_l_decoder_f32.safetensors"

echo
echo "Done. Contents:"
ls -lh "$DEST"
