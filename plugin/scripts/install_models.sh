#!/usr/bin/env bash
# install_models.sh — download SA3 Variations model weights from a GitHub
# release into the place the plugin looks at runtime:
#   ~/Library/Application Support/SA3 Variations/models/
#
# Usage:
#   ./install_models.sh                              # uses default release tag
#   ./install_models.sh v0.1.0                       # specific tag
#   SA3_RELEASE_BASE=<url> ./install_models.sh       # full override (any host)
#
# By default pulls from
#   https://github.com/maxgraf96/stable-audio-3/releases/download/<tag>/
# where each .safetensors lives as a release asset. dit_medium_f16 is
# split across two parts (GitHub's per-file cap is 2 GB) and we concat
# them client-side; the other three are single-file assets.
#
# Re-runs are idempotent — files that are already there with the right
# size get skipped.
set -euo pipefail

DEFAULT_TAG="v0.1.0"
TAG="${1:-$DEFAULT_TAG}"
REPO_SLUG="maxgraf96/stable-audio-3"
BASE_URL="${SA3_RELEASE_BASE:-https://github.com/${REPO_SLUG}/releases/download/${TAG}}"
DEST="$HOME/Library/Application Support/SA3 Variations/models"

mkdir -p "$DEST"
echo "Installing SA3 Variations models →  $DEST"
echo "Source: $BASE_URL"
echo

# fetch <remote_name> <local_name> [<expected_bytes>]
fetch() {
    local remote="$1" local_name="$2" expected="${3:-}"
    local dst="$DEST/$local_name"
    if [[ -f "$dst" ]]; then
        if [[ -n "$expected" ]]; then
            local sz; sz=$(stat -f%z "$dst")
            if [[ "$sz" == "$expected" ]]; then
                echo "  [skip] $local_name  (already present, $sz bytes)"
                return
            fi
            echo "  [redo] $local_name  (size $sz ≠ expected $expected — re-downloading)"
            rm -f "$dst"
        else
            echo "  [skip] $local_name  (already present, no size check)"
            return
        fi
    fi
    echo "  [get ] $local_name"
    curl -L --fail --progress-bar -o "$dst" "$BASE_URL/$remote"
}

# Single-file assets.
fetch "t5gemma_f16.safetensors"        "t5gemma_f16.safetensors"
fetch "same_l_encoder_f32.safetensors" "same_l_encoder_f32.safetensors"
fetch "same_l_decoder_f32.safetensors" "same_l_decoder_f32.safetensors"

# dit_medium_f16 is split because the unsplit file is >2 GB (GitHub's
# per-release-asset cap). Download both parts then concat.
if [[ ! -f "$DEST/dit_medium_f16.safetensors" ]]; then
    echo "  [get ] dit_medium_f16.safetensors  (split — part 1/2)"
    curl -L --fail --progress-bar \
        -o "$DEST/dit_medium_f16.safetensors.part-aa" \
        "$BASE_URL/dit_medium_f16.safetensors.part-aa"
    echo "  [get ] dit_medium_f16.safetensors  (split — part 2/2)"
    curl -L --fail --progress-bar \
        -o "$DEST/dit_medium_f16.safetensors.part-ab" \
        "$BASE_URL/dit_medium_f16.safetensors.part-ab"
    echo "  [join] dit_medium_f16.safetensors"
    cat "$DEST/dit_medium_f16.safetensors.part-aa" \
        "$DEST/dit_medium_f16.safetensors.part-ab" \
        > "$DEST/dit_medium_f16.safetensors"
    rm -f "$DEST/dit_medium_f16.safetensors.part-aa" \
          "$DEST/dit_medium_f16.safetensors.part-ab"
else
    echo "  [skip] dit_medium_f16.safetensors  (already present)"
fi

echo
echo "Done. Contents:"
ls -lh "$DEST"
