#!/usr/bin/env bash
# prepare_release_assets.sh — package the bundles + split the oversized
# safetensors file into pieces that fit under GitHub's 2 GB per-asset cap.
# After running this, upload everything in `release_assets/` to a fresh
# GitHub release.
#
# Pre-reqs: run a clean release build first:
#   cd plugin/build && rm -rf * && cmake .. \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DSA3_CODESIGN_IDENTITY="Developer ID Application: …" \
#     && cmake --build . --target SA3_Standalone SA3_VST3 SA3_AU -j
#
# Outputs all assets to plugin/build/release_assets/.
set -euo pipefail

cd "$(dirname "$0")/.."   # plugin/

BUILD=build
ART="$BUILD/SA3_artefacts/Release"
OUT="$BUILD/release_assets"
MODELS_SRC="${SA3_MODELS_DIR:-$(cd .. && pwd)/optimized/mlx/models/mlx}"

if [[ ! -d "$ART/Standalone/SA3 Variations.app" ]]; then
    echo "error: build artefacts missing at $ART — run cmake --build first" >&2
    exit 1
fi
if [[ ! -d "$MODELS_SRC" ]]; then
    echo "error: models dir not found at $MODELS_SRC" >&2
    echo "       set SA3_MODELS_DIR or place models at optimized/mlx/models/mlx/" >&2
    exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT"

echo "==> Plugins archive"
TMP=$(mktemp -d)
cp -R "$ART/Standalone/SA3 Variations.app"        "$TMP/"
cp -R "$ART/VST3/SA3 Variations.vst3"             "$TMP/"
cp -R "$ART/AU/SA3 Variations.component"          "$TMP/"
cat > "$TMP/README.txt" <<'EOF'
SA3 Variations — install
========================

1. Drag the three bundles to:
     SA3 Variations.app          ->  /Applications/
     SA3 Variations.vst3         ->  ~/Library/Audio/Plug-Ins/VST3/
     SA3 Variations.component    ->  ~/Library/Audio/Plug-Ins/Components/

2. Download and run install_models.sh (separate release asset) to put
   the ~6.4 GB of model weights in:
     ~/Library/Application Support/SA3 Variations/models/

3. First launch shows a Gatekeeper warning because this build isn't
   notarised. Right-click the .app / .vst3 / .component → Open → confirm.
   macOS remembers per bundle.

Requires Apple Silicon Mac on macOS 13.5+.
EOF
(cd "$TMP" && zip -r -y "$OLDPWD/$OUT/SA3-Variations-plugins.zip" \
    "SA3 Variations.app" \
    "SA3 Variations.vst3" \
    "SA3 Variations.component" \
    "README.txt")
rm -rf "$TMP"
ls -lh "$OUT/SA3-Variations-plugins.zip"

echo
echo "==> Model assets"
cp "$MODELS_SRC/t5gemma_f16.safetensors"        "$OUT/"
cp "$MODELS_SRC/same_l_encoder_f32.safetensors" "$OUT/"
cp "$MODELS_SRC/same_l_decoder_f32.safetensors" "$OUT/"

# dit_medium_f16 is ~2.7 GB. GitHub release assets cap at 2 GB per file,
# so split at 1.5 GB (gives us part-aa and part-ab). Plain `cat` on the
# user side rejoins.
DIT="$MODELS_SRC/dit_medium_f16.safetensors"
echo "    splitting dit_medium_f16.safetensors ($(stat -f%z "$DIT") bytes)"
split -b 1500m "$DIT" "$OUT/dit_medium_f16.safetensors.part-"
# `split -b` produces .part-aa, .part-ab, ... — matches install_models.sh

echo
echo "==> install script"
cp scripts/install_models.sh "$OUT/"
chmod +x "$OUT/install_models.sh"

echo
echo "==> Manifest"
ls -lh "$OUT"
echo
echo "Upload everything in $OUT to a new GitHub release tagged v0.1.0."
echo "install_models.sh defaults to that tag; pass a different tag to it"
echo "(or override SA3_RELEASE_BASE) for later versions."
