#!/usr/bin/env bash
# prepare_release_assets.sh — bundle up the three plugin artefacts + the
# install script into something you can drag into a GitHub release. The
# model weights live on HuggingFace separately and are not packaged here.
#
# Pre-req: a clean release build, e.g.
#   cd plugin/build && rm -rf * && \
#     cmake .. -DCMAKE_BUILD_TYPE=Release \
#              -DSA3_CODESIGN_IDENTITY="Developer ID Application: …" && \
#     cmake --build . --target SA3_Standalone SA3_VST3 SA3_AU -j
#
# Notarization runs automatically if plugin/.env contains valid
# APPLE_ID / APPLE_TEAM_ID / APPLE_APP_PASSWORD entries. Skip notarization
# by passing --no-notarize (or simply leaving .env unfilled).
#
# Outputs:
#   plugin/build/release_assets/SA3-Variations-plugins.zip   (~450 MB)
#   plugin/build/release_assets/install_models.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # plugin/

# Pull APPLE_* and SA3_* credentials out of plugin/.env if present.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

BUILD=build
ART="$BUILD/SA3_artefacts/Release"
OUT="$BUILD/release_assets"

# Decide whether to notarize. We skip if .env values are still
# placeholders OR if --no-notarize was passed on the command line.
notarize=1
for arg in "$@"; do
    [[ "$arg" == "--no-notarize" ]] && notarize=0
done
for v in APPLE_ID APPLE_TEAM_ID APPLE_APP_PASSWORD; do
    if [[ -z "${!v:-}" ]] || [[ "${!v}" == "FILL_ME_IN"* ]] \
                          || [[ "${!v}" == "xxxx-"* ]]      \
                          || [[ "${!v}" == "XXXXXXXXXX" ]]; then
        notarize=0
    fi
done

if [[ ! -d "$ART/Standalone/SA3 Variations.app" ]]; then
    echo "error: build artefacts missing at $ART — run cmake --build first" >&2
    exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT"

# Notarize each bundle in-place before zipping into the release archive.
# Each notarytool submit blocks on Apple's service (~1-10 min). Serialised
# here for readable output; parallel would shave a few minutes off.
if [[ "$notarize" == "1" ]]; then
    for b in "$ART/Standalone/SA3 Variations.app" \
             "$ART/VST3/SA3 Variations.vst3" \
             "$ART/AU/SA3 Variations.component"; do
        echo
        echo "############ Notarizing $b ############"
        scripts/notarize.sh "$b"
    done
else
    echo "(skipping notarization — .env credentials not set or --no-notarize passed)"
fi

echo "==> Plugins archive"
TMP=$(mktemp -d)
cp -R "$ART/Standalone/SA3 Variations.app"     "$TMP/"
cp -R "$ART/VST3/SA3 Variations.vst3"          "$TMP/"
cp -R "$ART/AU/SA3 Variations.component"       "$TMP/"
cat > "$TMP/README.txt" <<'EOF'
SA3 Variations — install
========================

1. Drag the three bundles to the standard locations:
     SA3 Variations.app          ->  /Applications/
     SA3 Variations.vst3         ->  ~/Library/Audio/Plug-Ins/VST3/
     SA3 Variations.component    ->  ~/Library/Audio/Plug-Ins/Components/

2. Download install_models.sh from the same GitHub release page and run
   it in Terminal:
     chmod +x install_models.sh
     ./install_models.sh
   It downloads ~6.4 GB of model weights from HuggingFace into:
     ~/Library/Application Support/SA3 Variations/models/

3. First launch shows a Gatekeeper warning because this build isn't
   notarised. Right-click the .app / .vst3 / .component → Open → confirm.
   macOS remembers the choice per bundle.

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
echo "==> install_models.sh"
cp scripts/install_models.sh "$OUT/"
chmod +x "$OUT/install_models.sh"

echo
echo "==> Manifest"
ls -lh "$OUT"
echo
echo "Upload both files in $OUT to a new GitHub release."
echo "Model weights live separately on HuggingFace — see RELEASE.md."
