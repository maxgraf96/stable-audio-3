#!/usr/bin/env bash
# build_release.sh — configure + build the three plugin formats with the
# Developer ID identity from plugin/.env. Re-runnable; uses an existing
# build/ tree incrementally.
#
# Usage:
#   plugin/scripts/build_release.sh           # incremental
#   plugin/scripts/build_release.sh --clean   # wipe build/ first
set -euo pipefail

cd "$(dirname "$0")/.."   # plugin/

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

CLEAN=0
for arg in "$@"; do
    [[ "$arg" == "--clean" ]] && CLEAN=1
done

if [[ "$CLEAN" == "1" ]]; then
    rm -rf build
fi
mkdir -p build
cd build

# `${VAR:--}` falls back to ad-hoc when .env doesn't have the value yet
# (template still has the FILL_ME_IN placeholder). Ad-hoc still
# produces a runnable local build; just not Gatekeeper-friendly.
identity="${SA3_CODESIGN_IDENTITY:--}"
case "$identity" in
    "Developer ID Application: Your Name "*) identity="-" ;;   # untouched template
esac

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSA3_CODESIGN_IDENTITY="$identity"

cmake --build . --target SA3_Standalone SA3_VST3 SA3_AU -j

echo
echo "build_release: ok  (identity: $identity)"
echo "Next step:"
echo "  plugin/scripts/prepare_release_assets.sh"
