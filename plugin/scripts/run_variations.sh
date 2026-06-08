#!/usr/bin/env bash
# run_variations.sh — launch the SA3 Variations standalone built locally.
#
# Builds incrementally first (skip with --no-build), then opens the .app.
# Honours $SA3_MODELS_DIR if set so you can point at an alternate model
# checkpoint without copying anything.
set -euo pipefail

cd "$(dirname "$0")/.."   # plugin/

BUILD=1
for arg in "$@"; do
    [[ "$arg" == "--no-build" ]] && BUILD=0
done

if [[ "$BUILD" == "1" ]]; then
    scripts/build_release.sh
fi

APP="build/SA3_artefacts/Release/Standalone/SA3 Variations.app"
if [[ ! -d "$APP" ]]; then
    echo "error: $APP not found — run scripts/build_release.sh first" >&2
    exit 1
fi

pkill -x "SA3 Variations" 2>/dev/null || true
open "$APP"
