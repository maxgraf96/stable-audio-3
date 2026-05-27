#!/usr/bin/env bash
# vendor_bundle.sh — make an SA3 plugin bundle self-contained on macOS.
#
# Copies the MLX + SentencePiece runtime libraries into Contents/Frameworks/,
# rewrites install names so they resolve via @loader_path/../Frameworks,
# strips the dev-tree LC_RPATH, optionally clones the four safetensors model
# files into Contents/Resources/models/, and ad-hoc re-signs everything. The
# script is idempotent — re-running on a bundle that's already vendored is a
# fast no-op.
#
# Usage:
#   vendor_bundle.sh <bundle> <mlx_lib_dir> <sp_lib_dir> <models_dir> \
#                    [<dev_rpath>] [<codesign_identity>] [<entitlements_plist>]
#
# `models_dir` may be empty ("") to skip vendoring the safetensors — the
# default at release time, when the model files live in
#   ~/Library/Application Support/SA3 Variations/models/
# (managed by the end-user) instead of being embedded per-bundle.
#
# `codesign_identity` is the argument passed to `codesign --sign`. Default
# "-" produces an ad-hoc signature (fine for local dev). For a release,
# pass a Developer ID Application identity like
#   "Developer ID Application: Your Name (TEAMID)"
#
# `entitlements_plist` is only used when identity != "-". When set, the
# outermost bundle signature gets `--options=runtime --timestamp
# --entitlements <plist>` so the resulting build is notarisation-eligible.
# Nested Mach-O dylibs get `--options=runtime --timestamp` but no
# entitlements (they inherit at load time from the loading binary).
set -euo pipefail

bundle="$1"
mlx_lib_dir="$2"
sp_lib_dir="$3"
models_dir="${4:-}"
dev_rpath="${5:-}"
codesign_identity="${6:--}"
entitlements_plist="${7:-}"

if [[ ! -d "$bundle" ]]; then
    echo "vendor_bundle: bundle not found: $bundle" >&2
    exit 1
fi

# JUCE names every binary "SA3 Variations" regardless of format.
binary="$bundle/Contents/MacOS/SA3 Variations"
if [[ ! -f "$binary" ]]; then
    echo "vendor_bundle: binary not found at $binary" >&2
    exit 1
fi

frameworks="$bundle/Contents/Frameworks"
resources="$bundle/Contents/Resources"
models_dst="$resources/models"
mkdir -p "$frameworks"

# ── Copy dylibs + Metal kernel library ────────────────────────────────
# `cp -c` uses an APFS clone — instant, no extra disk usage until modified.
# Falls back to a regular copy if the source/dest cross filesystems.
clone_into_bundle() {
    local src="$1" dst="$2"
    if [[ ! -f "$dst" ]] || [[ "$src" -nt "$dst" ]]; then
        # `cp -c` succeeds only on same-volume APFS; fall back otherwise.
        cp -c "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
        chmod u+w "$dst"
    fi
}

clone_into_bundle "$mlx_lib_dir/libmlx.dylib"          "$frameworks/libmlx.dylib"
clone_into_bundle "$mlx_lib_dir/libjaccl.dylib"        "$frameworks/libjaccl.dylib"
clone_into_bundle "$mlx_lib_dir/mlx.metallib"          "$frameworks/mlx.metallib"
clone_into_bundle "$sp_lib_dir/libsentencepiece.0.dylib" \
                  "$frameworks/libsentencepiece.0.dylib"

# Make libsentencepiece reachable via @rpath like the MLX dylibs.
install_name_tool -id "@rpath/libsentencepiece.0.dylib" \
    "$frameworks/libsentencepiece.0.dylib" 2>/dev/null || true

# ── Patch the plugin binary ───────────────────────────────────────────
# 1) Drop the dev-tree LC_RPATH (the venv path baked in at link time). If it
#    isn't present (e.g. on re-runs) install_name_tool exits non-zero, hence
#    the `|| true`.
if [[ -n "$dev_rpath" ]]; then
    install_name_tool -delete_rpath "$dev_rpath" "$binary" 2>/dev/null || true
fi

# 2) Add @loader_path/../Frameworks so libmlx/libjaccl/libsentencepiece
#    resolve via @rpath from inside the bundle.
if ! otool -l "$binary" | grep -A2 LC_RPATH | grep -q "@loader_path/../Frameworks"; then
    install_name_tool -add_rpath "@loader_path/../Frameworks" "$binary"
fi

# 3) Rewrite the hardcoded Homebrew sentencepiece path → @rpath lookup so it
#    finds the bundled copy.
install_name_tool -change \
    "$sp_lib_dir/libsentencepiece.0.dylib" \
    "@rpath/libsentencepiece.0.dylib" \
    "$binary" 2>/dev/null || true

# ── Optionally vendor model weights ───────────────────────────────────
if [[ -n "$models_dir" && -d "$models_dir" ]]; then
    mkdir -p "$models_dst"
    for f in t5gemma_f16.safetensors \
             dit_medium_f16.safetensors \
             same_l_encoder_f32.safetensors \
             same_l_decoder_f32.safetensors; do
        if [[ -f "$models_dir/$f" ]]; then
            clone_into_bundle "$models_dir/$f" "$models_dst/$f"
        else
            echo "vendor_bundle: warn — model missing: $models_dir/$f" >&2
        fi
    done
else
    # Vendoring is off but a previous build may have left ~6 GB of stale
    # safetensors in Contents/Resources/models. Strip them so the bundle
    # we ship matches the lean default layout (models live in
    # ~/Library/Application Support/SA3 Variations/models/ at runtime).
    if [[ -d "$models_dst" ]]; then
        rm -rf "$models_dst"
    fi
fi

# ── Re-sign everything ad-hoc ─────────────────────────────────────────
# Adding/replacing files inside a signed bundle invalidates the seal, so
# sign each nested dylib first then the bundle (codesign --deep is
# deprecated). Library validation in hardened-runtime hosts demands every
# loaded dylib carry the same team identity as the loader — ad-hoc on both
# sides satisfies that.
# mlx.metallib is a Metal-compiled shader library; codesign treats it as a
# subcomponent of the bundle and refuses to seal the outer .vst3/.component
# unless every nested Mach-O-style payload (dylibs *and* metallib) carries
# its own signature first.
# Common codesign args. For ad-hoc identity ("-") we keep things minimal
# (no hardened runtime, no timestamp — local dev). For a real Developer
# ID identity, hardened-runtime + a secure timestamp are required for the
# bundle to be notarisation-eligible later.
common_sign_args=(--force --sign "$codesign_identity")
runtime_sign_args=("${common_sign_args[@]}")
if [[ "$codesign_identity" != "-" ]]; then
    runtime_sign_args+=(--options=runtime --timestamp)
fi

# Inner dylibs: hardened-runtime + timestamp, no entitlements.
for f in "$frameworks"/*.dylib; do
    [[ -f "$f" ]] && codesign "${runtime_sign_args[@]}" "$f"
done
# Metal shader library: just sign it. Not a Mach-O CPU binary, so no
# runtime flag; codesign refuses --options=runtime for it.
for f in "$frameworks"/*.metallib; do
    [[ -f "$f" ]] && codesign "${common_sign_args[@]}" "$f"
done

# Outermost bundle: hardened runtime + timestamp + (when supplied) the
# entitlements plist. Without entitlements the standalone runs but MLX
# crashes inside hardened runtime, so always pair entitlements with a
# Developer ID identity for release builds.
bundle_sign_args=("${runtime_sign_args[@]}")
if [[ "$codesign_identity" != "-" && -n "$entitlements_plist" && -f "$entitlements_plist" ]]; then
    bundle_sign_args+=(--entitlements "$entitlements_plist")
fi
codesign "${bundle_sign_args[@]}" "$bundle"
echo "vendor_bundle: ok — $bundle  (identity: $codesign_identity)"
