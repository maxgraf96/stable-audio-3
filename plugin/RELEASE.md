# SA3 Variations — Release Notes

How to produce signed, notarised release builds of the standalone + VST3 + AU and what end-users need on their machines. Notarisation is optional — skip it and you'll ship signed-but-unstapled bundles that show a first-launch Gatekeeper warning. With notarisation on, the warning disappears entirely.

## Builder prerequisites (your machine only)

- Apple Silicon (arm64) Mac on macOS 13.5+.
- Xcode command-line tools.
- The repo's existing dev setup: MLX venv at `optimized/mlx/.venv/` and `brew install sentencepiece`. These are **build-time only** — end-users never see them; `vendor_bundle.sh` embeds the dylibs into each plugin bundle.
- A **Developer ID Application** certificate in your login keychain. Confirm with:
  ```
  security find-identity -v -p codesigning | grep "Developer ID Application"
  ```
- An **app-specific password** generated at <https://appleid.apple.com> → *Sign-In and Security* → *App-Specific Passwords*.

### Setting up credentials

The release scripts read everything from `plugin/.env`. Copy the template and fill in:

```bash
cp plugin/.env.example plugin/.env
$EDITOR plugin/.env
```

Required keys (`.env.example` has details on what each one is + where to get it):

| Key | Example |
|-----|---------|
| `APPLE_ID` | `you@example.com` |
| `APPLE_TEAM_ID` | `ABCDE12345` |
| `APPLE_APP_PASSWORD` | `xxxx-xxxx-xxxx-xxxx` |
| `SA3_CODESIGN_IDENTITY` | `Developer ID Application: Your Name (ABCDE12345)` |

`plugin/.env` is gitignored, so it never leaves your machine.

## End-user prerequisites

- Apple Silicon (arm64) Mac on **macOS 13.5+**. We deliberately drop x86_64 because MLX is Apple Silicon-only.
- The model files placed in `~/Library/Application Support/SA3 Variations/models/`:
  ```
  ~/Library/Application Support/SA3 Variations/models/
    t5gemma_f16.safetensors            (~540 MB)
    dit_medium_f16.safetensors         (~2.7 GB)
    same_l_encoder_f32.safetensors     (~1.6 GB)
    same_l_decoder_f32.safetensors     (~1.6 GB)
  ```
  Total ~6.4 GB. Ship these as a separate download (zip or `.tar.gz`); the plugin bundles themselves stay around **150 MB each** without them.

## Build the release artefacts

```bash
plugin/scripts/build_release.sh             # incremental
plugin/scripts/build_release.sh --clean     # nuke build/ first
```

The script sources `plugin/.env`, runs `cmake` with `SA3_CODESIGN_IDENTITY`, and builds all three formats. The post-build vendor step:
1. Copies `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`, `libsentencepiece.0.dylib` into each bundle's `Contents/Frameworks/`.
2. Strips the dev-tree rpath and adds `@loader_path/../Frameworks`.
3. Re-signs every nested Mach-O and the bundle itself with `SA3_CODESIGN_IDENTITY`. When that's a real Developer ID, the bundle gets `--options=runtime --timestamp` and the outermost signature picks up `plugin/entitlements.plist` — that's the configuration the notary service requires.

Artefacts land at:
- `build/SA3_artefacts/Release/Standalone/SA3 Variations.app`
- `build/SA3_artefacts/Release/VST3/SA3 Variations.vst3`
- `build/SA3_artefacts/Release/AU/SA3 Variations.component`

Verify each with:
```bash
codesign --verify --deep --strict --verbose=2 "<bundle path>"
spctl -a -vv "<bundle path>"      # standalone only; will warn-with-Developer-ID, reject without
otool -L "<binary inside bundle>" # nothing outside /System or /usr/lib or @rpath
otool -l "<binary>" | grep -A2 LC_BUILD_VERSION    # minos 13.5, not 15.x
```

## Packaging the plugin bundles

```bash
plugin/scripts/prepare_release_assets.sh                 # notarise + package
plugin/scripts/prepare_release_assets.sh --no-notarize   # signed but unstapled
```

The script reads the same `plugin/.env` and:
1. For each bundle, runs `notarize.sh` (zip → `notarytool submit --wait` → `stapler staple`). Each submit blocks on Apple's service for ~1–10 min, so the whole run takes 5–30 min for the three formats.
2. Bundles the (now stapled) artefacts into the release zip.

Notarisation is auto-skipped if the `.env` values are still template placeholders, so the script also works for contributors who only have an ad-hoc local build.

Reads `plugin/build/SA3_artefacts/Release/` and emits two files in `plugin/build/release_assets/`:

| File | Purpose |
|------|---------|
| `SA3-Variations-plugins.zip` | the three (stapled, if notarised) bundles + a `README.txt` |
| `install_models.sh` | end-user download script (pulls from HuggingFace) |

## Hosting the safetensors on HuggingFace

The four model files are hosted at <https://huggingface.co/maxgraf/sa3-variations-models> — `install_models.sh` is hardcoded to point at that repo (`DEFAULT_REPO`).

For new versions, just re-upload over the existing files via the HF web UI or:

```bash
huggingface-cli upload maxgraf/sa3-variations-models \
    optimized/mlx/models/mlx/t5gemma_f16.safetensors           t5gemma_f16.safetensors
huggingface-cli upload maxgraf/sa3-variations-models \
    optimized/mlx/models/mlx/dit_medium_f16.safetensors        dit_medium_f16.safetensors
huggingface-cli upload maxgraf/sa3-variations-models \
    optimized/mlx/models/mlx/same_l_encoder_f32.safetensors    same_l_encoder_f32.safetensors
huggingface-cli upload maxgraf/sa3-variations-models \
    optimized/mlx/models/mlx/same_l_decoder_f32.safetensors    same_l_decoder_f32.safetensors
```

`install_models.sh` resolves at runtime from the `main` revision, so users always get the latest upload. To pin a release to a specific snapshot, pass a commit SHA as the second arg to the script, or set `DEFAULT_REV` in the script before running `prepare_release_assets.sh`.

## Uploading to GitHub Releases

1. Tag the commit and push: `git tag v0.1.0 && git push --tags`
2. Create a release on GitHub for that tag.
3. Drag both files from `plugin/build/release_assets/` into the release's asset uploader.
4. Publish.

## End-user install steps

1. Open the release page on GitHub. Download:
   - `SA3-Variations-plugins.zip`
   - `install_models.sh`
2. In Terminal:
   ```
   chmod +x install_models.sh && ./install_models.sh
   ```
   Pulls all four safetensors (~6.4 GB) from HuggingFace into `~/Library/Application Support/SA3 Variations/models/`. Idempotent — re-run to resume after interruption.
3. Unzip the plugins archive and drag:
   - `SA3 Variations.app` → `/Applications/`
   - `SA3 Variations.vst3` → `~/Library/Audio/Plug-Ins/VST3/`
   - `SA3 Variations.component` → `~/Library/Audio/Plug-Ins/Components/`
4. If the release was notarised, the bundles launch with no warning the first time. If not, **first launch** shows a Gatekeeper warning — bypass via **right-click** → **Open** → confirm; macOS remembers per bundle.

## Where models are looked up at runtime

Resolution order (see `resolveModelsDir()` in `plugin/Source/VariationsEngine.cpp`):
1. `$SA3_MODELS_DIR` — dev override.
2. `<bundle>/Contents/Resources/models/` — only present if you built with `-DSA3_VENDOR_MODELS=ON` (fat self-contained bundle, mainly useful for local CI / testing).
3. `~/Library/Application Support/SA3 Variations/models/` — production default, where end-users drop the model archive.

## Known limitations

- **arm64 only**: MLX is Apple Silicon-only. Intel Macs cannot run this.
- **macOS 13.5+** only — driven by MLX's own minimum.
