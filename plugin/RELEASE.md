# SA3 Variations — Release Notes

How to produce signed release builds of the standalone + VST3 + AU and what end-users need on their machines. Notarisation is **out of scope**: this is a demo release; Gatekeeper will show a first-launch warning that users dismiss via right-click → **Open**.

## Builder prerequisites (your machine only)

- Apple Silicon (arm64) Mac on macOS 13.5+.
- Xcode command-line tools.
- The repo's existing dev setup: MLX venv at `optimized/mlx/.venv/` and `brew install sentencepiece`. These are **build-time only** — end-users never see them; `vendor_bundle.sh` embeds the dylibs into each plugin bundle.
- A **Developer ID Application** certificate in your login keychain. Confirm with:
  ```
  security find-identity -v -p codesigning | grep "Developer ID Application"
  ```
  Take note of the full identity string, e.g. `Developer ID Application: Max Graf (XXXXXXXXXX)`.

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
cd plugin
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DSA3_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"

cmake --build . --target SA3_Standalone SA3_VST3 SA3_AU -j
```

The post-build vendor step:
1. Copies `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`, `libsentencepiece.0.dylib` into each bundle's `Contents/Frameworks/`.
2. Strips the dev-tree rpath and adds `@loader_path/../Frameworks`.
3. Re-signs every nested Mach-O and the bundle itself with `SA3_CODESIGN_IDENTITY`.

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
plugin/scripts/prepare_release_assets.sh
```

Reads `plugin/build/SA3_artefacts/Release/` and emits two files in `plugin/build/release_assets/`:

| File | Purpose |
|------|---------|
| `SA3-Variations-plugins.zip` | the three bundles + a `README.txt` |
| `install_models.sh` | end-user download script (pulls from HuggingFace) |

## Hosting the safetensors on HuggingFace

One-time setup:

1. Create a new HuggingFace model repo (e.g. `maxgraf96/sa3-variations-models`).
2. Upload the four files at the top level of the repo (web UI or `huggingface-cli upload`):
   ```
   t5gemma_f16.safetensors
   dit_medium_f16.safetensors
   same_l_encoder_f32.safetensors
   same_l_decoder_f32.safetensors
   ```
3. If the repo slug isn't `maxgraf96/sa3-variations-models`, edit `DEFAULT_REPO` at the top of `plugin/scripts/install_models.sh` (and re-run `prepare_release_assets.sh` so the bundled copy reflects the change).

Subsequent releases reuse the same HF repo — bump `DEFAULT_REV` only if you pin to a specific revision, otherwise `main` keeps pointing at the latest upload.

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
4. **First launch** of the standalone (or first scan of the AU/VST3 in a DAW) shows a Gatekeeper warning because we don't notarise. Bypass:
   - **Right-click** → **Open** → confirm. macOS remembers per bundle.
   - DAWs typically run plugin scans non-interactively; if a host refuses to load on first scan, launch the standalone once to whitelist the signing identity, then rescan.

## Where models are looked up at runtime

Resolution order (see `resolveModelsDir()` in `plugin/Source/VariationsEngine.cpp`):
1. `$SA3_MODELS_DIR` — dev override.
2. `<bundle>/Contents/Resources/models/` — only present if you built with `-DSA3_VENDOR_MODELS=ON` (fat self-contained bundle, mainly useful for local CI / testing).
3. `~/Library/Application Support/SA3 Variations/models/` — production default, where end-users drop the model archive.

## Known limitations

- **No notarisation**: every fresh download triggers a Gatekeeper warning. Acceptable for a demo; rolling notarisation in later is a one-line addition to vendor_bundle.sh (`xcrun notarytool submit`).
- **arm64 only**: MLX is Apple Silicon-only. Intel Macs cannot run this.
- **macOS 13.5+** only — driven by MLX's own minimum.
