# SA3 JUCE plugin.
#
# macOS:   AU + VST3 + Standalone, inference via the in-process C++ MLX
#          orchestrator in optimized/cpp (Metal, Apple-Silicon-only).
# Windows: Standalone only, inference via a resident Python worker. The MLX
#          orchestrator is not built at all off Apple — nothing in it compiles
#          without Metal — so optimized/cpp is added conditionally below.
#
# JUCE is pulled at configure time via FetchContent; first configure takes
# several minutes (downloads ~500 MB JUCE source).
#
# Build:
#   cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build plugin/build --config Release -j
#
# Sanity-test:
#   macOS:   plugin/build/SA3_artefacts/Release/Standalone/SA3 Variations.app/Contents/MacOS/SA3 Variations
#   Windows: plugin/build/SA3_artefacts/Release/Standalone/SA3 Variations.exe
cmake_minimum_required(VERSION 3.22)

# macOS deployment target MUST be set before project() — project() bakes
# the target into the compiler's -mmacosx-version-min flag and CMake's
# internal state. Setting it later (as we used to) silently no-op'd and
# binaries demanded macOS 15. 13.5 is MLX's documented minimum and covers
# macOS 13 (Ventura) onward — three OS generations of users.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.5" CACHE STRING "")
endif()

project(SA3Plugin VERSION 0.1.0 LANGUAGES C CXX)

# AU exists only on Apple, and juce_add_plugin errors rather than skipping it
# elsewhere. Windows builds Standalone alone for now — the VST3 needs the
# Python worker wired up before it is worth installing into a DAW.
if(APPLE)
    set(SA3_FORMATS AU VST3 Standalone)
else()
    set(SA3_FORMATS Standalone)
endif()

# The editor's UI is a WebBrowserComponent. On macOS that is WKWebView, always
# present. On Windows the modern backend is WebView2, which JUCE only compiles
# in when this is set — without it Backend::defaultBackend falls back to the
# legacy IE control, which cannot run the UI's JS. The Evergreen runtime ships
# with Edge, so only the build-time SDK (restored via NuGet) is needed here.
if(APPLE)
    set(SA3_NEEDS_WEBVIEW2 FALSE)
else()
    set(SA3_NEEDS_WEBVIEW2 TRUE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── JUCE ─────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

# ── Plugin target ────────────────────────────────────────────────────
juce_add_plugin(SA3
    PRODUCT_NAME              "SA3 Variations"
    COMPANY_NAME              "MaxGraf"
    BUNDLE_ID                 "com.maxgraf.sa3variations"
    PLUGIN_MANUFACTURER_CODE  "Maxg"
    PLUGIN_CODE               "Sa3p"
    FORMATS                   ${SA3_FORMATS}
    NEEDS_WEBVIEW2            ${SA3_NEEDS_WEBVIEW2}
    # IS_SYNTH=FALSE: VST3 category stays Fx|Generator. Setting it to TRUE
    # tags the plugin as Instrument|Synth in VST3, which Ableton Live then
    # refuses to instantiate unless we also declare a MIDI input bus — and
    # we genuinely don't want MIDI on what is an audio-track generator.
    # The "Audio input is muted to avoid feedback loop" banner in the
    # standalone is suppressed independently by leaving the input bus out
    # of BusesProperties() in PluginProcessor.cpp.
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    # FALSE — the editor itself shouldn't accept focus; the WebView (its
    # only child) is the keyboard target. With TRUE the editor would
    # compete with the WebView for focus and JS keydown handlers would
    # never fire.
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    # Auto-install AU + VST3 into ~/Library/Audio/Plug-Ins/{Components,VST3}
    # after each build — fast iteration loop when testing in a DAW.
    COPY_PLUGIN_AFTER_BUILD   TRUE
)

juce_generate_juce_header(SA3)

target_sources(SA3 PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/VariationsEngine.cpp
)
# Exactly one InferenceBackend::create() must be linked — see InferenceBackend.h.
if(APPLE)
    target_sources(SA3 PRIVATE Source/MlxBackend.cpp Source/MacClipboard.mm)
else()
    target_sources(SA3 PRIVATE Source/WorkerBackend.cpp)
endif()

target_compile_definitions(SA3 PUBLIC
    # WebView UI (HTML/CSS/JS via WebBrowserComponent + native bridge); CURL
    # stays off — we don't talk to the network at runtime.
    JUCE_WEB_BROWSER=1
    # Windows-only flag, ignored elsewhere. NEEDS_WEBVIEW2 above only finds and
    # links the SDK; this is what actually compiles the WebView2 backend in.
    # Without it JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE is 0 on Windows
    # (it requires JUCE_WINDOWS && JUCE_USE_WIN_WEBVIEW2), so
    # Options::withResourceProvider doesn't exist and the editor won't compile
    # — the UI is served entirely through that provider.
    JUCE_USE_WIN_WEBVIEW2=1
    # JUCE's NEEDS_WEBVIEW2 links WebView2LoaderStatic.lib, but the runtime path
    # only *uses* the statically linked symbol when this is set — otherwise it
    # LoadLibrary()s a WebView2Loader.dll that a static build never ships, gets
    # null back, and calls through it unchecked (juce_WebBrowserComponent_windows.cpp
    # never null-checks the result). That's an access violation at offset 0 the
    # instant the editor opens, with no diagnostic.
    JUCE_USE_WIN_WEBVIEW2_WITH_STATIC_LINKING=1
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

# ── Embedded UI resources (BinaryData) ────────────────────────────────
# Resources/index.html, app.js, styles.css are baked into BinaryData.h/cpp and
# served by the editor's WebView resource provider. For dev iteration without
# rebuilding, set SA3_PLUGIN_RESOURCES=<repo>/plugin/Resources before launching
# the standalone — the resource provider prefers disk over BinaryData.
file(GLOB SA3_WEB_RESOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.html"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.js"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.css")
juce_add_binary_data(SA3Resources
    HEADER_NAME BinaryData.h
    NAMESPACE BinaryData
    SOURCES ${SA3_WEB_RESOURCES})

# ── SA3 C++ port (optimized/cpp/) ─────────────────────────────────────
# The bit-exact C++ inference path. The orchestrator target transitively
# depends on t5gemma, dit, samel, sa3_pipeline, and mlx. Apple-only: its
# find_package(MLX) is REQUIRED and MLX has no non-Metal build, so adding
# this subdirectory elsewhere fails at configure time. Off Apple the plugin
# links no native inference at all and WorkerBackend drives Python instead.
if(APPLE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../optimized/cpp
                     ${CMAKE_BINARY_DIR}/sa3_cpp_build)
endif()

target_link_libraries(SA3 PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_extra            # WebBrowserComponent + native bridge
    SA3Resources                    # baked HTML/CSS/JS (BinaryData.h)
    $<$<BOOL:${APPLE}>:sa3_orchestrator>
    # WorkerBackend reads dedicated VRAM straight from DXGI. Asking the Python
    # worker instead would mean spawning it and importing torch (~10 s) inside
    # the engine constructor, stalling app startup just to draw a memory notice.
    $<$<BOOL:${WIN32}>:dxgi>
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# ── Smoke test for the Python-worker backend ──────────────────────────
# Exercises WorkerBackend end to end (spawn, handshake, generate, decode)
# without the GUI, which is otherwise only reachable by dropping a file by
# hand. Compiles WorkerBackend.cpp a second time rather than linking the
# plugin's SharedCode, so it stays independent of the audio-plugin glue.
if(NOT APPLE)
    juce_add_console_app(sa3_worker_smoke PRODUCT_NAME "sa3_worker_smoke")
    target_sources(sa3_worker_smoke PRIVATE
        tests/worker_backend_smoke.cpp
        Source/WorkerBackend.cpp)
    target_compile_definitions(sa3_worker_smoke PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_STANDALONE_APPLICATION=1)
    target_link_libraries(sa3_worker_smoke PRIVATE
        juce::juce_core
        juce::juce_audio_formats
        $<$<BOOL:${WIN32}>:dxgi>
        PUBLIC
        juce::juce_recommended_config_flags)
endif()

# ── macOS bundle plumbing ─────────────────────────────────────────────
# rpaths, dylib vendoring and codesigning are all Mach-O concerns. The
# Windows standalone is a plain .exe next to its DLLs, and the Python worker
# it talks to lives in the repo venv, so none of this applies there.
if(APPLE)

    # libmlx.dylib lives inside the project venv; bake that path in as the
    # link-time rpath so the build-tree binary can be run directly during
    # development. For distribution, vendor_bundle.sh (POST_BUILD below)
    # copies libmlx + libjaccl + mlx.metallib + libsentencepiece into the
    # bundle's Contents/Frameworks/, strips this dev rpath, and adds
    # @loader_path/../Frameworks so the bundle is self-contained and loads
    # inside a hardened-runtime DAW.
    get_filename_component(MLX_LIB_DIR "${MLX_VENV_CMAKE}/../../../lib" ABSOLUTE)
    foreach(fmt SA3_AU SA3_VST3 SA3_Standalone)
        if(TARGET ${fmt})
            set_target_properties(${fmt} PROPERTIES
                INSTALL_RPATH "${MLX_LIB_DIR}"
                BUILD_RPATH   "${MLX_LIB_DIR}")
        endif()
    endforeach()

    # ── Post-build: vendor dylibs (and optionally models) into each bundle ──
    # SA3_VENDOR_MODELS=OFF by default: production layout ships ~10 MB plugin
    # bundles plus a separate ~6 GB models download that end-users drop into
    #   ~/Library/Application Support/SA3 Variations/models/
    # (resolveModelsDir() in VariationsEngine.cpp picks them up there). Flip
    # ON to produce fully self-contained bundles for local dev — `cp -c`
    # uses APFS clones so the on-disk overhead is zero until distribution
    # packaging expands them.
    option(SA3_VENDOR_MODELS "Clone MLX safetensors into each plugin bundle" OFF)

    # Codesign identity passed to vendor_bundle.sh. Default "-" = ad-hoc,
    # fine for local dev. For a notarisable release, configure with:
    #   -DSA3_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
    # When the identity is non-ad-hoc, vendor_bundle.sh also applies
    # --options=runtime --timestamp and (on the outermost bundle) the
    # entitlements file below — that's what makes the build notarisation-
    # eligible. See plugin/RELEASE.md for the full notarisation workflow.
    set(SA3_CODESIGN_IDENTITY "-" CACHE STRING
        "Codesign identity for bundle re-signing (\"-\" for ad-hoc)")
    set(SA3_ENTITLEMENTS "${CMAKE_CURRENT_SOURCE_DIR}/entitlements.plist" CACHE FILEPATH
        "Entitlements plist applied to the outermost bundle signature")

    set(SENTENCEPIECE_LIB_DIR "${SENTENCEPIECE_PREFIX}/lib")
    if(SA3_VENDOR_MODELS)
        set(SA3_MODELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../optimized/mlx/models/mlx")
    else()
        set(SA3_MODELS_SRC "")
    endif()

    set(VENDOR_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor_bundle.sh")
    set(_artefacts "${CMAKE_BINARY_DIR}/SA3_artefacts/$<CONFIG>")

    # JUCE's COPY_PLUGIN_AFTER_BUILD step runs before our POST_BUILD command
    # (it was registered first by juce_add_plugin), so the bundle installed to
    # ~/Library is still un-vendored when we get here. We vendor the build-tree
    # bundle, then vendor the installed copy too so both load in DAWs without
    # a second rebuild.
    set(_user_plugins "$ENV{HOME}/Library/Audio/Plug-Ins")
    set(_fmt_targets SA3_AU SA3_VST3 SA3_Standalone)
    set(_fmt_kinds   AU      VST3     Standalone)
    set(_fmt_bundles "SA3 Variations.component" "SA3 Variations.vst3" "SA3 Variations.app")
    set(_fmt_install_dirs
        "${_user_plugins}/Components"
        "${_user_plugins}/VST3"
        "")  # Standalone has no auto-install location.
    foreach(fmt_target fmt_kind fmt_bundle fmt_install IN ZIP_LISTS
            _fmt_targets _fmt_kinds _fmt_bundles _fmt_install_dirs)
        if(TARGET ${fmt_target})
            add_custom_command(TARGET ${fmt_target} POST_BUILD
                COMMAND "${VENDOR_SCRIPT}"
                        "${_artefacts}/${fmt_kind}/${fmt_bundle}"
                        "${MLX_LIB_DIR}"
                        "${SENTENCEPIECE_LIB_DIR}"
                        "${SA3_MODELS_SRC}"
                        "${MLX_LIB_DIR}"
                        "${SA3_CODESIGN_IDENTITY}"
                        "${SA3_ENTITLEMENTS}"
                COMMENT "Vendoring ${fmt_kind} bundle (dylibs + models + re-sign)"
                VERBATIM)
            if(fmt_install)
                add_custom_command(TARGET ${fmt_target} POST_BUILD
                    COMMAND "${VENDOR_SCRIPT}"
                            "${fmt_install}/${fmt_bundle}"
                            "${MLX_LIB_DIR}"
                            "${SENTENCEPIECE_LIB_DIR}"
                            "${SA3_MODELS_SRC}"
                            "${MLX_LIB_DIR}"
                            "${SA3_CODESIGN_IDENTITY}"
                    COMMENT "Vendoring installed ${fmt_kind} (~/Library copy)"
                    VERBATIM)
            endif()
        endif()
    endforeach()

endif()  # APPLE
