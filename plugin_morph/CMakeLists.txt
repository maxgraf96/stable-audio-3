# SA3 Morph — real-time style-morph JUCE plugin (sibling of plugin/).
#
# Loads a loop and continuously morphs its style while playing, driven by the
# C++ streaming engine (optimized/cpp: stream_runtime -> StreamGenerator +
# AudioRing on the sm-music DiT + SAME-S codec). WebView UI, same patterns as
# the Variations plugin (plugin/).
#
# Build:
#   cmake -S plugin_morph -B plugin_morph/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build plugin_morph/build -j
cmake_minimum_required(VERSION 3.22)

# macOS deployment target MUST be set before project() (see plugin/CMakeLists.txt).
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.5" CACHE STRING "")
endif()

project(SA3MorphPlugin VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Reusable assets live in the sibling Variations plugin dir.
set(SA3_VARIATIONS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../plugin")

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
juce_add_plugin(SA3Morph
    PRODUCT_NAME              "SA3 Realtime"
    COMPANY_NAME              "MaxGraf"
    BUNDLE_ID                 "com.maxgraf.sa3realtime"
    PLUGIN_MANUFACTURER_CODE  "Maxg"
    PLUGIN_CODE               "Sa3r"
    FORMATS                   AU VST3 Standalone
    IS_SYNTH                  FALSE          # Fx|Generator (see plugin/ rationale)
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE        # WebView is the keyboard target
    COPY_PLUGIN_AFTER_BUILD   TRUE
)

juce_generate_juce_header(SA3Morph)

target_sources(SA3Morph PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/MorphEngine.cpp
)
if(APPLE)
    target_sources(SA3Morph PRIVATE Source/MacClipboard.mm)
endif()

target_compile_definitions(SA3Morph PUBLIC
    JUCE_WEB_BROWSER=1
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

# ── Embedded UI resources (BinaryData) ────────────────────────────────
# Dev iteration: set SA3_PLUGIN_RESOURCES=<repo>/plugin_morph/Resources to
# serve from disk without rebuilding.
file(GLOB SA3M_WEB_RESOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.html"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.js"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/*.css")
juce_add_binary_data(SA3MorphResources
    HEADER_NAME BinaryData.h
    NAMESPACE BinaryData
    SOURCES ${SA3M_WEB_RESOURCES})

# ── SA3 C++ port (optimized/cpp/) ─────────────────────────────────────
# Brings in stream_runtime (-> stream + sa3_orchestrator -> dit_sm, sames,
# t5gemma, sa3_pipeline, mlx). Defines MLX_VENV_CMAKE / SENTENCEPIECE_PREFIX /
# MLX_LIB_DIR used below.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../optimized/cpp
                 ${CMAKE_BINARY_DIR}/sa3_cpp_build)

target_link_libraries(SA3Morph PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_extra            # WebBrowserComponent + native bridge
    SA3MorphResources               # baked HTML/CSS/JS (BinaryData.h)
    stream_runtime                  # AudioRing + StreamGenerator
    sa3_orchestrator                # Pipeline loader + WAV read/write helpers
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# ── MLX dylib rpath (dev) ─────────────────────────────────────────────
get_filename_component(MLX_LIB_DIR "${MLX_VENV_CMAKE}/../../../lib" ABSOLUTE)
foreach(fmt SA3Morph_AU SA3Morph_VST3 SA3Morph_Standalone)
    if(TARGET ${fmt})
        set_target_properties(${fmt} PROPERTIES
            INSTALL_RPATH "${MLX_LIB_DIR}"
            BUILD_RPATH   "${MLX_LIB_DIR}")
    endif()
endforeach()

# ── Post-build: vendor dylibs (+ optionally models) into each bundle ──
# Reuses the Variations plugin's vendor_bundle.sh + entitlements verbatim.
option(SA3_VENDOR_MODELS "Clone MLX safetensors into each plugin bundle" OFF)
set(SA3_CODESIGN_IDENTITY "-" CACHE STRING
    "Codesign identity for bundle re-signing (\"-\" for ad-hoc)")
set(SA3_ENTITLEMENTS "${SA3_VARIATIONS_DIR}/entitlements.plist" CACHE FILEPATH
    "Entitlements plist applied to the outermost bundle signature")

set(SENTENCEPIECE_LIB_DIR "${SENTENCEPIECE_PREFIX}/lib")
if(SA3_VENDOR_MODELS)
    set(SA3_MODELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../optimized/mlx/models/mlx")
else()
    set(SA3_MODELS_SRC "")
endif()

set(VENDOR_SCRIPT "${SA3_VARIATIONS_DIR}/cmake/vendor_bundle.sh")
set(_artefacts "${CMAKE_BINARY_DIR}/SA3Morph_artefacts/$<CONFIG>")
set(_user_plugins "$ENV{HOME}/Library/Audio/Plug-Ins")
set(_fmt_targets SA3Morph_AU SA3Morph_VST3 SA3Morph_Standalone)
set(_fmt_kinds   AU          VST3          Standalone)
set(_fmt_bundles "SA3 Realtime.component" "SA3 Realtime.vst3" "SA3 Realtime.app")
set(_fmt_install_dirs
    "${_user_plugins}/Components"
    "${_user_plugins}/VST3"
    "")
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
