# SA3 JUCE plugin — Phase 1 (skeleton, no MLX wiring yet).
#
# Targets: AU + VST3 + Standalone (macOS only — MLX is Apple-Silicon-only).
# JUCE is pulled at configure time via FetchContent; first configure takes
# several minutes (downloads ~500 MB JUCE source).
#
# Build:
#   cmake -S plugin -B plugin/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build plugin/build -j
#
# Sanity-test:
#   plugin/build/SA3_artefacts/Release/Standalone/SA3.app/Contents/MacOS/SA3
cmake_minimum_required(VERSION 3.22)
project(SA3Plugin VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# macOS deployment target — Apple Silicon DAWs typically need 11+.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
endif()

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
    FORMATS                   AU VST3 Standalone
    # We're a generator (sample-variations source), not an effect — no
    # audio input. IS_SYNTH=TRUE picks the right AU type
    # (kAudioUnitType_MusicDevice) and VST3 category (Instrument|Synth) so
    # hosts file the plugin under instruments rather than effects, and the
    # standalone wrapper stops complaining about feedback loops.
    IS_SYNTH                  TRUE
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

target_compile_definitions(SA3 PUBLIC
    # WebView UI (HTML/CSS/JS via WebBrowserComponent + native bridge); CURL
    # stays off — we don't talk to the network at runtime.
    JUCE_WEB_BROWSER=1
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
# Phase 2: pull in the bit-exact C++ inference path. The orchestrator
# target transitively depends on t5gemma, dit, samel, sa3_pipeline, and
# mlx. After this add_subdirectory call, sa3_orchestrator (along with its
# test executables and dependencies) is part of the plugin's build.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../optimized/cpp
                 ${CMAKE_BINARY_DIR}/sa3_cpp_build)

target_link_libraries(SA3 PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_extra            # WebBrowserComponent + native bridge
    SA3Resources                    # baked HTML/CSS/JS (BinaryData.h)
    sa3_orchestrator
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

# libmlx.dylib lives inside the project venv. Embed an rpath into the
# plugin bundles so they can find it at runtime without DYLD_LIBRARY_PATH.
# When we ship the plugin, libmlx.dylib will be vendored into the bundle's
# Contents/Frameworks/ and the rpath swapped to @loader_path/../Frameworks.
get_filename_component(MLX_LIB_DIR "${MLX_VENV_CMAKE}/../../../lib" ABSOLUTE)
foreach(fmt SA3_AU SA3_VST3 SA3_Standalone)
    if(TARGET ${fmt})
        set_target_properties(${fmt} PROPERTIES
            INSTALL_RPATH "${MLX_LIB_DIR}"
            BUILD_RPATH   "${MLX_LIB_DIR}")
    endif()
endforeach()
