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
    PRODUCT_NAME              "SA3"
    COMPANY_NAME              "MaxGraf"
    BUNDLE_ID                 "com.maxgraf.sa3"
    PLUGIN_MANUFACTURER_CODE  "Maxg"
    PLUGIN_CODE               "Sa3p"
    FORMATS                   AU VST3 Standalone
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD   FALSE
    # Standalone app convenience: open in standalone with stereo I/O.
    AU_MAIN_TYPE              kAudioUnitType_Effect
    VST3_CATEGORIES           "Fx"
)

juce_generate_juce_header(SA3)

target_sources(SA3 PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(SA3 PUBLIC
    # Trim JUCE's optional features we don't need — smaller binary, fewer deps.
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

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
