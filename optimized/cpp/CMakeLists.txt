cmake_minimum_required(VERSION 3.20)
project(sa3_cpp_spike CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Point at the MLX package shipped inside the project venv.  When you eventually
# vendor MLX into the plugin tree, swap this for an in-tree path.
set(MLX_VENV_CMAKE
    "${CMAKE_SOURCE_DIR}/../mlx/.venv/lib/python3.11/site-packages/mlx/share/cmake/MLX"
    CACHE PATH "Location of MLX CMake config (from `pip install mlx`)")

list(PREPEND CMAKE_PREFIX_PATH "${MLX_VENV_CMAKE}")
find_package(MLX REQUIRED)

add_library(sa3_pipeline STATIC sa3_pipeline.cpp)
target_link_libraries(sa3_pipeline PUBLIC mlx)
target_include_directories(sa3_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(samel STATIC samel_common.cpp samel_encoder.cpp samel_decoder.cpp)
target_link_libraries(samel PUBLIC mlx)
target_include_directories(samel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(t5gemma STATIC t5gemma.cpp)
target_link_libraries(t5gemma PUBLIC mlx)
target_include_directories(t5gemma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dit STATIC dit.cpp)
target_link_libraries(dit PUBLIC mlx)
target_include_directories(dit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# End-to-end orchestrator: ties T5Gemma + conditioner + DiT + SAME-L decoder
# into a single Pipeline and exposes a CLI.  Depends on every other static lib.
add_library(sa3_orchestrator STATIC sa3_orchestrator.cpp)
target_link_libraries(sa3_orchestrator PUBLIC t5gemma dit samel sa3_pipeline)
target_include_directories(sa3_orchestrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# SentencePiece — used for the T5Gemma::encode wrapper that tokenizes strings
# before the encoder forward pass. Located via brew prefix for the spike; for
# production we'd vendor the SentencePiece source instead of relying on brew.
set(SENTENCEPIECE_PREFIX "/opt/homebrew/opt/sentencepiece" CACHE PATH
    "Root of SentencePiece install (containing include/ and lib/)")
find_path(SP_INCLUDE_DIR
    NAMES sentencepiece_processor.h
    HINTS "${SENTENCEPIECE_PREFIX}/include")
find_library(SP_LIBRARY
    NAMES sentencepiece
    HINTS "${SENTENCEPIECE_PREFIX}/lib")
if(NOT SP_INCLUDE_DIR OR NOT SP_LIBRARY)
    message(FATAL_ERROR
        "SentencePiece not found. Run `brew install sentencepiece`, or set "
        "-DSENTENCEPIECE_PREFIX=/path/to/sentencepiece.")
endif()
target_include_directories(t5gemma PRIVATE ${SP_INCLUDE_DIR})
target_link_libraries(t5gemma PUBLIC ${SP_LIBRARY})

add_executable(spike main.cpp)
target_link_libraries(spike PRIVATE mlx)

add_executable(test_sa3_pipeline test_sa3_pipeline.cpp)
target_link_libraries(test_sa3_pipeline PRIVATE sa3_pipeline)

add_executable(test_samel_encoder test_samel_encoder.cpp)
target_link_libraries(test_samel_encoder PRIVATE samel)

add_executable(test_samel_decoder test_samel_decoder.cpp)
target_link_libraries(test_samel_decoder PRIVATE samel)

add_executable(test_t5gemma test_t5gemma.cpp)
target_link_libraries(test_t5gemma PRIVATE t5gemma samel)

add_executable(test_t5gemma_encode test_t5gemma_encode.cpp)
target_link_libraries(test_t5gemma_encode PRIVATE t5gemma samel)

add_executable(test_dit test_dit.cpp)
target_link_libraries(test_dit PRIVATE dit samel)

add_executable(sa3_cli sa3_cli.cpp)
target_link_libraries(sa3_cli PRIVATE sa3_orchestrator)

# The MLX dylib lives in the venv; embed its directory as an rpath so the
# binaries find libmlx.dylib without DYLD_LIBRARY_PATH gymnastics.
get_filename_component(MLX_LIB_DIR "${MLX_VENV_CMAKE}/../../../lib" ABSOLUTE)
foreach(tgt spike test_sa3_pipeline test_samel_encoder test_samel_decoder test_t5gemma test_t5gemma_encode test_dit sa3_cli)
    set_target_properties(${tgt} PROPERTIES
        INSTALL_RPATH "${MLX_LIB_DIR}"
        BUILD_RPATH   "${MLX_LIB_DIR}")
endforeach()
