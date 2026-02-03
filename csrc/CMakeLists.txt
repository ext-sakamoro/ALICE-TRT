# ALICE-TRT: CMake build for TensorRT Plugin + CUDA Kernels
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
#            -DTENSORRT_ROOT=/usr/local/TensorRT
#   make -j$(nproc)
#
# Targets:
#   alice_trt_cuda   — Static library (CUDA kernels + C API, no TensorRT dependency)
#   alice_trt_plugin — Shared library (TensorRT plugin, requires NvInfer.h)

cmake_minimum_required(VERSION 3.18)
project(alice-trt-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architectures: Pascal (6.1), Volta (7.0), Turing (7.5),
#                     Ampere (8.0/8.6), Ada (8.9), Hopper (9.0)
set(CMAKE_CUDA_ARCHITECTURES "61;70;75;80;86;89;90")

# ================================================================
# Target 1: alice_trt_cuda (standalone, no TensorRT)
#
# This is what Rust's build.rs links against.
# Contains: BitNet CUDA kernels (wmma + dp4a + popc) + C API
# ================================================================

add_library(alice_trt_cuda STATIC
    bitnet_kernel.cu
)

target_include_directories(alice_trt_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(alice_trt_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
    >
)

# ================================================================
# Target 2: alice_trt_plugin (TensorRT plugin, optional)
#
# Only builds if TensorRT is found.
# ================================================================

set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT installation")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu
    DOC "TensorRT include directory"
)

find_library(NVINFER_LIB nvinfer
    PATHS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu
    DOC "TensorRT nvinfer library"
)

if(TENSORRT_INCLUDE_DIR AND NVINFER_LIB)
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")

    add_library(alice_trt_plugin SHARED
        plugin/alice_ternary_plugin.cpp
        plugin/alice_ternary_plugin.cu
        bitnet_kernel.cu
    )

    target_include_directories(alice_trt_plugin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
        ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(alice_trt_plugin PRIVATE
        ${NVINFER_LIB}
    )

    target_compile_options(alice_trt_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -O3
        >
    )

    set_target_properties(alice_trt_plugin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    message(WARNING "TensorRT not found. Skipping plugin build.")
    message(WARNING "Set -DTENSORRT_ROOT=/path/to/TensorRT to enable.")
endif()
