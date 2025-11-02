cmake_minimum_required(VERSION 3.16)
project(jp6_trt_cpp_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ---- Sysroot from env (set by your cross image) ----
set(_SYSROOT $ENV{SYSROOT})

# ---- Candidate paths for CUDA ----
set(_CUDA_INC_CAND
  "${_SYSROOT}/usr/local/cuda/include"
  "${_SYSROOT}/usr/local/cuda/targets/aarch64-linux/include"
)
set(_CUDA_LIB_CAND
  "${_SYSROOT}/usr/local/cuda/lib64"
  "${_SYSROOT}/usr/local/cuda/targets/aarch64-linux/lib"
)

# ---- Find CUDA runtime headers/libs in the SYSROOT ----
find_path(CUDA_RUNTIME_INCLUDE
  NAMES cuda_runtime.h
  HINTS ${_CUDA_INC_CAND}
  NO_DEFAULT_PATH
)
find_library(CUDART_LIBRARY
  NAMES cudart
  HINTS ${_CUDA_LIB_CAND}
  NO_DEFAULT_PATH
)

# ---- Find TensorRT headers/libs in the SYSROOT ----
find_path(TRT_INCLUDE
  NAMES NvInfer.h
  HINTS
    "${_SYSROOT}/usr/include/aarch64-linux-gnu"
    "${_SYSROOT}/usr/include"
  NO_DEFAULT_PATH
)
find_library(NVINFER_LIBRARY
  NAMES nvinfer
  HINTS
    "${_SYSROOT}/usr/lib/aarch64-linux-gnu"
    "${_SYSROOT}/usr/lib"
    "${_SYSROOT}/usr/local/lib"
  NO_DEFAULT_PATH
)

# ---- Sanity checks ----
if(NOT CUDA_RUNTIME_INCLUDE OR NOT CUDART_LIBRARY)
  message(FATAL_ERROR "CUDA headers/libs not found in SYSROOT. Make sure CUDA dev packages are installed in the exported sysroot.")
endif()
if(NOT TRT_INCLUDE OR NOT NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/libs not found in SYSROOT. Install TensorRT dev packages in the sysroot.")
endif()

# ---- Target ----
add_executable(device_query main.cpp)
target_include_directories(device_query PRIVATE
  "${CUDA_RUNTIME_INCLUDE}"
  "${TRT_INCLUDE}"
)

# Link against absolute libs found in sysroot
target_link_libraries(device_query PRIVATE
  "${NVINFER_LIBRARY}"
  "${CUDART_LIBRARY}"
)

# Optional: PIC and some compilers’ defaults
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
