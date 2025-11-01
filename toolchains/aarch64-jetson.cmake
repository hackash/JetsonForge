
# CMake toolchain for Jetson (aarch64) cross-compilation
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(SYSROOT $ENV{SYSROOT})
set(CMAKE_SYSROOT ${SYSROOT})

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Ensure CMake looks into the sysroot for libs/includes/packages
set(CMAKE_FIND_ROOT_PATH  ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# CUDA (JetPack 6.0 uses CUDA 12.2)
# If present in the sysroot, this will let find_package(CUDAToolkit) work.
set(CUDAToolkit_ROOT "${SYSROOT}/usr/local/cuda")
# Or set NVCC explicitly if needed:
# set(CMAKE_CUDA_COMPILER "${SYSROOT}/usr/local/cuda/bin/nvcc")
# Or in project: enable_language(CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 87)  # Orin (Ampere) SM 87 (adjust to your devices)
