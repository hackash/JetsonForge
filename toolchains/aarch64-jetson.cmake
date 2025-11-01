# Target triple
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Use host cross-compilers (amd64 binaries)
set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER /usr/bin/aarch64-linux-gnu-gcc)

# Optional (helps some toolchains):
set(CMAKE_C_COMPILER_TARGET   aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# Exact-match sysroot (your exported Jetson userland)
set(SYSROOT $ENV{SYSROOT})
set(CMAKE_SYSROOT "${SYSROOT}")

# Search inside the sysroot
set(CMAKE_FIND_ROOT_PATH "${SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Make the linker see transitive libs in the sysroot
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT
  " -Wl,-rpath-link,${SYSROOT}/usr/lib/aarch64-linux-gnu"
  " -Wl,-rpath-link,${SYSROOT}/usr/lib/aarch64-linux-gnu/tegra"
  " -Wl,-rpath-link,${SYSROOT}/lib/aarch64-linux-gnu"
  " -Wl,-rpath-link,${SYSROOT}/usr/local/lib")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT}")

# CUDA hints inside sysroot (adjust if needed)
set(CUDA_TOOLKIT_ROOT_DIR "${SYSROOT}/usr/local/cuda" CACHE PATH "")

# Prevent try-run of target binaries (compile-only)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
