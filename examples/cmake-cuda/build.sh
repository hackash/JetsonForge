
#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

: "${CMAKE_TOOLCHAIN_FILE:=/toolchains/aarch64-jetson.cmake}"

cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
file build/cuda_hello
readelf -h build/cuda_hello | grep 'Class\|Machine'
echo "Built aarch64 binary should report 'Machine: AArch64'"
