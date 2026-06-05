#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:?IMAGE_TAG env var is required}"

echo "::group::Verifying Docker image"

echo "Testing Docker image: $IMAGE_TAG"

docker run --rm "$IMAGE_TAG" bash -c '
  echo "=== Cross-Compilation Environment Verification ==="
  echo ""
  echo "Architecture:"
  uname -m
  echo ""
  echo "Cross-compiler:"
  aarch64-linux-gnu-gcc --version | head -n1
  echo ""
  echo "CMake:"
  cmake --version | head -n1
  echo ""
  echo "Toolchain file:"
  ls -l /toolchains/aarch64-jetson.cmake
  echo ""
  echo "Sysroot:"
  ls -ld $SYSROOT
  echo ""
  echo "CUDA (in sysroot):"
  ls -ld $SYSROOT/usr/local/cuda 2>/dev/null || echo "CUDA symlink not found"
  echo ""
  echo "=== Verification Complete ==="
'

echo "::endgroup::"
