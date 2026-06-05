#!/usr/bin/env bash
set -euo pipefail

SYSROOT_BASENAME="${SYSROOT_BASENAME:?SYSROOT_BASENAME env var is required}"
IMAGE_TAG="${IMAGE_TAG:?IMAGE_TAG env var is required}"
ACTION_PATH="${ACTION_PATH:?ACTION_PATH env var is required}"

echo "::group::Building Docker cross-compilation image"

cd "$ACTION_PATH"

echo "Building Docker image: $IMAGE_TAG"
echo "Using sysroot:         $SYSROOT_BASENAME"

docker build \
  --network host \
  --build-arg TAR_ZST_NAME="$SYSROOT_BASENAME" \
  -t "$IMAGE_TAG" \
  -f docker/x86-cross/Dockerfile \
  .

echo "Docker image built successfully: $IMAGE_TAG"
docker images | grep "$(echo "$IMAGE_TAG" | cut -d: -f1)" || true

echo "image_tag=$IMAGE_TAG" >> "$GITHUB_OUTPUT"

echo "::endgroup::"
