#!/usr/bin/env bash
set -euo pipefail

USER_TAG="${USER_TAG:-}"
JP_VERSION="${JP_VERSION:?JP_VERSION env var is required}"
TARGET="${TARGET:?TARGET env var is required}"

echo "::group::Computing Docker image tag"

if [[ -n "$USER_TAG" ]]; then
  IMAGE_TAG="$USER_TAG"
  echo "Using user-provided image tag: $IMAGE_TAG"
else
  # JETSON_ORIN_NANO_TARGETS -> orin-nano
  # JETSON_AGX_ORIN_TARGETS  -> agx-orin
  # JETSON_XAVIER_NX_TARGETS -> xavier-nx
  SIMPLE_TARGET=$(echo "$TARGET" \
    | sed 's/JETSON_//g; s/_TARGETS//g' \
    | tr '[:upper:]' '[:lower:]' \
    | tr '_' '-')

  IMAGE_TAG="jetson-cross-base:jp${JP_VERSION}-${SIMPLE_TARGET}"
  echo "Auto-generated image tag: $IMAGE_TAG"
fi

echo "image_tag=$IMAGE_TAG" >> "$GITHUB_OUTPUT"

echo ""
echo "✅ Docker image will be tagged as: $IMAGE_TAG"
echo "::endgroup::"
