#!/usr/bin/env bash
set -euo pipefail

SYSROOT_SRC="${SYSROOT_SRC:?SYSROOT_SRC env var is required}"
ACTION_PATH="${ACTION_PATH:?ACTION_PATH env var is required}"

echo "::group::Preparing sysroot for Docker build"

SYSROOT_BASENAME="$(basename "$SYSROOT_SRC")"
SYSROOT_DEST="${ACTION_PATH}/sysroots/${SYSROOT_BASENAME}"

mkdir -p "${ACTION_PATH}/sysroots"

if [[ ! -f "$SYSROOT_SRC" ]]; then
  echo "::error::Sysroot file not found: $SYSROOT_SRC"
  exit 1
fi

echo "Copying sysroot to repository..."
echo "  Source:      $SYSROOT_SRC"
echo "  Destination: $SYSROOT_DEST"

cp "$SYSROOT_SRC" "$SYSROOT_DEST"

echo "Sysroot copied successfully"
ls -lh "$SYSROOT_DEST"

echo "sysroot_basename=$SYSROOT_BASENAME" >> "$GITHUB_OUTPUT"

echo "::endgroup::"
