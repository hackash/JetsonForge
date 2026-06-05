#!/usr/bin/env bash
set -euo pipefail

SYSROOT_BASENAME="${SYSROOT_BASENAME:-}"
ACTION_PATH="${ACTION_PATH:?ACTION_PATH env var is required}"
WORK_FOLDER="${WORK_FOLDER:-}"

echo "::group::Cleanup"

# Remove the temporary sysroot copy placed inside the repo for Docker build.
# The cached archive at the user-specified output path is kept untouched.
if [[ -n "$SYSROOT_BASENAME" ]]; then
  rm -f "${ACTION_PATH}/sysroots/${SYSROOT_BASENAME}" || true
fi

if [[ -n "$WORK_FOLDER" ]] && [[ -d "$WORK_FOLDER" ]]; then
  echo "Cleaning up work folder: $WORK_FOLDER"
  rm -rf "$WORK_FOLDER" || true
fi

echo "Cleanup complete"
echo "::endgroup::"
