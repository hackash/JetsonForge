#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_FOLDER="${DOWNLOAD_FOLDER:?DOWNLOAD_FOLDER env var is required}"
WORK_FOLDER="${WORK_FOLDER:?WORK_FOLDER env var is required}"
SYSROOT_OUTPUT="${SYSROOT_OUTPUT:?SYSROOT_OUTPUT env var is required}"

# Expand tilde to home directory
DOWNLOAD_FOLDER="${DOWNLOAD_FOLDER/#\~/$HOME}"
WORK_FOLDER="${WORK_FOLDER/#\~/$HOME}"
SYSROOT_OUTPUT="${SYSROOT_OUTPUT/#\~/$HOME}"

echo "download_folder=$DOWNLOAD_FOLDER" >> "$GITHUB_OUTPUT"
echo "work_folder=$WORK_FOLDER"         >> "$GITHUB_OUTPUT"
echo "sysroot_output=$SYSROOT_OUTPUT"   >> "$GITHUB_OUTPUT"

echo "Expanded paths:"
echo "  Download folder: $DOWNLOAD_FOLDER"
echo "  Work folder:     $WORK_FOLDER"
echo "  Sysroot output:  $SYSROOT_OUTPUT"

mkdir -p "$DOWNLOAD_FOLDER" "$WORK_FOLDER" "$(dirname "$SYSROOT_OUTPUT")"
