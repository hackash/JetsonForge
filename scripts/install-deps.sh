#!/usr/bin/env bash
set -euo pipefail

echo "::group::Installing system dependencies"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  bzip2 zstd xz-utils dpkg-dev qemu-user-static pv \
  curl jq ca-certificates
echo "::endgroup::"
