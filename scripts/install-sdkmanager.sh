#!/usr/bin/env bash
set -euo pipefail

echo "::group::Installing NVIDIA SDK Manager"

if command -v sdkmanager &> /dev/null; then
  echo "SDK Manager is already installed"
  sdkmanager --version || true
else
  echo "Installing SDK Manager from NVIDIA repository..."

  . /etc/os-release
  case "$VERSION_ID" in
    24.04) DISTRO="ubuntu2404" ;;
    22.04) DISTRO="ubuntu2204" ;;
    *)
      echo "::warning::Unsupported Ubuntu version: $VERSION_ID, defaulting to ubuntu2204"
      DISTRO="ubuntu2204"
      ;;
  esac

  echo "Detected distro: $DISTRO (Ubuntu $VERSION_ID)"

  wget "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb

  sudo apt-get update
  sudo apt-get install -y sdkmanager

  echo "SDK Manager installed successfully"
  sdkmanager --version || true
fi

echo "::endgroup::"
