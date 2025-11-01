#!/usr/bin/env bash
set -euo pipefail

# ----------------- CONFIG -----------------
DL="${DL:-$HOME/Downloads/nvidia/sdkm_downloads}"   # where you downloaded files
WORK="${WORK:-$HOME/l4t}"                            # workspace (will create)
OUT="${OUT:-$HOME/jetpack-6.0-aarch64.tar.zst}"      # output sysroot tarball

JETSON_LINUX_TBZ="${JETSON_LINUX_TBZ:-$DL/Jetson_Linux_R36.3.0_aarch64.tbz2}"
SAMPLE_ROOTFS_TBZ="${SAMPLE_ROOTFS_TBZ:-$DL/Tegra_Linux_Sample-Root-Filesystem_R36.3.0_aarch64.tbz2}"

CUDA_LOCAL_REPO_DEB="${CUDA_LOCAL_REPO_DEB:-$DL/cuda-tegra-repo-ubuntu2204-12-2-local_12.2.12-1_arm64.deb}"
CUDNN_LOCAL_REPO_DEB="${CUDNN_LOCAL_REPO_DEB:-$DL/cudnn-local-tegra-repo-ubuntu2204-8.9.4.25_1.0-1_arm64.deb}"
TRT_LOCAL_REPO_DEB="${TRT_LOCAL_REPO_DEB:-$DL/nv-tensorrt-local-repo-l4t-8.6.2-cuda-12.2_1.0-1_arm64.deb}"

EXTRA_ARM64_DEBS=(
  "$DL/nvidia-l4t-gstreamer_36.3.0-20240506102626_arm64.deb"
  "$DL/nvidia-l4t-jetson-multimedia-api_36.3.0-20240506102626_arm64.deb"
  "$DL/OpenCV-4.8.0-1-g6371ee1-aarch64-libs.deb"
  "$DL/libnvidia-container1_1.14.2-1_arm64.deb"
  "$DL/libnvidia-container-tools_1.14.2-1_arm64.deb"
  "$DL/nvidia-container-toolkit-base_1.14.2-1_arm64.deb"
  "$DL/nvidia-container-toolkit_1.14.2-1_arm64.deb"
  "$DL/nvidia-l4t-dla-compiler_36.3.0-20240506102626_arm64.deb"
  "$DL/cupva-2.5.1-l4t.deb"
  "$DL/pva-allow-1.0.0.deb"
)

# ----------------- PREP -----------------
echo "==> Reset workspace (owned by user)"
sudo rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

# Extract Jetson_Linux as USER (keeps ownership = you)
echo "==> Extracting Jetson Linux package (user)"
tar -xjf "$JETSON_LINUX_TBZ"   # creates Linux_for_Tegra/

# Extract Sample RootFS into rootfs as ROOT (needs root perms there)
echo "==> Extracting Sample RootFS (sudo)"
sudo tar -xjf "$SAMPLE_ROOTFS_TBZ" -C Linux_for_Tegra/rootfs

cd Linux_for_Tegra

# Apply BSP binaries as ROOT
echo "==> Applying BSP binaries (sudo)"
sudo ./apply_binaries.sh

ROOT="$WORK/Linux_for_Tegra/rootfs"
STATUS="$ROOT/var/lib/dpkg/status"
sudo mkdir -p "$(dirname "$STATUS")"
sudo touch "$STATUS"

# ----------------- HELPERS -----------------
install_deb_into_rootfs() {
  local deb="$1"
  echo "  -> extracting $(basename "$deb")"
  sudo dpkg-deb -x "$deb" "$ROOT"
  # Append control to dpkg status for version checks
  local tmpd
  tmpd=$(mktemp -d)
  dpkg-deb -e "$deb" "$tmpd"
  sudo bash -c "cat '$tmpd/control' >> '$STATUS'; echo >> '$STATUS'"
  rm -rf "$tmpd"
}

extract_local_repo_and_install_pool() {
  local repo_deb="$1"
  local tmpd
  tmpd=$(mktemp -d)
  echo "==> Expanding local repo $(basename "$repo_deb")"
  dpkg-deb -x "$repo_deb" "$tmpd"
  mapfile -t inner < <(find "$tmpd"/var -type f -name "*.deb" | sort)
  for d in "${inner[@]}"; do
    if dpkg-deb -I "$d" control | grep -q '^Architecture: arm64'; then
      install_deb_into_rootfs "$d"
    fi
  done
  rm -rf "$tmpd"
}

# ----------------- INSTALL NVIDIA STACK -----------------
echo "==> Installing TensorRT 8.6.2.3, cuDNN 8.9.4.25, CUDA 12.2 from local repos (sudo)"
extract_local_repo_and_install_pool "$TRT_LOCAL_REPO_DEB"
extract_local_repo_and_install_pool "$CUDNN_LOCAL_REPO_DEB"
extract_local_repo_and_install_pool "$CUDA_LOCAL_REPO_DEB"

echo "==> Installing extra arm64 runtime libs (sudo)"
for deb in "${EXTRA_ARM64_DEBS[@]}"; do
  [[ -f "$deb" ]] && install_deb_into_rootfs "$deb" || true
done

# Ensure /usr/local/cuda symlink exists inside sysroot (helps CMake find CUDA)
echo "==> Ensuring /usr/local/cuda symlink → cuda-12.2 (sudo)"
if sudo test -d "$ROOT/usr/local/cuda-12.2" && ! sudo test -e "$ROOT/usr/local/cuda"; then
  sudo ln -s cuda-12.2 "$ROOT/usr/local/cuda"
fi

# Optional sanity check (no fail on missing file)
echo "==> CUDA version.json (if present):"
sudo bash -lc 'test -f "'"$ROOT"'/usr/local/cuda/version.json" && cat "'"$ROOT"'/usr/local/cuda/version.json" || echo "CUDA version.json not found"'

# ----------------- PACKAGE SYSROOT -----------------
echo "==> Packaging sysroot (sudo, numeric owners) → $OUT"
#sudo tar -C "$ROOT" --numeric-owner -I 'zstd -19' -cpf "$OUT" .
# Zip with progress bar
sudo bash -c "
  cd '$ROOT' &&
  tar -cf - --numeric-owner . | pv -s \$(du -sb . | awk '{print \$1}') | zstd -19 -T0 -o '$OUT'
"

echo "==> Done. Sysroot: $OUT"
