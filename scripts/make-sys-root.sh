#!/usr/bin/env bash
set -euo pipefail

# ----------------- CONFIG -----------------
JETSON_FORGE_DL="${JETSON_FORGE_DL:-$HOME/Downloads/nvidia/sdkm_downloads}"
JETSON_FORGE_WORK="${JETSON_FORGE_WORK:-$HOME/l4t}"
JETSON_FORGE_OUT="${JETSON_FORGE_OUT:-$HOME/jetpack-aarch64.tar.zst}"

# Helper: safely get the newest file matching a pattern, or empty string if none
get_latest() {
  local pattern="$1"
  local file
  file=$(ls -1v "$JETSON_FORGE_DL"/$pattern 2>/dev/null | tail -n1 || true)
  echo "$file"
}

# ----------------- CORE FILES -----------------
JETSON_LINUX_TBZ="$(get_latest 'Jetson_Linux_R*_aarch64.tbz2')"
SAMPLE_ROOTFS_TBZ="$(get_latest 'Tegra_Linux_Sample-Root-Filesystem_R*_aarch64.tbz2')"
CUDA_LOCAL_REPO_DEB="$(get_latest 'cuda-tegra-repo-ubuntu*_arm64.deb')"
CUDNN_LOCAL_REPO_DEB="$(get_latest 'cudnn-local-tegra-repo-ubuntu*_arm64.deb')"
TRT_LOCAL_REPO_DEB="$(get_latest 'nv-tensorrt-local-repo-l4t-*_arm64.deb')"

for var in JETSON_LINUX_TBZ SAMPLE_ROOTFS_TBZ CUDA_LOCAL_REPO_DEB CUDNN_LOCAL_REPO_DEB TRT_LOCAL_REPO_DEB; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: Required file for $var not found in $JETSON_FORGE_DL"
    exit 1
  fi
done

# ----------------- EXTRA PACKAGES -----------------
EXTRA_ARM64_DEBS=()
while IFS= read -r -d '' f; do
  case "$f" in
    "$JETSON_LINUX_TBZ" | "$SAMPLE_ROOTFS_TBZ" | \
    "$CUDA_LOCAL_REPO_DEB" | "$CUDNN_LOCAL_REPO_DEB" | "$TRT_LOCAL_REPO_DEB")
      ;;
    *)
      EXTRA_ARM64_DEBS+=("$f")
      ;;
  esac
done < <(find "$JETSON_FORGE_DL" -maxdepth 1 -type f -name '*.deb' -print0 | sort -zV)

# ----------------- SUMMARY -----------------
echo "Resolved artifacts:"
echo "  JETSON_LINUX_TBZ     = ${JETSON_LINUX_TBZ:-<missing>}"
echo "  SAMPLE_ROOTFS_TBZ    = ${SAMPLE_ROOTFS_TBZ:-<missing>}"
echo "  CUDA_LOCAL_REPO_DEB  = ${CUDA_LOCAL_REPO_DEB:-<missing>}"
echo "  CUDNN_LOCAL_REPO_DEB = ${CUDNN_LOCAL_REPO_DEB:-<missing>}"
echo "  TRT_LOCAL_REPO_DEB   = ${TRT_LOCAL_REPO_DEB:-<missing>}"
printf '  EXTRA_ARM64_DEBS (%d items):\n' "${#EXTRA_ARM64_DEBS[@]}"
printf '    - %s\n' "${EXTRA_ARM64_DEBS[@]}"


# Extract Jetson_Linux as USER (keeps ownership = you)
echo "==> Extracting Jetson Linux package (user)"
tar -xjf "$JETSON_LINUX_TBZ" -C "$JETSON_FORGE_WORK"   # creates Linux_for_Tegra/

# Extract Sample RootFS into rootfs as ROOT (needs root perms there)
echo "==> Extracting Sample RootFS (sudo)"
sudo tar -xjf "$SAMPLE_ROOTFS_TBZ" -C "$JETSON_FORGE_WORK/Linux_for_Tegra/rootfs"

cd "$JETSON_FORGE_WORK/Linux_for_Tegra"

# Apply BSP binaries as ROOT
echo "==> Applying BSP binaries (sudo)"
sudo ./apply_binaries.sh

ROOT="$JETSON_FORGE_WORK/Linux_for_Tegra/rootfs"
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
extract_local_repo_and_install_pool "$TRT_LOCAL_REPO_DEB"
extract_local_repo_and_install_pool "$CUDNN_LOCAL_REPO_DEB"
extract_local_repo_and_install_pool "$CUDA_LOCAL_REPO_DEB"

echo "==> Installing extra arm64 runtime libs (sudo)"
for deb in "${EXTRA_ARM64_DEBS[@]}"; do
  [[ -f "$deb" ]] && install_deb_into_rootfs "$deb" || true
done

echo "==> Ensuring /usr/local/cuda symlink"
cuda_version=$(ls "$JETSON_FORGE_DL"/cuda-repo-ubuntu*-local_*_*.deb 2>/dev/null | sort -V | tail -1 | sed -E 's/.*ubuntu[0-9]+-([0-9]+)-([0-9]+)-local_.*/\1.\2/')
[ -n "$cuda_version" ] && sudo ln -sfn "cuda-$cuda_version" "$ROOT/usr/local/cuda"

# Ensure /usr/local/cuda symlink exists inside sysroot (helps CMake find CUDA)
if sudo test -d "$ROOT/usr/local/cuda-$cuda_version" && ! sudo test -e "$ROOT/usr/local/cuda"; then
  sudo ln -s "cuda-$cuda_version" "$ROOT/usr/local/cuda"
fi

echo "==> CUDA version.json (if present):"
sudo bash -lc 'test -f "'"$ROOT"'/usr/local/cuda/version.json" && cat "'"$ROOT"'/usr/local/cuda/version.json" || echo "CUDA version.json not found"'

# ----------------- PACKAGE SYSROOT -----------------
echo "==> Packaging sysroot (sudo, numeric owners) → $JETSON_FORGE_OUT"
#sudo tar -C "$ROOT" --numeric-owner -I 'zstd -19' -cpf "$JETSON_FORGE_OUT" .
# Zip with progress bar
sudo bash -c "
  cd '$ROOT' &&
  tar -cf - --numeric-owner . | pv -s \$(du -sb . | awk '{print \$1}') | zstd -19 -T0 -o '$JETSON_FORGE_OUT'
"

echo "==> Done. Sysroot: $JETSON_FORGE_OUT"
