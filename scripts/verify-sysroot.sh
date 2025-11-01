
#!/usr/bin/env bash
set -euo pipefail

SYSROOT="${SYSROOT:-/opt/jetson-sysroot}"

status="${SYSROOT}/var/lib/dpkg/status"
if [[ ! -f "$status" ]]; then
  echo "WARN: dpkg status file not found at $status. Skipping package checks."
  exit 0
fi

want_pkg() { grep -E "Package: $1$" -A 10 "$status" | awk -F': ' '/^Version:/{print $2; exit}'; }

check_pkg_version() {
  local pkg="$1" want="$2"
  local got
  got="$(want_pkg "$pkg" || true)"
  if [[ -z "$got" ]]; then
    echo "MISSING: $pkg (wanted $want)"
    return 1
  fi
  if [[ "$got" == "$want" ]]; then
    echo "OK: $pkg == $want"
  else
    echo "MISMATCH: $pkg got=$got want=$want"
    return 1
  fi
}

fail=0
# Required versions per README
check_pkg_version "libcudnn8" "8.9.4.25-1+cuda12.2" || fail=1
check_pkg_version "libcudnn8-dev" "8.9.4.25-1+cuda12.2" || fail=1
check_pkg_version "libnvinfer8" "8.6.2.3-1+cuda12.2" || fail=1
check_pkg_version "libnvinfer-dev" "8.6.2.3-1+cuda12.2" || fail=1

cuda_json="${SYSROOT}/usr/local/cuda/version.json"
if [[ -f "$cuda_json" ]]; then
  echo "CUDA version.json:"
  cat "$cuda_json"
else
  echo "WARN: CUDA version.json not found at $cuda_json"
fi

exit $fail
