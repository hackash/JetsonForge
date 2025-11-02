FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TAR_ZST_NAME

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git ccache \
    pkg-config \
    python3 python3-pip rsync file \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    zstd ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Sysroot (exact copy of Jetson userland) ----
COPY sysroots/${TAR_ZST_NAME} /tmp/sysroot.tar.zst
RUN mkdir -p /opt/jetson-sysroot \
 && unzstd -c /tmp/sysroot.tar.zst | tar -x -C /opt/jetson-sysroot --strip-components=1 \
 && rm -f /tmp/sysroot.tar.zst

ENV SYSROOT=/opt/jetson-sysroot

# ---- Cross toolchain env ----
ENV CC=aarch64-linux-gnu-gcc \
    CXX=aarch64-linux-gnu-g++ \
    AR=aarch64-linux-gnu-ar \
    AS=aarch64-linux-gnu-as \
    LD=aarch64-linux-gnu-ld \
    STRIP=aarch64-linux-gnu-strip

# ---- pkg-config & CMake discovery rooted in the sysroot ----
# define PKG_CONFIG_LIBDIR first
ENV PKG_CONFIG_SYSROOT_DIR="${SYSROOT}"
ENV PKG_CONFIG_LIBDIR="${SYSROOT}/usr/local/cuda/lib64/pkgconfig:${SYSROOT}/usr/lib/aarch64-linux-gnu/tegra/pkgconfig:${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}/usr/share/pkgconfig"
# now safely reference it
ENV PKG_CONFIG_PATH="${PKG_CONFIG_LIBDIR}"
# make CMake prefer packages inside sysroot
ENV CMAKE_PREFIX_PATH="${SYSROOT}/usr:${SYSROOT}/usr/local"

# CUDA hints (some find scripts look at these)
ENV CUDA_HOME="${SYSROOT}/usr/local/cuda" \
    CUDA_TOOLKIT_ROOT_DIR="${SYSROOT}/usr/local/cuda"

# ---- CMake toolchain ----
COPY toolchains /toolchains
ENV CMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake

WORKDIR /work
CMD ["/bin/bash"]
