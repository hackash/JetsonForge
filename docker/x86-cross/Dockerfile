
# x86_64 cross-compile builder for JetPack 6.0 (CUDA 12.2, cuDNN 8.9.4, TensorRT 8.6.2)
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential cmake ninja-build git ccache pkg-config     python3 python3-pip rsync file     gcc-aarch64-linux-gnu g++-aarch64-linux-gnu     zstd ca-certificates curl  && rm -rf /var/lib/apt/lists/*

# Expect a sysroot tarball in build context at sysroots/jetpack-6.0-aarch64.tar.zst
# This contains CUDA 12.2, cuDNN 8.9.4.25, TensorRT 8.6.2.3, headers, libs, etc.
COPY sysroots/jetpack-6.0-aarch64.tar.zst /tmp/sysroot.tar.zst

RUN mkdir -p /opt/jetson-sysroot  && unzstd -c /tmp/sysroot.tar.zst | tar -x -C /opt/jetson-sysroot --strip-components=1  && rm -f /tmp/sysroot.tar.zst

# Environment for cross compile
ENV SYSROOT=/opt/jetson-sysroot
ENV CC=aarch64-linux-gnu-gcc     CXX=aarch64-linux-gnu-g++
ENV PKG_CONFIG_LIBDIR=$SYSROOT/usr/lib/aarch64-linux-gnu/pkgconfig:$SYSROOT/usr/lib/pkgconfig:$SYSROOT/usr/share/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=$SYSROOT

# Common search paths so CMake/find_package picks up from sysroot
ENV CMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake

# Copy toolchain file and helper scripts
COPY toolchains /toolchains
COPY scripts/verify-sysroot.sh /usr/local/bin/verify-sysroot
RUN chmod +x /usr/local/bin/verify-sysroot

# Optional: Run a quick sysroot check during build (can be disabled with --build-arg)
ARG RUN_VERIFY=1
RUN if [ "$RUN_VERIFY" = "1" ]; then /usr/local/bin/verify-sysroot; fi

WORKDIR /work
CMD ["/bin/bash"]
