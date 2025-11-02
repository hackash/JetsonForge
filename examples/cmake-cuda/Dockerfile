# syntax=docker/dockerfile:1.7
# Ensure these are defined *before* any FROM, with sane defaults
ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM=linux/arm64
ARG L4T_CROSS_BASE=l4t-cross-base:r36.3.0
ARG TARGET_JETPACK_TAG=r36.3.0

# -------- Stage 1: build on your x86 cross image --------
FROM --platform=${BUILDPLATFORM} ${L4T_CROSS_BASE} AS build
WORKDIR /app

# L4T_CROSS_BASE already has toolchain + env; install extras only if needed
# RUN apt-get update && apt-get install -y <extras> && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt main.cpp ./

RUN cmake -B build -S . -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)"

# -------- Stage 2: runtime on Jetson (arm64) --------
FROM nvcr.io/nvidia/l4t-jetpack:${TARGET_JETPACK_TAG}
COPY --from=build /app/build/device_query /usr/local/bin/device_query
CMD ["/usr/local/bin/device_query"]
