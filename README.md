
# Jetson Virtual Build Environment (JP 6.0 • CUDA 12.2 • cuDNN 8.9.4 • TensorRT 8.6.2)

This repo scaffolds a **fast x86 builder image** that cross-compiles for **aarch64 Jetson (Orin)** without needing physical hardware.  
It targets **JetPack 6.0** with the following library expectations in the **sysroot**:

- `cuda-12.2`
- `libcudnn8=8.9.4.25-1+cuda12.2`
- `libcudnn8-dev=8.9.4.25-1+cuda12.2`
- `libnvinfer8=8.6.2.3-1+cuda12.2`
- `libnvinfer-dev=8.6.2.3-1+cuda12.2`

> ⚠️ **NVIDIA licensing:** This project does **not** redistribute NVIDIA binaries.  
> You must provide a **sysroot tarball** built from your own JetPack 6.0/L4T environment.  
> Place it at: `sysroots/jetpack-6.0-aarch64.tar.zst` before building the image.

---

## Quick Start

1) **Create a sysroot tarball** for JetPack 6.0 (L4T R36.x) that contains:
   - `/usr/local/cuda` (CUDA 12.2)
   - cuDNN 8.9.4.25 packages (runtime + dev)
   - TensorRT 8.6.2.3 packages (runtime + dev)
   - `/lib` and `/usr/lib/aarch64-linux-gnu` with matching glibc/Ubuntu 22.04
   - `/usr/include` (headers), `/var/lib/dpkg/status` (for version checks)

   See `scripts/prepare-sysroot-jp6.sh` for guidance.

2) **Build the x86 cross builder image**:
   ```bash
   docker build -f docker/x86-cross/Dockerfile      -t ghcr.io/YOUR_ORG/jetson-cross:jp6.0-cuda12.2      .
   ```

3) **Build the example project (aarch64 artifacts)**:
   ```bash
   docker run --rm -it -v $(pwd):/work ghcr.io/YOUR_ORG/jetson-cross:jp6.0-cuda12.2      bash -lc 'cd examples/cmake-cuda && ./build.sh'
   ```

4) **Package your app** using a multi-stage Dockerfile with `docker buildx` to produce an `arm64` image for Jetson.

---

## What’s inside

- `docker/x86-cross/Dockerfile` – x86_64 builder with aarch64 toolchain and your JP6 sysroot
- `toolchains/aarch64-jetson.cmake` – CMake toolchain file for cross-compiles
- `examples/cmake-cuda/` – Minimal CUDA example + build script
- `scripts/verify-sysroot.sh` – Sanity checks for CUDA/cuDNN/TensorRT versions in the sysroot
- `.github/workflows/build-cross.yml` – Example CI pipeline (customize/publish as you wish)

---

## Example multi-stage packaging (arm64 runtime)

After building aarch64 artifacts with the cross image, you can produce a final Jetson runtime image like this:

```Dockerfile
# ---- Stage 1: build (x86 cross)
FROM ghcr.io/YOUR_ORG/jetson-cross:jp6.0-cuda12.2 AS builder
WORKDIR /src
COPY . .
RUN cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake \
 && cmake --build build -j$(nproc)

# ---- Stage 2: runtime (arm64 Jetson/L4T)
FROM nvcr.io/nvidia/l4t-base:36.3.0
WORKDIR /app
COPY --from=builder /src/build/mybin /app/mybin
ENTRYPOINT ["/app/mybin"]
```

Build on x86 with Buildx:
```bash
docker buildx build --platform=linux/arm64 -t your/app:jetson .
```

---

## Notes / Gotchas

- Ensure the sysroot **exactly matches** your deployment Jetson’s JetPack/L4T version.
- If your build steps **execute** target binaries, prefer building those helper tools for the **host (x86)**, or use a small emulated step with QEMU.
- Keep your sysroot **versioned** and reproducible to avoid heisenbugs.
