# JetsonForge

## Introduction

Building software for NVIDIA Jetson devices presents several challenges for developers. Typically, you need access to a physical Jetson device, and must contend with different versions of JetPack, system libraries, and hardware dependencies. This complexity makes it difficult to set up a consistent and reproducible build environment, especially when targeting multiple Jetson models or JetPack releases.

JetsonForge addresses these problems by providing build scripts and guidance for creating base Docker images tailored for cross-compiling Jetson software on a regular x86_64 PC. With JetsonForge, developers can prepare cross-compilation environments without needing an actual Jetson device, streamlining the development and CI/CD process for Jetson-targeted applications.

## Prerequisites

To use JetsonForge, ensure you have the following:

- **Ubuntu 22.04 LTS** (server or desktop) on your x86_64 PC / server 
- **NVIDIA Developer Account** (required to access JetPack, L4T, and related downloads)
- **NVIDIA SDK Manager** (for downloading JetPack and L4T files, and preparing sysroot)
- **Docker** installed and configured
- **Sufficient disk space** (recommended: at least 20GB for images and sysroot)

Optional but recommended:
- Familiarity with cross-compilation and Docker
- Basic knowledge of CMake and build systems

> ⚠️ You must obtain NVIDIA JetPack and L4T files legally via your developer account. This project does not redistribute proprietary binaries.

## Getting Started

Follow these steps to build your base docker image for Jetson cross-compilation 

1. **Install Prerequisites**
   - Set up Ubuntu 22.04 on your x86_64 PC or server.
   - Install Docker.
   - Install NVIDIA SDK Manager.

2. **Download JetPack and L4T Files**

To see available download options run the following command: 

```bash 
sdkmanager --query non-interactive --action downloadonly
```

NOTE that archived versions will not show up, to see those add `--archived-versions` to the command as follows: 

```bash
 sdkmanager --query non-interactive --action downloadonly --archived-versions
```

Example downloading Jetpack 6.0 r36.3.0 for Jetson Orin Nano 8GB module: 

```bash
cd ~
mkdir jetpack-r36.3.0
sdkmanager --cli --action downloadonly --download-folder ~/jetpack-r36.3.0 --login-type devzone --exit-on-finish --license accept --product Jetson --archived-versions --version 6.0 --target-os Linux --host --target JETSON_ORIN_NANO_TARGETS
```

NOTE that above command will download all libraries that come by default with Jetpack SDK, if you want to be selective, run `sdkmanager` and follow instructions to download only whats needed

Once download is complete, Install required packages: 

```bash
sudo apt update
sudo apt install -y bzip2 zstd xz-utils dpkg-dev qemu-user-static pv
```

Clone JetsonForge repo

```bash
mkdir workspace
git clone https://github.com/hackash/JetsonForge.git
cd JetsonForge/scripts
chmod +x make-sys-root.sh
export JETSON_FORGE_DL=/home/ubuntu/jetpack-r36.3.0
export JETSON_FORGE_WORK=/home/ubuntu/workspace 
export JETSON_FORGE_OUT=/home/ubuntu/workspace/jetpack-6.0-aarch64.tar.zst
./make-sys-root.sh
```

3. **Place the Sysroot Tarball**
   - Copy your sysroot tarball to `sysroots/jetpack-6.0-aarch64.tar.zst` in this repository.

4. **Build the Cross-Compilation Docker Image**
   - Run:
     ```bash
     docker build --network host -t cross:latest -f docker/x86-cross/Dockerfile .
     ```

5. **Build Example Project or Your Own Code**
   - Use the cross-compilation image to build aarch64 binaries:
     ```bash
     docker run --rm -it -v $(pwd):/work cross:latest bash -lc 'cd examples && ./build.sh'
     ```

6. **Package and Deploy**
   - Use multi-stage Docker builds or copy the built binaries to your Jetson device for deployment.

Refer to the scripts and example projects in this repository for more details and customization options.



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
   docker build --network host -t cross:latest -f docker/x86-cross/Dockerfile .
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
