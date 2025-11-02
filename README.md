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

Once download is complete, install required packages: 

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

This command will create jetpack-6.0-aarch64.tar.zst under `JETSON_FORGE_OUT` directory, copy that to JetsonForge repo: 

```bash 
cp /home/ubuntu/workspace/jetpack-6.0-aarch64.tar.zst JetsonForge/sysroots
```

Navigate to root of JetsonForge repository `cd JetsonForge` and build base docker image:

```bash
docker build --network host --build-arg TAR_ZST_NAME=jetpack-6.0-aarch64.tar.zst -t l4t-cross-base:r36.3.0 -f docker/x86-cross/Dockerfile .
```

NOTE: replace image name with your own: e.g [your org]/l4t-cross:36.3.0

At this point `l4t-cross-base:r36.3.0` is fully build compatible jetson base container, similar to what you would have on real device with a fresh installed Jetpack 6.0, we can use this to build our software which will later run on real device, to test, build example app:

```bash
docker build --network host --build-arg L4T_CROSS_BASE=l4t-cross-base:r36.3.0  --build-arg TARGET_JETPACK_TAG=r36.3.0 -t l4t-cross-example-app:r36.3.0-latest -f examples/cmake-cuda/Dockerfile examples/cmake-cuda
```

At this point `l4t-cross-example-app:36.3.0-latest` is ready software image that can run on real jetson device like Orin Nano 8GB with Jetpack 6.0, to test, push your image to dockerhub and then pull to your device, once ready you can run with the following command: 

```bash 
docker run -it --rm --runtime nvidia l4t-cross-example-app:r36.3.0-latest
```

You should see similar output: 

```bash 
Detected 1 CUDA device(s)

Device 0: Orin
  Compute capability: 8.7
  Total global memory: 7621 MB
  Multiprocessors: 8
  Max threads per block: 1024
  Clock rate: 624 MHz
```

Congratulations! You successfully built example app on x86_64 machine without Nvidia GPU and ran on real Jetson device with real GPU!

## What’s inside

- `docker/x86-cross/Dockerfile` – x86_64 builder with aarch64 toolchain and your JP6 sysroot
- `toolchains/aarch64-jetson.cmake` – CMake toolchain file for cross-compiles
- `examples/cmake-cuda/` – Minimal CUDA example + Dockerfile
- `scripts/verify-sysroot.sh` – Sanity checks for CUDA/cuDNN/TensorRT versions in the sysroot
- `.github/workflows/build-cross.yml` – Example CI pipeline (customize/publish as you wish)

## Notes / Gotchas

- Ensure the sysroot **exactly matches** your deployment Jetson’s JetPack/L4T version.
- If your build steps **execute** target binaries, prefer building those helper tools for the **host (x86)**, or use a small emulated step with QEMU.
- Keep your sysroot **versioned** and reproducible to avoid heisenbugs.
