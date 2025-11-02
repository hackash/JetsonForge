# 🚀 JetsonForge

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/hackash/JetsonForge/build-cross.yml?label=CI%20Build)](https://github.com/hackash/JetsonForge/actions)
[![Platform](https://img.shields.io/badge/platform-x86__64%20%7C%20Jetson-blue)](#)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen)](#)

---

## 📘 Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
   - [1. Install Prerequisites](#1-install-prerequisites)
   - [2. Download JetPack and L4T Files](#2-download-jetpack-and-l4t-files)
   - [3. Install Required Packages](#3-install-required-packages)
   - [4. Clone JetsonForge Repository](#4-clone-jetsonforge-repository)
   - [5. Build the Base Docker Image](#5-build-the-base-docker-image)
   - [6. Build and Test an Example Application](#6-build-and-test-an-example-application)
4. [What’s Inside](#whats-inside)
5. [Notes and Gotchas](#notes-and-gotchas)
6. [License](#license)

---

## 🧠 Introduction

Building software for **NVIDIA Jetson** devices often presents challenges.  
Typically, developers must have physical access to a Jetson board and manage varying **JetPack**, **L4T**, and **system library** versions.  
This can make it hard to maintain a consistent, reproducible build environment—especially when supporting multiple Jetson models.

**JetsonForge** simplifies this by providing scripts and tooling to create **cross-compilation Docker images** on a standard x86_64 machine.  
You can now build Jetson software **without owning a Jetson device**, enabling faster development and CI/CD workflows.

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

- **Ubuntu 22.04 LTS** (desktop or server)  
- **NVIDIA Developer Account** — required to access JetPack/L4T downloads  
- **NVIDIA SDK Manager** — for downloading JetPack and preparing the sysroot  
- **Docker** — installed and configured  
- **20 GB+ of free disk space** — for images, SDKs, and sysroot data  

**Optional (but recommended):**
- Basic knowledge of Docker, cross-compilation, and CMake

> ⚠️ **Legal Notice:**  
> You must download JetPack and L4T files **only** via your official NVIDIA Developer Account.  
> JetsonForge does **not** distribute proprietary binaries.

---

## 🚀 Getting Started

Follow these steps to set up your Jetson cross-compile environment.

---

### 1. Install Prerequisites

Set up Ubuntu 22.04 and install:

```bash
sudo apt update
sudo apt install -y docker.io nvidia-sdk-manager
```

---

### 2. Download JetPack and L4T Files

List available JetPack versions:

```bash
sdkmanager --query non-interactive --action downloadonly
```

To include archived versions:

```bash
sdkmanager --query non-interactive --action downloadonly --archived-versions
```

**Example:** Download JetPack 6.0 (r36.3.0) for **Jetson Orin Nano 8 GB**:

```bash
cd ~
mkdir jetpack-r36.3.0
sdkmanager --cli --action downloadonly   --download-folder ~/jetpack-r36.3.0   --login-type devzone   --exit-on-finish   --license accept   --product Jetson   --archived-versions   --version 6.0   --target-os Linux   --host   --target JETSON_ORIN_NANO_TARGETS
```

> 💡 This downloads all default JetPack 6.0 components.  
> To select only specific packages, launch `sdkmanager` in interactive mode.

---

### 3. Install Required Packages

After downloading, install dependencies:

```bash
sudo apt update
sudo apt install -y bzip2 zstd xz-utils dpkg-dev qemu-user-static pv
```

---

### 4. Clone JetsonForge Repository

```bash
mkdir ~/workspace
git clone https://github.com/hackash/JetsonForge.git
cd JetsonForge/scripts
chmod +x make-sys-root.sh
```

Set environment variables and generate the sysroot:

```bash
export JETSON_FORGE_DL=/home/ubuntu/jetpack-r36.3.0
export JETSON_FORGE_WORK=/home/ubuntu/workspace 
export JETSON_FORGE_OUT=/home/ubuntu/workspace/jetpack-6.0-aarch64.tar.zst
./make-sys-root.sh
```

Once complete, copy the generated sysroot archive into JetsonForge:

```bash
cp /home/ubuntu/workspace/jetpack-6.0-aarch64.tar.zst JetsonForge/sysroots
```

---

### 5. Build the Base Docker Image

From the root of the JetsonForge repo:

```bash
cd JetsonForge
docker build --network host   --build-arg TAR_ZST_NAME=jetpack-6.0-aarch64.tar.zst   -t l4t-cross-base:r36.3.0   -f docker/x86-cross/Dockerfile .
```

> 🏷️ Rename as desired, e.g. `yourorg/l4t-cross:r36.3.0`.

The resulting image `l4t-cross-base:r36.3.0` replicates a clean JetPack 6.0 environment and can be used to build software for Jetson targets.

---

### 6. Build and Test an Example Application

Build the example CUDA project:

```bash
docker build --network host   --build-arg L4T_CROSS_BASE=l4t-cross-base:r36.3.0   --build-arg TARGET_JETPACK_TAG=r36.3.0   -t l4t-cross-example-app:r36.3.0-latest   -f examples/cmake-cuda/Dockerfile examples/cmake-cuda
```

You can now run this image on your Jetson device:

```bash
docker run -it --rm --runtime nvidia l4t-cross-example-app:r36.3.0-latest
```

Expected output:

```
Detected 1 CUDA device(s)

Device 0: Orin
  Compute capability: 8.7
  Total global memory: 7621 MB
  Multiprocessors: 8
  Max threads per block: 1024
  Clock rate: 624 MHz
```

🎉 **Success!**  
You’ve cross-compiled a CUDA app on an x86_64 host (no GPU required) and executed it natively on a Jetson board.

---

## 📦 What’s Inside

| Path | Description |
|------|--------------|
| `docker/x86-cross/Dockerfile` | x86_64 build image with AArch64 toolchain & JetPack sysroot |
| `toolchains/aarch64-jetson.cmake` | CMake toolchain configuration for cross-compiling |
| `examples/cmake-cuda/` | Minimal CUDA example and Dockerfile |
| `scripts/verify-sysroot.sh` | Verifies CUDA/cuDNN/TensorRT compatibility in the sysroot |
| `.github/workflows/build-cross.yml` | Example GitHub Actions CI pipeline |

---

## ⚠️ Notes and Gotchas

- Ensure your **sysroot** matches the **exact JetPack/L4T version** of your deployment device.  
- If a build step executes target binaries, compile those utilities for **x86** or use **QEMU** emulation.  
- Keep sysroots **version-controlled and reproducible** to prevent inconsistencies across environments.

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

### ❤️ Contributing

Pull requests and improvements are always welcome!  
If you find bugs or want to suggest enhancements, open an issue on [GitHub](https://github.com/hackash/JetsonForge).

---

### 📫 Support

For setup questions or contribution discussions, reach out via GitHub Issues or start a Discussion thread.

---

> © 2025 JetsonForge — Empowering Jetson developers everywhere.
