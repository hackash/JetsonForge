# 🚀 JetsonForge GitHub Action

Build **NVIDIA Jetson cross-compilation Docker images** directly in your GitHub Actions workflows with intelligent caching and headless authentication support.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [JetPack Versions & Device Compatibility](#jetpack-versions--device-compatibility)
- [Runner Recommendations](#runner-recommendations)
- [Webhook Configuration](#webhook-configuration)
- [Advanced Usage](#advanced-usage)
- [Complete Workflow Examples](#complete-workflow-examples)
- [Docker Registry Deployment](#docker-registry-deployment)
- [Inputs & Outputs](#inputs--outputs)
- [Caching Strategy](#caching-strategy)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## 🧠 Overview

The **JetsonForge GitHub Action** automates the creation of cross-compilation environments for NVIDIA Jetson devices. It handles:

- ✅ **Headless NVIDIA SDK Manager authentication** with webhook notifications
- ✅ **Intelligent caching** of SDK sessions, downloads, and sysroots
- ✅ **Automated sysroot generation** from JetPack downloads
- ✅ **Docker image building** ready for cross-compilation
- ✅ **Support for both GitHub-hosted and self-hosted runners**

Once the action completes, you'll have a Docker image (`jetson-cross-base:local` by default) that contains the full cross-compilation toolchain and Jetson sysroot, ready to build your application.

---

## ✨ Features

### 🔐 Headless Authentication
- Extracts NVIDIA SDK Manager login URLs automatically
- Sends interactive authentication links via **Slack** and/or **Microsoft Teams** webhooks
- Waits for authentication with configurable timeout
- Caches authenticated sessions for subsequent runs

### 💾 Intelligent Caching
- **SDK Manager session**: Persists authentication state across runs
- **JetPack downloads**: ~10-20 GB download cached (no re-download needed)
- **Built sysroot**: ~5-8 GB compressed archive cached
- **Cache invalidation**: Automatic based on JetPack version and target device

### 🏗️ Automated Build Pipeline
- Downloads JetPack components via SDK Manager
- Extracts and configures Jetson userland (sysroot)
- Builds Docker image with cross-compilation toolchain
- Verifies the image and toolchain setup

---

## ⚙️ Prerequisites

### Required
- **NVIDIA Developer Account** — [Sign up here](https://developer.nvidia.com/login)
- **GitHub repository** with JetsonForge action configured
- **Know your target device** — Check [JetPack compatibility guide](JETPACK-VERSIONS.md)

### Optional (Recommended)
- **Slack Workspace** with incoming webhook capability
- **Microsoft Teams** with incoming webhook configured
- **Self-hosted GitHub runner** with Docker and sufficient disk space (~60 GB recommended)

> 💡 **Important**: Make sure to verify your JetPack version and target device are compatible using the [JETPACK-VERSIONS.md](JETPACK-VERSIONS.md) reference guide before configuring the action.

---

## 🚀 Quick Start

### Basic Workflow

Create `.github/workflows/build-jetson.yml`:

```yaml
name: Build Jetson Application

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
      
      - name: Setup JetsonForge Cross-Compiler
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
          slack_webhook: ${{ secrets.SLACK_WEBHOOK }}
          # build_image_tag is auto-generated as: jetson-cross-base:jp6.0-orin-nano
      
      - name: Build Application
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/workspace \
            -w /workspace \
            jetson-cross-base:jp6.0-orin-nano \
            bash -c "cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake && cmake --build build"
```

### First Run
1. Push your workflow file
2. GitHub Actions will start and pause for authentication
3. Check your Slack/Teams channel for the authentication link
4. Click the link and log in with your NVIDIA Developer credentials
5. The workflow will automatically continue once authenticated

### Subsequent Runs
- Authentication is cached — no login required until the token expires (typically weeks/months)
- Downloads and sysroot are cached — builds complete in minutes instead of hours

---

## 🎯 JetPack Versions & Device Compatibility

### Quick Reference

Choosing the correct JetPack version and target device is critical for successful builds.

| Your Jetson Device | JetPack Version | Target Value | L4T Release |
|-------------------|-----------------|--------------|-------------|
| Orin Nano (8GB/4GB) | `6.0` or `6.1` | `JETSON_ORIN_NANO_TARGETS` | r36.3.0 |
| AGX Orin (64GB/32GB) | `6.0` or `6.1` | `JETSON_AGX_ORIN_TARGETS` | r36.3.0 |
| Orin NX (16GB/8GB) | `6.0` or `6.1` | `JETSON_ORIN_NX_TARGETS` | r36.3.0 |
| AGX Xavier | `5.1.2` or `4.6.4` | `JETSON_AGX_XAVIER_TARGETS` | r35.4.1 |
| Xavier NX | `5.1.2` or `4.6.4` | `JETSON_XAVIER_NX_TARGETS` | r35.4.1 |
| Nano (4GB/2GB) | `4.6.4` | `JETSON_NANO_TARGETS` | r32.7.4 |
| TX2 | `4.6.4` | `JETSON_TX2_TARGETS` | r32.7.4 |

### How to Find the Right Values

**Method 1: Check Your Physical Device**
```bash
# SSH into your Jetson and run:
cat /etc/nv_tegra_release

# Output example:
# R36 (release), REVISION: 3.0 → Use JetPack 6.0
# R35 (release), REVISION: 4.1 → Use JetPack 5.1.2
# R32 (release), REVISION: 7.4 → Use JetPack 4.6.4
```

**Method 2: Query SDK Manager Locally**
```bash
# List all available versions
sdkmanager --query --archived-versions
```

**Method 3: Consult the Complete Guide**

📖 **[Complete JetPack Version & Target Reference →](JETPACK-VERSIONS.md)**

This comprehensive guide includes:
- Full version compatibility matrix
- CUDA, cuDNN, and TensorRT versions for each JetPack
- Troubleshooting common version mismatches
- Verification commands

### ⚠️ Common Version Mistakes

| ❌ Wrong | ✅ Correct | Issue |
|---------|-----------|-------|
| `jetpack_version: '36.3.0'` | `jetpack_version: '6.0'` | Use JetPack version, not L4T |
| `jetpack_version: 'latest'` | `jetpack_version: '6.0'` | SDK Manager doesn't support 'latest' |
| JP 6.0 + `JETSON_NANO_TARGETS` | JP 4.6.4 + `JETSON_NANO_TARGETS` | Nano only supports JetPack 4.x |
| JP 4.6 + `JETSON_ORIN_NANO_TARGETS` | JP 6.0 + `JETSON_ORIN_NANO_TARGETS` | Orin requires JetPack 5.0+ |

---

## 🖥️ Runner Recommendations

### ⭐ Self-Hosted Runners (Recommended)

**Why Self-Hosted?**
- ✅ **Ample disk space** (60+ GB) without cleanup scripts
- ✅ **Persistent Docker layer cache** speeds up builds significantly
- ✅ **No GitHub Cache size limits** (free tier: 10 GB)
- ✅ **Faster downloads** with persistent cache
- ✅ **No hourly minute consumption** for private repos

**Setup:**
```bash
# On your Linux server (Ubuntu 22.04 recommended)
sudo apt update
sudo apt install -y docker.io

# Add GitHub runner
# Follow: https://github.com/YOUR_ORG/YOUR_REPO/settings/actions/runners/new
```

**Workflow example:**
```yaml
jobs:
  build:
    runs-on: [self-hosted, linux, x64]
    # No disk cleanup needed!
```

---

### 🌐 GitHub-Hosted Runners

**⚠️ Important Limitations:**
- Standard runners have **~14 GB** available disk space
- JetPack downloads can be **10-20 GB** (depending on version)
- **You will run out of space** without cleanup

**✅ Solution: Maximize Build Space**

Use the community-standard [`easimon/maximize-build-space`](https://github.com/easimon/maximize-build-space) action:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      # ⚠️ CRITICAL: Run this FIRST on GitHub-hosted runners
      - name: Maximize Build Space
        uses: easimon/maximize-build-space@v10
        with:
          root-reserve-mb: 5120      # Reserve space for system
          swap-size-mb: 1024          # Add swap memory
          remove-dotnet: 'true'       # Remove unnecessary SDKs
          remove-android: 'true'
          remove-haskell: 'true'
          remove-codeql: 'true'
      
      - name: Checkout Repository
        uses: actions/checkout@v4
      
      - name: Setup JetsonForge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
```

**⚠️ DO NOT run disk cleanup on self-hosted runners** — it may delete system files!

**Conditional cleanup:**
```yaml
- name: Maximize Build Space (GitHub-Hosted Only)
  # Only run on GitHub-hosted runners
  if: ${{ !contains(runner.name, 'self-hosted') }}
  uses: easimon/maximize-build-space@v10
  with:
    root-reserve-mb: 5120
```

---

## 🔔 Webhook Configuration

The action can send authentication notifications to Slack and/or Microsoft Teams.

### Slack Webhook Setup

1. **Create Incoming Webhook:**
   - Go to https://api.slack.com/messaging/webhooks
   - Click **"Create your Slack app"** → **"From scratch"**
   - Enable **Incoming Webhooks**
   - Click **"Add New Webhook to Workspace"**
   - Select a channel and authorize

2. **Copy the Webhook URL:**
   ```
   https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
   ```

3. **Add to GitHub Secrets:**
   - Go to your repository → **Settings** → **Secrets and variables** → **Actions**
   - Click **"New repository secret"**
   - Name: `SLACK_WEBHOOK`
   - Value: `<paste webhook URL>`

4. **Use in Workflow:**
   ```yaml
   - uses: hackash/JetsonForge@main
     with:
       slack_webhook: ${{ secrets.SLACK_WEBHOOK }}
   ```

**Example Notification:**

> **🔐 NVIDIA SDK Manager Authentication Required**
> 
> Your JetsonForge GitHub Action build is waiting for authentication.
> 
> **Please click the link below to authenticate:**
> 
> 🔗 [Authenticate SDK Manager](https://static.nvidia.com/sdk-manager/login.html?code=...)

---

### Microsoft Teams Webhook Setup

1. **Create Incoming Webhook:**
   - Open Microsoft Teams
   - Navigate to your target channel
   - Click **⋯** (More options) → **Connectors**
   - Search for **"Incoming Webhook"** and click **Configure**
   - Name it "JetsonForge Notifications" and optionally upload an icon
   - Copy the webhook URL

2. **Add to GitHub Secrets:**
   - Go to your repository → **Settings** → **Secrets and variables** → **Actions**
   - Click **"New repository secret"**
   - Name: `TEAMS_WEBHOOK`
   - Value: `<paste webhook URL>`

3. **Use in Workflow:**
   ```yaml
   - uses: hackash/JetsonForge@main
     with:
       teams_webhook: ${{ secrets.TEAMS_WEBHOOK }}
   ```

**Example Notification:**

> **🔐 NVIDIA SDK Manager Authentication Required**
> 
> **JetsonForge GitHub Action**
> Authentication needed to continue build
> 
> **Timeout:** 300 seconds  
> **Action:** JetsonForge Cross-Compilation
> 
> [🔗 Authenticate Now]

---

### Using Both Webhooks

Send notifications to both Slack **and** Teams:

```yaml
- uses: hackash/JetsonForge@main
  with:
    slack_webhook: ${{ secrets.SLACK_WEBHOOK }}
    teams_webhook: ${{ secrets.TEAMS_WEBHOOK }}
```

---

## 🎯 Advanced Usage

### Docker Image Naming

By default, the action **automatically generates** image tags based on JetPack version and target device:

```yaml
# Automatic naming (recommended)
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.0'
    jetson_target: 'JETSON_ORIN_NANO_TARGETS'
    # Generates: jetson-cross-base:jp6.0-orin-nano

- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '5.1.2'
    jetson_target: 'JETSON_AGX_XAVIER_TARGETS'
    # Generates: jetson-cross-base:jp5.1.2-agx-xavier
```

**Why automatic naming?**
- ✅ Prevents cache conflicts between different versions/targets
- ✅ Multiple images can coexist on the same runner
- ✅ Clear identification of what each image contains

**Custom naming:**
```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.0'
    jetson_target: 'JETSON_ORIN_NANO_TARGETS'
    build_image_tag: 'my-custom-image:v1.0'
```

### Custom JetPack Version & Target

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '5.1.2'
    jetson_target: 'JETSON_AGX_ORIN_TARGETS'    # Auto-generates: jetson-cross-base:jp5.1.2-agx-xavier```

**Available Targets:**
- `JETSON_ORIN_NANO_TARGETS`
- `JETSON_AGX_ORIN_TARGETS`
- `JETSON_ORIN_NX_TARGETS`
- `JETSON_XAVIER_TARGETS`
- `JETSON_TX2_TARGETS`
- `JETSON_NANO_TARGETS`

### Archived/Legacy JetPack Versions

The action **automatically detects** when you're building for older JetPack versions and uses the `--archived-versions` flag:

```yaml
# Auto mode (default) - smart detection
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '4.6.4'  # Older version (< 6.0)
    jetson_target: 'JETSON_NANO_TARGETS'
    # ✅ Automatically uses --archived-versions flag

- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.0'  # Current version (>= 6.0)
    jetson_target: 'JETSON_ORIN_NANO_TARGETS'
    # ✅ Does NOT use --archived-versions flag
```

**Auto-detection logic:**
- JetPack **< 6.0** (4.x, 5.x series) → Uses `--archived-versions`
- JetPack **≥ 6.0** (current releases) → Does not use flag

**Manual override if needed:**
```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '5.1.2'
    use_archived_versions: 'true'   # Force archived flag
    # or
    use_archived_versions: 'false'  # Force no archived flag
```

### Custom Paths & Image Tag

```yaml
- uses: hackash/JetsonForge@main
  with:
    download_folder: '/mnt/cache/jetpack-downloads'
    work_folder: '/mnt/scratch/jetpack-work'
    sysroot_output: '/mnt/cache/sysroot.tar.zst'
    build_image_tag: 'my-org/jetson-builder:v1.0'  # Override auto-generated tag
```

### Authentication Timeout

```yaml
- uses: hackash/JetsonForge@main
  with:
    auth_timeout: '600'  # Wait up to 10 minutes
```

### Skip Steps (Pre-Configured Environment)

```yaml
- uses: hackash/JetsonForge@main
  with:
    skip_authentication: 'true'  # SDK Manager already authenticated
    skip_sysroot_build: 'false'
```

### Force Cache Invalidation

```yaml
- uses: hackash/JetsonForge@main
  with:
    cache_key_suffix: 'v2'  # Change to force fresh download/build
```

---

## 📝 Complete Workflow Examples

### Example 1: Basic CUDA Application

```yaml
name: Build CUDA App for Jetson

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Free Disk Space (GitHub-hosted only)
        if: ${{ !contains(runner.name, 'self-hosted') }}
        uses: easimon/maximize-build-space@v10
        with:
          root-reserve-mb: 5120
          remove-dotnet: 'true'
          remove-android: 'true'
      
      - name: Build Jetson Cross-Compiler
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
          slack_webhook: ${{ secrets.SLACK_WEBHOOK }}
      
      - name: Build Production Docker Image
        run: |
          docker build \
            --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
            --build-arg TARGET_JETPACK_TAG=r36.3.0 \
            -t my-jetson-app:${{ github.sha }} \
            -t my-jetson-app:latest \
            -f Dockerfile \
            .
      
      - name: Push to GitHub Container Registry
        if: github.ref == 'refs/heads/main'
        run: |
          IMAGE_NAME="ghcr.io/${{ github.repository_owner }}/my-jetson-app"
          
          echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          
          docker tag my-jetson-app:${{ github.sha }} "$IMAGE_NAME:${{ github.sha }}"
          docker tag my-jetson-app:latest "$IMAGE_NAME:latest"
          
          docker push "$IMAGE_NAME:${{ github.sha }}"
          docker push "$IMAGE_NAME:latest"
```

**Required Multi-Stage Dockerfile:**
```dockerfile
# Multi-stage build for Jetson deployment
ARG L4T_CROSS_BASE=jetson-cross-base:jp6.0-orin-nano
ARG TARGET_JETPACK_TAG=r36.3.0

# Stage 1: Cross-compile on x86_64
FROM ${L4T_CROSS_BASE} AS builder
WORKDIR /app
COPY . .
RUN cmake -B build -S . \
      -DCMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# Stage 2: Runtime for Jetson ARM64
FROM nvcr.io/nvidia/l4t-jetpack:${TARGET_JETPACK_TAG}
COPY --from=builder /app/build/my-app /usr/local/bin/my-app
CMD ["/usr/local/bin/my-app"]
```

**Deploy to Jetson Device:**
```bash
# Pull and run on your Jetson
docker pull ghcr.io/your-org/my-jetson-app:latest
docker run -d --runtime nvidia \
  --name my-app \
  --restart unless-stopped \
  ghcr.io/your-org/my-jetson-app:latest
```

---

### Example 2: AWS ECR Deployment

```yaml
name: Build & Package Jetson Docker Image

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: [self-hosted, linux]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Cross-Compiler Base
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
          teams_webhook: ${{ secrets.TEAMS_WEBHOOK }}
      
      - name: Build Multi-Stage Application Image
        run: |
          docker build \
            --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
            --build-arg TARGET_JETPACK_TAG=r36.3.0 \
            -t my-jetson-app:${{ github.ref_name }} \
            -f Dockerfile \
            .
      
      - name: Test on Jetson (if hardware available)
        if: ${{ env.JETSON_DEVICE_IP != '' }}
        run: |
          # Push to Jetson device and test
          docker save my-jetson-app:${{ github.ref_name }} | \
            ssh jetson@${{ env.JETSON_DEVICE_IP }} docker load
          
          ssh jetson@${{ env.JETSON_DEVICE_IP }} \
            "docker run --rm --runtime nvidia my-jetson-app:${{ github.ref_name }}"
      
      - name: Push to Registry
        run: |
          docker tag my-jetson-app:${{ github.ref_name }} \
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
          docker push ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

---

### Example 3: Matrix Build (Multiple Targets)

```yaml
name: Build for Multiple Jetson Platforms

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target:
          - name: 'Orin Nano'
            device: 'JETSON_ORIN_NANO_TARGETS'
          - name: 'AGX Orin'
            device: 'JETSON_AGX_ORIN_TARGETS'
          - name: 'Xavier'
            device: 'JETSON_XAVIER_TARGETS'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Free Disk Space
        uses: easimon/maximize-build-space@v10
        with:
          root-reserve-mb: 5120
      
      - name: Build Cross-Compiler for ${{ matrix.target.name }}
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: ${{ matrix.target.device }}
          slack_webhook: ${{ secrets.SLACK_WEBHOOK }}
      
      - name: Compile for ${{ matrix.target.name }}
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/src \
            -w /src \
            ${{ steps.jetsonforge.outputs.docker_image }} \
            bash -c "cmake -B build-${{ matrix.target.device }} -S . && cmake --build build-${{ matrix.target.device }}"
```

---

## � Docker Registry Deployment

The final deliverable for Jetson deployment is a **Docker image** pushed to a container registry. The action builds a cross-compilation base image, which you use in a multi-stage Dockerfile to create your production image.

### Required: Multi-Stage Dockerfile

Your repository needs a Dockerfile that:
1. Uses the JetsonForge cross-compiler image to build (Stage 1 - x86_64)
2. Creates a runtime image based on NVIDIA L4T (Stage 2 - ARM64)

**Example Dockerfile:**
```dockerfile
ARG L4T_CROSS_BASE=jetson-cross-base:jp6.0-orin-nano
ARG TARGET_JETPACK_TAG=r36.3.0

# Stage 1: Cross-compile on x86_64
FROM ${L4T_CROSS_BASE} AS builder
WORKDIR /app
COPY . .
RUN cmake -B build -S . \
      -DCMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# Stage 2: Runtime for Jetson ARM64
FROM nvcr.io/nvidia/l4t-jetpack:${TARGET_JETPACK_TAG}
COPY --from=builder /app/build/my-app /usr/local/bin/my-app
CMD ["/usr/local/bin/my-app"]
```

### Deployment Workflow

1. **Build cross-compiler** (handled by JetsonForge action)
2. **Build production image** (multi-stage Docker build)
3. **Push to registry** (GHCR, ECR, Docker Hub, etc.)
4. **Pull and run on Jetson** (with NVIDIA runtime)

### Supported Registries

📖 **[Complete Deployment Guide →](DEPLOYMENT.md)**

The deployment guide includes detailed examples for:
- **GitHub Container Registry (GHCR)** - Free, integrated
- **AWS Elastic Container Registry (ECR)** - IoT fleet management
- **Docker Hub** - Public images
- **Google Artifact Registry** - GCP integration
- Fleet deployment with Docker Compose
- Automated update scripts
- Security best practices

### Quick Example

```yaml
- name: Build & Push to GHCR
  run: |
    # Build production image
    docker build \
      --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
      -t my-app:latest .
    
    # Push to registry
    echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
    docker tag my-app:latest ghcr.io/${{ github.repository }}:latest
    docker push ghcr.io/${{ github.repository }}:latest
```

**Deploy to Jetson:**
```bash
docker pull ghcr.io/your-org/my-app:latest
docker run -d --runtime nvidia ghcr.io/your-org/my-app:latest
```

---

## �📥 Inputs & Outputs

### Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `jetpack_version` | JetPack version (e.g., 6.0, 5.1.2) | No | `6.0` |
| `jetson_target` | Target device identifier | No | `JETSON_ORIN_NANO_TARGETS` |
| `slack_webhook` | Slack incoming webhook URL | No | `''` |
| `teams_webhook` | Microsoft Teams webhook URL | No | `''` |
| `build_image_tag` | Docker image tag (auto-generated if empty) | No | `''` (auto) |
| `download_folder` | SDK Manager download directory | No | `~/jetpack-downloads` |
| `work_folder` | Sysroot build working directory | No | `~/jetpack-work` |
| `sysroot_output` | Output path for sysroot archive | No | `~/jetpack-sysroot.tar.zst` |
| `auth_timeout` | Authentication timeout (seconds) | No | `300` |
| `skip_authentication` | Skip authentication check | No | `false` |
| `skip_sysroot_build` | Skip sysroot generation | No | `false` |
| `cache_key_suffix` | Suffix for cache key invalidation | No | `''` |
| `use_archived_versions` | SDK Manager archived flag ('auto', 'true', 'false') | No | `'auto'` |

### Outputs

| Output | Description |
|--------|-------------|
| `docker_image` | Tag of the built Docker image |
| `sysroot_path` | Path to generated sysroot archive |
| `cache_hit_session` | Whether SDK session was restored from cache |
| `cache_hit_sysroot` | Whether sysroot was restored from cache |

**Using outputs:**
```yaml
- id: jetson
  uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.0'

- name: Show Image
  run: |
    echo "Built image: ${{ steps.jetson.outputs.docker_image }}"
    # Output: Built image: jetson-cross-base:jp6.0-orin-nano
```

---

## 💾 Caching Strategy

### What Gets Cached?

1. **SDK Manager Session** (`~/.nvsdkm`, `~/.config/sdkmanager`)
   - Size: < 1 MB
   - Duration: Until token expires (weeks/months)
   - Key: `nvsdkm-session-{jetpack_version}`

2. **JetPack Downloads** (`~/jetpack-downloads/*.deb`)
   - Size: 10-20 GB (compressed in cache)
   - Duration: Until JetPack version changes
   - Key: `jetpack-downloads-{version}-{target}`

3. **Built Sysroot** (`~/jetpack-sysroot.tar.zst`)
   - Size: 5-8 GB
   - Duration: Until downloads or version changes
   - Key: `jetpack-sysroot-{version}-{target}-{hash}`

### Cache Limits

**GitHub-Hosted Runners:**
- Free tier: 10 GB total cache per repository
- Pro/Enterprise: 10 GB total cache per repository
- ⚠️ May evict old caches when limit is reached

**Self-Hosted Runners:**
- No limits! Cache stored locally on runner
- Persistent across workflow runs
- Faster access than GitHub Cache API

### Force Cache Refresh

```yaml
- uses: hackash/JetsonForge@main
  with:
    cache_key_suffix: ${{ github.run_number }}  # Unique per run
```

Or manually delete cache:
- Repository → **Actions** → **Caches** → Delete specific cache entries

---

## 🔧 Troubleshooting

### Authentication Issues

**Problem:** Authentication link not appearing in logs

**Solution:**
1. Check that SDK Manager is installed correctly
2. Verify webhook URLs are valid (test with `curl`)
3. Check GitHub Actions logs for error messages

```bash
# Test Slack webhook
curl -X POST -H 'Content-Type: application/json' \
  --data '{"text":"Test notification"}' \
  "$SLACK_WEBHOOK_URL"
```

---

**Problem:** Authentication timeout

**Solution:**
1. Increase timeout: `auth_timeout: '600'`
2. Check if NVIDIA Developer site is responsive
3. Manually authenticate before workflow (saves to cache)

---

### Disk Space Issues

**Problem:** "No space left on device"

**Solution for GitHub-hosted:**
```yaml
- uses: easimon/maximize-build-space@v10
  with:
    root-reserve-mb: 4096
    remove-dotnet: 'true'
    remove-android: 'true'
    remove-haskell: 'true'
    remove-codeql: 'true'
```

**Solution for self-hosted:**
- Ensure runner has 60+ GB free space
- Clean old Docker images: `docker system prune -a`

---

### Download Failures

**Problem:** SDK Manager download fails

**Solution:**
1. Verify NVIDIA Developer account is active
2. Check JetPack version is valid: `sdkmanager --query --archived-versions`
3. Verify target device name is correct
4. Check network connectivity to NVIDIA servers

**For older/archived versions (JetPack < 6.0):**

The action automatically detects and uses `--archived-versions` flag for older versions. If downloads still fail:

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '4.6.4'
    use_archived_versions: 'true'  # Force archived flag
```

Check available archived versions:
```bash
sdkmanager --query --archived-versions
```

---

### Docker Build Failures

**Problem:** Sysroot not found in Docker build

**Solution:**
- Verify sysroot was generated: Check action outputs
- Ensure sysroot is copied to `sysroots/` directory
- Check build arg: `TAR_ZST_NAME` matches filename

---

### Image Tag Issues

**Problem:** Can't find the built Docker image

**Solution:**
The action automatically generates unique image tags. Use the action output:

```yaml
- id: jetsonforge
  uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.0'

- name: Use the image
  run: |
    # ✅ Correct: Use the output
    docker run ${{ steps.jetsonforge.outputs.docker_image }} ...
    
    # ❌ Wrong: Hardcoded tag may not match
    # docker run jetson-cross-base:local ...
```

**Problem:** Building for multiple targets overwrites images

**Solution:**
This is now fixed! Each JetPack version + target combination gets a unique tag:
- JetPack 6.0 + Orin Nano → `jetson-cross-base:jp6.0-orin-nano`
- JetPack 5.1.2 + Xavier → `jetson-cross-base:jp5.1.2-agx-xavier`

Multiple images can coexist on the same runner.

---

### Cache Not Working

**Problem:** Cache miss on every run

**Solution:**
1. Check cache key includes stable values (not `${{ github.run_number }}`)
2. Verify cache size is under GitHub limits (10 GB)
3. Check cache was successfully saved (not evicted)
4. Review cache keys in repository settings

---

## ❓ FAQ

### Q: Do I need a physical Jetson device?

**A:** No! JetsonForge enables cross-compilation on x86_64 machines. You only need a Jetson device for final testing and deployment.

---

### Q: How are Docker images named?

**A:** By default, the action **automatically generates unique tags** based on JetPack version and target:

```
jetson-cross-base:jp{VERSION}-{TARGET}
```

Examples:
- JetPack 6.0 + Orin Nano → `jetson-cross-base:jp6.0-orin-nano`
- JetPack 5.1.2 + AGX Xavier → `jetson-cross-base:jp5.1.2-agx-xavier`
- JetPack 4.6.4 + Nano → `jetson-cross-base:jp4.6.4-nano`

This prevents cache conflicts and allows multiple images to coexist. You can override with `build_image_tag` input if needed.

**Always use the action output** in subsequent steps:
```yaml
docker run ${{ steps.jetsonforge.outputs.docker_image }} ...
```

---

### Q: Can I use this with private repositories?

**A:** Yes! The action works with private repositories. Note that GitHub-hosted minutes are billed for private repos on free tier.

---

### Q: How long does the first run take?

**A:** 
- **First run (no cache):** 30-60 minutes (download + build)
- **Subsequent runs (cache hit):** 2-5 minutes (Docker build only)

---

### Q: What JetPack versions are supported?

**A:** Any version available via SDK Manager, including archived versions. Common versions:
- JetPack 6.0 (L4T r36.3.0) — Latest for Orin
- JetPack 5.1.2 (L4T r35.4.1) — Xavier, Orin
- JetPack 4.6.4 (L4T r32.7.4) — Nano, TX2

---

### Q: Can I customize the toolchain or sysroot?

**A:** Yes! After the action completes:
1. Extend the Docker image with additional packages
2. Mount custom toolchain files
3. Modify `scripts/make-sys-root.sh` for custom sysroot config

---

### Q: Is this faster than building on Jetson device?

**A:** Usually yes! x86_64 machines typically have:
- More CPU cores (faster compilation)
- More RAM (larger parallel builds)
- Better I/O performance
- No thermal throttling concerns

---

### Q: Can I run tests on the cross-compiled binary?

**A:** Cross-compiled ARM binaries won't run on x86 hosts. Options:
1. Use QEMU emulation (slow, limited GPU support)
2. Deploy to actual Jetson hardware for testing
3. Use GitHub Actions self-hosted runner on Jetson device

---

### Q: How do I update to a new JetPack version?

**A:** Simply change the input:
```yaml
with:
  jetpack_version: '6.1'  # New version
```

The action will automatically download new files and rebuild the sysroot.

---

### Q: What's the difference between this and NVIDIA's Docker images?

**A:** 
- **NVIDIA L4T images:** ARM64-only, run on Jetson hardware
- **JetsonForge:** x86_64 cross-compilation environment, runs on any x86 machine
- **Use both:** Build with JetsonForge, deploy with L4T base images

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/hackash/JetsonForge/issues)
- **Discussions:** [GitHub Discussions](https://github.com/hackash/JetsonForge/discussions)
- **Documentation:** [Main README](README.md)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Credits

- Built on [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager)
- Inspired by the Jetson developer community
- Community contributions welcome!

---

**Happy Building! 🚀**
