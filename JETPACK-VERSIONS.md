# JetPack Version & Target Device Reference

This document provides the official mappings between JetPack versions, L4T releases, and supported Jetson hardware targets for use with the JetsonForge GitHub Action.

---

## 📋 Quick Reference Table

| JetPack | L4T Version | SDK Manager Value | Supported Devices | Release Date |
|---------|-------------|-------------------|-------------------|--------------|
| **6.1** | r36.4.0 | `6.1` | Orin series | Q2 2026 |
| **6.0** | r36.3.0 | `6.0` | Orin series | Q4 2023 |
| **5.1.3** | r35.5.0 | `5.1.3` | Orin, Xavier | Q4 2023 |
| **5.1.2** | r35.4.1 | `5.1.2` | Orin, Xavier | Q3 2023 |
| **5.1.1** | r35.3.1 | `5.1.1` | Orin, Xavier | Q1 2023 |
| **5.1** | r35.2.1 | `5.1` | Orin, Xavier | Q4 2022 |
| **5.0.2** | r35.1.0 | `5.0.2` | Orin, Xavier | Q3 2022 |
| **5.0.1** | r34.1.1 | `5.0.1` | Xavier | Q2 2022 |
| **5.0** | r34.1 | `5.0` | Xavier | Q1 2022 |
| **4.6.5** | r32.7.5 | `4.6.5` | Nano, TX2, Xavier | Q4 2023 |
| **4.6.4** | r32.7.4 | `4.6.4` | Nano, TX2, Xavier | Q3 2023 |
| **4.6.3** | r32.7.3 | `4.6.3` | Nano, TX2, Xavier | Q1 2023 |
| **4.6.2** | r32.7.2 | `4.6.2` | Nano, TX2, Xavier | Q3 2022 |
| **4.6.1** | r32.7.1 | `4.6.1` | Nano, TX2, Xavier | Q1 2022 |
| **4.6** | r32.6.1 | `4.6` | Nano, TX2, Xavier | Q3 2021 |
| **4.5.1** | r32.5.1 | `4.5.1` | Nano, TX2, Xavier | Q2 2021 |
| **4.5** | r32.5.0 | `4.5` | Nano, TX2, Xavier | Q1 2021 |

---

## 🎯 Jetson Target Device Identifiers

### Orin Family (JetPack 5.0+, 6.0+)

| Device | SDK Manager Target | Recommended JP |
|--------|-------------------|----------------|
| **Jetson Orin Nano 8GB** | `JETSON_ORIN_NANO_TARGETS` | 6.0, 6.1 |
| **Jetson Orin Nano 4GB** | `JETSON_ORIN_NANO_TARGETS` | 6.0, 6.1 |
| **Jetson Orin NX 16GB** | `JETSON_ORIN_NX_TARGETS` | 6.0, 6.1 |
| **Jetson Orin NX 8GB** | `JETSON_ORIN_NX_TARGETS` | 6.0, 6.1 |
| **Jetson AGX Orin 64GB** | `JETSON_AGX_ORIN_TARGETS` | 6.0, 6.1 |
| **Jetson AGX Orin 32GB** | `JETSON_AGX_ORIN_TARGETS` | 6.0, 6.1 |
| **Jetson AGX Orin (Industrial)** | `JETSON_AGX_ORIN_TARGETS` | 6.0, 6.1 |

### Xavier Family (JetPack 4.5 - 5.1.3)

| Device | SDK Manager Target | Recommended JP |
|--------|-------------------|----------------|
| **Jetson AGX Xavier 64GB** | `JETSON_AGX_XAVIER_TARGETS` | 5.1.2, 4.6.4 |
| **Jetson AGX Xavier 32GB** | `JETSON_AGX_XAVIER_TARGETS` | 5.1.2, 4.6.4 |
| **Jetson AGX Xavier (Industrial)** | `JETSON_AGX_XAVIER_TARGETS` | 5.1.2, 4.6.4 |
| **Jetson Xavier NX 16GB** | `JETSON_XAVIER_NX_TARGETS` | 5.1.2, 4.6.4 |
| **Jetson Xavier NX 8GB** | `JETSON_XAVIER_NX_TARGETS` | 5.1.2, 4.6.4 |

### Nano & TX2 Family (JetPack 4.x only)

| Device | SDK Manager Target | Recommended JP |
|--------|-------------------|----------------|
| **Jetson Nano (4GB)** | `JETSON_NANO_TARGETS` | 4.6.4 |
| **Jetson Nano (2GB)** | `JETSON_NANO_TARGETS` | 4.6.4 |
| **Jetson TX2** | `JETSON_TX2_TARGETS` | 4.6.4 |
| **Jetson TX2i (Industrial)** | `JETSON_TX2_TARGETS` | 4.6.4 |
| **Jetson TX2 4GB** | `JETSON_TX2_TARGETS` | 4.6.4 |

---

## 🔍 How to Find Available Versions

### Method 1: Query SDK Manager (Recommended)

If you have SDK Manager installed locally:

```bash
# List all available JetPack versions
sdkmanager --query non-interactive --action downloadonly

# Include archived/older versions
sdkmanager --query non-interactive --action downloadonly --archived-versions
```

**Example output:**
```
Available versions:
- JetPack 6.1 (r36.4.0)
- JetPack 6.0 (r36.3.0)
- JetPack 5.1.3 (r35.5.0)
- JetPack 5.1.2 (r35.4.1)
...
```

### Method 2: NVIDIA Developer Website

Visit the official documentation:
- **JetPack Archive**: https://developer.nvidia.com/embedded/jetpack-archive
- **L4T Archive**: https://developer.nvidia.com/embedded/linux-tegra-archive

### Method 3: Check Device Documentation

Each Jetson device has specific JetPack compatibility:
- **Orin series**: https://developer.nvidia.com/embedded/jetson-orin
- **Xavier series**: https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit
- **Nano series**: https://developer.nvidia.com/embedded/jetson-nano-developer-kit

---

## 📝 Usage Examples

### Example 1: Orin Nano with Latest JetPack

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.0'
    jetson_target: 'JETSON_ORIN_NANO_TARGETS'
```

### Example 2: AGX Xavier with Stable Release

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '5.1.2'
    jetson_target: 'JETSON_AGX_XAVIER_TARGETS'
```

### Example 3: Nano with Final JetPack 4.x

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '4.6.4'
    jetson_target: 'JETSON_NANO_TARGETS'
```

### Example 4: Using L4T Version Directly

Some configurations may accept L4T release tags:

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: 'r36.3.0'  # L4T release
    jetson_target: 'JETSON_ORIN_NANO_TARGETS'
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Invalid version specified"

**Problem:** SDK Manager doesn't recognize the version string.

**Solutions:**
1. Use the JetPack version number (e.g., `6.0`), not the L4T version
2. Add `--archived-versions` flag if using older versions
3. Verify version exists with `sdkmanager --query`

**Action Configuration:**
The action automatically adds `--archived-versions` flag, so older versions should work.

---

### Issue 2: "Target not found for version"

**Problem:** Specified target device is not compatible with JetPack version.

**Examples of incompatible combinations:**
- ❌ JetPack 6.0 + `JETSON_NANO_TARGETS` (Nano only supports up to 4.6.x)
- ❌ JetPack 4.6 + `JETSON_ORIN_NANO_TARGETS` (Orin requires 5.0+)

**Solution:** Check the compatibility table above and use appropriate versions.

---

### Issue 3: "Multiple targets or unsure which to use"

**Problem:** You have multiple Jetson devices or want to build for a family.

**Solution:** SDK Manager targets often cover device families:

```yaml
# This covers ALL Xavier NX variants (8GB, 16GB)
jetson_target: 'JETSON_XAVIER_NX_TARGETS'

# This covers ALL AGX Orin variants (32GB, 64GB, Industrial)
jetson_target: 'JETSON_AGX_ORIN_TARGETS'
```

**For multiple device families, run separate jobs:**

```yaml
strategy:
  matrix:
    include:
      - jetpack: '6.0'
        target: 'JETSON_ORIN_NANO_TARGETS'
      - jetpack: '5.1.2'
        target: 'JETSON_AGX_XAVIER_TARGETS'
```

---

## 🧪 Verification Commands

### Verify JetPack Version in Sysroot

After the action completes, verify the installed version:

```bash
docker run --rm jetson-cross-base:local bash -c '
  cat $SYSROOT/usr/local/cuda/version.json 2>/dev/null || echo "Not found"
  cat $SYSROOT/etc/nv_tegra_release 2>/dev/null || echo "Not found"
'
```

### Check CUDA Version

```bash
docker run --rm jetson-cross-base:local bash -c '
  ls -la $SYSROOT/usr/local/cuda
  cat $SYSROOT/usr/local/cuda/version.txt 2>/dev/null || \
  cat $SYSROOT/usr/local/cuda/version.json 2>/dev/null
'
```

### Verify L4T Release

```bash
docker run --rm jetson-cross-base:local bash -c '
  cat $SYSROOT/etc/nv_tegra_release
'
```

Expected output example:
```
# R36 (release), REVISION: 3.0, GCID: 36360404, BOARD: generic, EABI: aarch64
```

---

## 📊 Component Version Matrix

### JetPack 6.0 (L4T r36.3.0)

| Component | Version |
|-----------|---------|
| CUDA | 12.2 |
| cuDNN | 8.9.4 |
| TensorRT | 8.6.2 |
| VPI | 3.0 |
| OpenCV | 4.8.0 |
| Python | 3.10 |

### JetPack 5.1.2 (L4T r35.4.1)

| Component | Version |
|-----------|---------|
| CUDA | 11.4 |
| cuDNN | 8.6.0 |
| TensorRT | 8.5.2 |
| VPI | 2.3 |
| OpenCV | 4.5.4 |
| Python | 3.8 |

### JetPack 4.6.4 (L4T r32.7.4)

| Component | Version |
|-----------|---------|
| CUDA | 10.2 |
| cuDNN | 8.2.1 |
| TensorRT | 8.2.1 |
| VPI | 1.2 |
| OpenCV | 4.1.1 |
| Python | 3.6 |

---

## 🔗 Official Resources

- **JetPack Release Notes**: https://developer.nvidia.com/embedded/jetpack
- **L4T Documentation**: https://docs.nvidia.com/jetson/
- **SDK Manager User Guide**: https://docs.nvidia.com/sdk-manager/
- **Developer Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

---

## 💡 Best Practices

### 1. **Pin Versions in Production**

```yaml
# Good: Specific version
jetpack_version: '6.0'

# Avoid: Using "latest" or omitting version
# jetpack_version: 'latest'  # Not supported by SDK Manager
```

### 2. **Match Deployment Hardware**

Always use the same JetPack version for cross-compilation as deployed on your physical Jetson device:

```yaml
# If your physical Jetson Orin Nano runs JetPack 6.0
jetpack_version: '6.0'
jetson_target: 'JETSON_ORIN_NANO_TARGETS'
```

### 3. **Test with Multiple Versions**

Use matrix builds for compatibility testing:

```yaml
strategy:
  matrix:
    jetpack: ['6.0', '5.1.2']
```

### 4. **Cache Key Suffix for Version Changes**

When upgrading JetPack versions, add a cache suffix:

```yaml
- uses: hackash/JetsonForge@main
  with:
    jetpack_version: '6.1'
    cache_key_suffix: 'v2'  # Force new cache
```

---

## 🆘 Getting Help

If you encounter issues:

1. **Verify version availability**: Run `sdkmanager --query --archived-versions`
2. **Check logs**: Look for SDK Manager error messages in GHA output
3. **Validate target**: Confirm your device is supported by the JetPack version
4. **Consult this guide**: Cross-reference the tables above
5. **Open an issue**: https://github.com/hackash/JetsonForge/issues

---

**Last Updated**: May 2026
**Maintained by**: JetsonForge Contributors
