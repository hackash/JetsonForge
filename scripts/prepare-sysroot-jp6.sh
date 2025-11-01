
#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
This script provides guidance for assembling a JetPack 6.0 sysroot tarball.

Goal: Create sysroots/jetpack-6.0-aarch64.tar.zst with:
  - /usr/local/cuda (CUDA 12.2)
  - cuDNN 8.9.4.25 (runtime+dev)
  - TensorRT 8.6.2.3 (runtime+dev)
  - /usr/include, /lib, /usr/lib/aarch64-linux-gnu
  - /etc/ld.so.conf.d/* and /etc/alternatives as needed
  - /var/lib/dpkg/status (so verify-sysroot can check versions)

Two common ways:

1) From an actual Jetson (once):
   - Make sure the device is on JetPack 6.0 and has the required CUDA/cuDNN/TensorRT packages installed.
   - rsync the rootfs (excluding /proc, /sys, /dev, /tmp) to a staging dir on your PC:
     rsync -aHAX --numeric-ids --exclude={"/proc/*","/sys/*","/dev/*","/tmp/*"} jetson:/ /tmp/jp6-rootfs
   - Then create the archive:
     tar -C /tmp/jp6-rootfs -I 'zstd -19' -cpf sysroots/jetpack-6.0-aarch64.tar.zst .

2) From the L4T sample rootfs + NVIDIA debs:
   - Use NVIDIA SDK Manager or download L4T sample rootfs for R36.x
   - Apply the board support packages
   - Install CUDA 12.2, cuDNN 8.9.4.25, TensorRT 8.6.2.3 (aarch64 debs)
   - Package it:
     tar -C <rootfs> -I 'zstd -19' -cpf sysroots/jetpack-6.0-aarch64.tar.zst .

Place the resulting tarball at sysroots/jetpack-6.0-aarch64.tar.zst before building the cross image.
EOF
