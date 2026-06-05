#!/usr/bin/env bash
set -euo pipefail

JP_VERSION="${JP_VERSION:?JP_VERSION env var is required}"
TARGET="${TARGET:?TARGET env var is required}"

echo "::group::Validating JetPack configuration"

echo "Validating:"
echo "  JetPack Version: $JP_VERSION"
echo "  Target Device:   $TARGET"
echo ""

VALIDATION_PASSED=true

if [[ "$JP_VERSION" =~ ^r[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "::warning::Detected L4T release format ($JP_VERSION). SDK Manager expects JetPack version."
  echo "::warning::Example: Use '6.0' instead of 'r36.3.0'"
  echo "::warning::Continuing anyway - SDK Manager may accept L4T format..."
fi

if [[ "$TARGET" == *"ORIN"* ]]; then
  JP_MAJOR=$(echo "$JP_VERSION" | cut -d. -f1)
  if [[ "$JP_MAJOR" -lt 5 ]] 2>/dev/null; then
    echo "::error::Orin devices require JetPack 5.0 or later. You specified: $JP_VERSION"
    echo "::error::Recommended: Use jetpack_version: '6.0' for Orin devices"
    VALIDATION_PASSED=false
  fi
fi

if [[ "$TARGET" == *"NANO"* ]] || [[ "$TARGET" == *"TX2"* ]]; then
  if [[ ! "$TARGET" == *"ORIN"* ]]; then
    JP_MAJOR=$(echo "$JP_VERSION" | cut -d. -f1)
    if [[ "$JP_MAJOR" -ge 5 ]] 2>/dev/null; then
      echo "::error::Jetson Nano and TX2 only support JetPack 4.x. You specified: $JP_VERSION"
      echo "::error::Recommended: Use jetpack_version: '4.6.4' for Nano/TX2 devices"
      VALIDATION_PASSED=false
    fi
  fi
fi

if [[ "$JP_VERSION" == "latest" ]]; then
  echo "::error::SDK Manager does not support 'latest' as a version specifier"
  echo "::error::Please specify an explicit version like '6.0', '5.1.2', or '4.6.4'"
  VALIDATION_PASSED=false
fi

if [[ "$VALIDATION_PASSED" == "false" ]]; then
  echo ""
  echo "::error::Configuration validation failed!"
  echo "::error::See: https://github.com/hackash/JetsonForge/blob/main/JETPACK-VERSIONS.md"
  exit 1
fi

echo "✅ Configuration validated successfully"
echo "::endgroup::"
