#!/bin/bash
# Cross-compile native ASR worker for OpenWrt MIPS (skeleton).
#
# You must provide an OpenWrt SDK or toolchain that already built sherpa-onnx
# for the target. This script only wires CMake to the SDK.
#
# Usage:
#   export OPENWRT_SDK=/path/to/openwrt-sdk
#   export SHERPA_ONNX_ROOT=$OPENWRT_SDK/staging_dir/.../sherpa-onnx
#   export STAGING_DIR=$OPENWRT_SDK/staging_dir/target-mipsel_24kc_musl
#   bash native/build-mips-openwrt.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/native/build-mips"

: "${OPENWRT_SDK:?Set OPENWRT_SDK}"
: "${STAGING_DIR:?Set STAGING_DIR to OpenWrt target staging_dir}"
: "${SHERPA_ONNX_ROOT:?Set SHERPA_ONNX_ROOT to cross-compiled sherpa-onnx install}"

TOOLCHAIN_FILE="$ROOT/native/toolchain-openwrt.cmake"
cat > "$TOOLCHAIN_FILE" <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER "$STAGING_DIR/usr/bin/mipsel-openwrt-linux-gcc")
set(CMAKE_CXX_COMPILER "$STAGING_DIR/usr/bin/mipsel-openwrt-linux-g++")
set(CMAKE_FIND_ROOT_PATH "$STAGING_DIR")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

cmake -S "$ROOT/native" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DSHERPA_ONNX_ROOT="$SHERPA_ONNX_ROOT"

cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 2)"

OUT="$ROOT/native/bin/voice-router-asr-worker-mips"
mkdir -p "$ROOT/native/bin"
cp -f "$BUILD_DIR/voice-router-asr-worker" "$OUT"
echo "✅ Built: $OUT"
echo "Copy to router: scp $OUT root@router:/opt/voice_router/bin/voice-router-asr-worker"
