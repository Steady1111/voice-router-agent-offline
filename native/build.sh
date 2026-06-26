#!/bin/bash
# Build native ASR worker for the current host (macOS/Linux dev machine).
#
# Prerequisites: cmake, C++ compiler, and sherpa-onnx C API.
# If SHERPA_ONNX_ROOT is unset, uses the pip-installed sherpa_onnx package.
#
# Usage:
#   bash native/build.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/native/build"

if [[ -z "${SHERPA_ONNX_ROOT:-}" ]]; then
  SHERPA_ONNX_ROOT="$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("sherpa_onnx")
if spec and spec.origin:
    import os
    print(os.path.dirname(spec.origin))
PY
)"
  if [[ -z "$SHERPA_ONNX_ROOT" || ! -f "$SHERPA_ONNX_ROOT/lib/libsherpa-onnx-c-api.dylib" && ! -f "$SHERPA_ONNX_ROOT/lib/libsherpa-onnx-c-api.so" ]]; then
    echo "ERROR: sherpa_onnx pip package not found or missing C API lib" >&2
    echo "  pip3 install sherpa-onnx" >&2
    echo "  or export SHERPA_ONNX_ROOT=/path/to/sherpa-onnx/install" >&2
    exit 1
  fi
  echo "Using pip sherpa_onnx: $SHERPA_ONNX_ROOT"
fi

CMAKE="${CMAKE:-cmake}"
if ! command -v "$CMAKE" >/dev/null 2>&1; then
  CMAKE="/opt/homebrew/bin/cmake"
fi

"$CMAKE" -S "$ROOT/native" -B "$BUILD_DIR" -DSHERPA_ONNX_ROOT="$SHERPA_ONNX_ROOT"
"$CMAKE" --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

BIN="$BUILD_DIR/voice-router-asr-worker"
mkdir -p "$ROOT/native/bin"
cp -f "$BIN" "$ROOT/native/bin/voice-router-asr-worker"
echo "✅ Built: $ROOT/native/bin/voice-router-asr-worker"
