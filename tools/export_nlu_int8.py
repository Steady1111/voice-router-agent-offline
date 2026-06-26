#!/usr/bin/env python3
"""Export NLU INT8 ONNX from existing fp32 model (task 1C.1)."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32",
        default="models/nlu/intent_model.onnx",
        help="Source fp32 ONNX",
    )
    parser.add_argument(
        "--out",
        default="models/nlu/intent_model.int8.onnx",
        help="INT8 output path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.fp32):
        print(f"Missing fp32 model: {args.fp32}", file=sys.stderr)
        return 1

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("Install onnxruntime for quantization", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quantize_dynamic(args.fp32, args.out, weight_type=QuantType.QUInt8)
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
