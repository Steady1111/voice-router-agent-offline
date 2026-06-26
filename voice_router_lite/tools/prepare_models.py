"""Validate and optionally download offline models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_models() -> list[str]:
    from voice_router_lite.config import DEFAULT_CONFIG

    missing = DEFAULT_CONFIG.models.validate()
    nlu_files = [
        DEFAULT_CONFIG.models.nlu_vocab,
        DEFAULT_CONFIG.models.nlu_intent_labels,
        DEFAULT_CONFIG.models.nlu_slot_labels,
    ]
    for path in nlu_files:
        if not os.path.exists(path) and path not in missing:
            missing.append(path)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Voice Router Lite models")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ASR/KWS models via download_models.sh",
    )
    parser.add_argument(
        "--train-nlu",
        action="store_true",
        help="Train NLU model (requires PyTorch)",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    os.chdir(root)

    if args.download:
        script = os.path.join(root, "download_models.sh")
        if not os.path.isfile(script):
            print(f"缺少脚本: {script}", file=sys.stderr)
            return 1
        print("运行 download_models.sh ...")
        subprocess.run(["bash", script], check=True)

    if args.train_nlu:
        print("训练 NLU 模型 ...")
        subprocess.run(
            [sys.executable, "-m", "voice_router_lite.nlu.train_nlu"],
            check=True,
        )

    missing = validate_models()
    if missing:
        print("缺少模型文件:")
        for path in missing:
            print(f"  - {path}")
        print("\n建议: python -m voice_router_lite prepare-models --download")
        if not os.path.exists("models/nlu/intent_model.onnx"):
            print("      python -m voice_router_lite prepare-models --train-nlu")
        return 1

    print("✅ 模型检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
