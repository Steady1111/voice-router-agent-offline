"""
ASR 一次性子进程 worker。

从 stdin 读取 int16 PCM，向 stdout 打印识别文本（单行）。
父进程通过 subprocess 拉起，退出后释放全部 ASR 运行时内存。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voice Router ASR worker")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    raw = sys.stdin.buffer.read()
    if not raw:
        return 0

    audio = np.frombuffer(raw, dtype=np.int16)
    if len(audio) == 0:
        return 0

    from voice_router_lite.asr.engine import ASREngine
    from voice_router_lite.config import ModelPaths, AudioConfig

    models = ModelPaths(base_dir=os.getenv("VOICE_ROUTER_MODELS_DIR", "models"))
    audio_cfg = AudioConfig(sample_rate=args.sample_rate)

    asr = ASREngine(models, audio_cfg, num_threads=args.threads)
    if not asr.initialize():
        return 1

    text = asr.transcribe(audio).text.strip()
    sys.stdout.write(text)
    sys.stdout.flush()
    asr.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
