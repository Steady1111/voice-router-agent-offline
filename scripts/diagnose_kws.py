"""
诊断：用 ASR 模型识别音频，输出 token 序列 → 用于生成正确的 KWS 关键词格式。

用法：python scripts/diagnose_kws.py
对着麦克风说 "小T小T"，脚本会输出模型识别的文本和 token
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
from voice_router_lite.config import ModelPaths, AudioConfig
from voice_router_lite.asr.engine import ASREngine

SAMPLE_RATE = 16000


def main():
    print("=" * 60)
    print("ASR 诊断 - 说话后看模型输出什么")
    print("=" * 60)

    paths = ModelPaths()
    cfg = AudioConfig()

    asr = ASREngine(paths, cfg)
    ok = asr.initialize()
    print(f"ASR 初始化: {ok}")

    # 加载 tokens
    tokens_file = paths.asr_tokens
    tokens = {}
    if os.path.exists(tokens_file):
        with open(tokens_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    tokens[int(parts[1])] = parts[0]
    print(f"ASR tokens 加载: {len(tokens)} 个")

    # 加载 KWS tokens
    kws_tokens = {}
    if os.path.exists(paths.kws_tokens):
        with open(paths.kws_tokens, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    kws_tokens[int(parts[1])] = parts[0]
    print(f"KWS tokens 加载: {len(kws_tokens)} 个")

    DURATION = 4
    print(f"\n🎤 录音 {DURATION} 秒... 请说 '小T小T'")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype=np.float32)
    sd.wait()
    audio = audio[:, 0]
    print(f"录音完成: {len(audio)} samples")

    # 转 int16
    audio_int16 = (audio * 32767).astype(np.int16)

    print("\n--- ASR 识别结果 ---")
    try:
        result = asr.transcribe(audio_int16)
        print(f"文本: '{result.text}'")
        print(f"置信度: {result.confidence}")
        print(f"Tokens: {result.tokens}")
    except Exception as e:
        print(f"ASR 异常: {e}")


if __name__ == "__main__":
    main()
