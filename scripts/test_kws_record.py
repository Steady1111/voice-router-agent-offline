"""
简单 KWS 测试：录音 N 秒 → KWS 检测
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
from voice_router_lite.config import ModelPaths, AudioConfig
from voice_router_lite.kws.engine import KWSEngine

SAMPLE_RATE = 16000
DURATION = 5  # 录音5秒


def main():
    print("初始化...")
    paths = ModelPaths()
    cfg = AudioConfig()

    kws = KWSEngine(paths, cfg)
    ok = kws.initialize()
    print(f"KWS: ok={ok}, mode={kws.mode}")

    print(f"\n🎤 录音 {DURATION} 秒... (请说 '小T小T')")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype=np.float32)
    sd.wait()
    audio = audio[:, 0]
    print(f"录音完成: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.1f}s")
    print(f"能量: {float(np.sqrt(np.mean(audio**2))):.5f}")

    # 切成 30ms 帧喂给 KWS
    CHUNK = 480
    total_frames = 0
    for i in range(0, len(audio) - CHUNK, CHUNK):
        chunk = audio[i:i + CHUNK].astype(np.float32)
        result = kws.detect(chunk)
        total_frames += 1
        if result.detected:
            print(f"🎯 检测到! keyword='{result.keyword}' confidence={result.confidence:.4f} "
                  f"位置={i/SAMPLE_RATE:.1f}s")
            return

    print(f"❌ 未检测到唤醒词 (总计 {total_frames} 帧)")
    kws.close()


if __name__ == "__main__":
    main()
