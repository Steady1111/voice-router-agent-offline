"""
快速测试 KWS 模型是否能识别唤醒词。
用 Mac 麦克风录音，实时运行 KWS 检测。
"""
import sys
import os
import time
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.config import ModelPaths, AudioConfig
from voice_router_lite.kws.engine import KWSEngine, KWSResult

SAMPLE_RATE = 16000
CHUNK = 480  # 30ms per chunk


def main():
    print("=" * 60)
    print("KWS 实时测试 - 请对着麦克风说唤醒词")
    print("=" * 60)

    paths = ModelPaths()
    cfg = AudioConfig()

    # 初始化 KWS 引擎
    kws = KWSEngine(paths, cfg)
    ok = kws.initialize()
    print(f"KWS 初始化: {ok}, 模式: {kws.mode}")

    if kws.mode != "sherpa":
        print("❌ sherpa-onnx 模式未启用，退出")
        return

    # 打印关键词
    kw_file = paths.kws_keywords
    if os.path.exists(kw_file):
        with open(kw_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    print(f"  关键词: {line}")

    print(f"\n🎤 开始录音... 说 '小T小T' 试试")
    print("按 Ctrl+C 停止\n")

    frame_count = 0
    detect_count = 0
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal frame_count, detect_count
        if status:
            print(f"⚠️ 音频状态: {status}")

        audio = indata[:, 0].astype(np.float32)  # mono channel
        energy = float(np.sqrt(np.mean(audio ** 2)))

        kws_result = kws.detect(audio)
        frame_count += 1

        if kws_result.detected:
            detect_count += 1
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"🎯 检测到唤醒词! keyword='{kws_result.keyword}' confidence={kws_result.confidence:.4f}")
            print(f"   耗时: {elapsed:.1f}s, 帧数: {frame_count}")
            print(f"{'='*60}\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⏹️  停止。总计 {frame_count} 帧, {elapsed:.1f}s, "
              f"检测 {detect_count} 次")


if __name__ == "__main__":
    main()
