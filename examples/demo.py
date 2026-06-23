"""
离线语音管道 - 快速演示脚本

展示 VoiceRouterPipeline 的核心使用方式和 API。

运行方式:
    python examples/demo.py

前置条件:
    pip install -e ".[audio]"
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite import VoiceRouterPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# 演示 1: 文本指令模拟（不需要麦克风）
# ---------------------------------------------------------------------------
def demo_text_mode():
    """纯文本指令理解演示（不依赖麦克风和 ASR 模型）"""
    print("\n" + "=" * 60)
    print("演示 1: 纯文本指令理解 (跳过音频采集和 ASR)")
    print("=" * 60)

    pipeline = VoiceRouterPipeline()
    pipeline.initialize()

    test_commands = [
        "打开风扇",
        "关闭灯光",
        "调大一点",
        "睡眠模式",
        "亮度调暗一点",
        "WiFi断了",
        "今天天气不错",
        "帮助",
    ]

    for cmd in test_commands:
        print(f"\n📣 指令: '{cmd}'")
        result = pipeline.process_utterance(text=cmd)

        print(f"   意图: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"   槽位: {result['slots']}")
        print(f"   结果: {'✅ 成功' if result['success'] else '❌ 失败'} - {result['message']}")

    pipeline.close()


# ---------------------------------------------------------------------------
# 演示 2: 合成音频模拟（生成纯音模拟语音）
# ---------------------------------------------------------------------------
def demo_synthetic_audio():
    """使用生成的音频模拟 ASR 处理（需要 sherpa-onnx 模型文件）"""
    print("\n" + "=" * 60)
    print("演示 2: 合成音频处理 (生成纯音模拟语音输入)")
    print("=" * 60)

    pipeline = VoiceRouterPipeline()
    pipeline.initialize()

    # 生成一段模拟音频 (1秒，440Hz)
    duration_sec = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    fake_audio = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

    print("\n📣 处理合成音频...")
    result = pipeline.process_utterance(audio_data=fake_audio)

    print(f"   ASR 转写: '{result.get('text', '')}'")
    print(f"   意图: {result.get('intent', 'unknown')} (置信度: {result.get('confidence', 0):.2f})")
    print(f"   结果: {'✅ 成功' if result.get('success') else '❌ 失败'} - {result.get('message', '')}")

    pipeline.close()


# ---------------------------------------------------------------------------
# 演示 3: 音频降噪处理
# ---------------------------------------------------------------------------
def demo_denoise():
    """降噪和预处理演示"""
    print("\n" + "=" * 60)
    print("演示 3: 音频降噪与预处理")
    print("=" * 60)

    from voice_router_lite.audio.denoise import AudioPreprocessor
    from voice_router_lite.config import AudioConfig

    cfg = AudioConfig(denoise_enabled=True, denoise_algorithm="spectral_gate")
    preproc = AudioPreprocessor(cfg)

    # 生成带噪声的信号
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    clean_signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    noise = (np.random.randn(len(clean_signal)) * 0.1).astype(np.float32)
    noisy_signal = clean_signal + noise

    print(f"   干净信号 RMS: {np.sqrt(np.mean(clean_signal**2)):.4f}")
    print(f"   带噪信号 RMS: {np.sqrt(np.mean(noisy_signal**2)):.4f}")

    # 校准噪音
    noise_calib = (np.random.randn(int(sample_rate * 1.0)) * 0.05).astype(np.float32)
    preproc.calibrate_noise(noise_calib)

    # 降噪
    cleaned = preproc.process(noisy_signal.astype(np.float32))
    print(f"   降噪后 RMS: {np.sqrt(np.mean(cleaned**2)):.4f}")
    print("   ✅ 降噪处理完成")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎙️ Voice Router Lite - 离线语音控制管道演示")
    print("=" * 60)

    # 先做不需要外部依赖的演示
    demo_text_mode()

    try:
        demo_denoise()
    except ImportError as e:
        print(f"\n⚠️ 降噪演示跳过 (缺少依赖): {e}")

    try:
        demo_synthetic_audio()
    except Exception as e:
        print(f"\n⚠️ 合成音频演示跳过: {e}")

    print("\n" + "=" * 60)
    print("✅ 演示完成")
