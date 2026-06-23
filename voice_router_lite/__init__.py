"""
Voice Router Lite - 离线语音控制引擎

为128MB MIPS架构路由器设计的轻量级离线语音处理模块。
所有语音处理逻辑在本地设备完成，无网络依赖。

核心组件:
- KWS:  语音唤醒 (Keyword Spotting)
- ASR:  语音识别 (Automatic Speech Recognition)
- NLU:  指令理解 (Natural Language Understanding)
- TTS:  语音合成 (Text-to-Speech, 预录制音频拼接)
- Device: 设备控制 (GPIO/Relay/Fan/LED)

使用方法:
    from voice_router_lite import VoiceRouterPipeline

    pipeline = VoiceRouterPipeline(config_path="config.yaml")
    pipeline.initialize()
    pipeline.start()
    # ... 语音交互 ...
    pipeline.stop()
"""

from voice_router_lite.config import PipelineConfig, ModelPaths, AudioConfig
from voice_router_lite.pipeline import VoiceRouterPipeline

__version__ = "0.1.0"
__all__ = [
    "VoiceRouterPipeline",
    "PipelineConfig",
    "ModelPaths",
    "AudioConfig",
]
