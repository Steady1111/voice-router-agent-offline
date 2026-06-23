"""
全局配置管理

统一管理模型路径、音频参数、设备配置、性能指标。
所有配置项均有默认值，支持从 YAML/JSON 文件加载。
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 性能指标约束
# ---------------------------------------------------------------------------
@dataclass
class PerformanceTargets:
    """离线处理性能指标"""

    # 延迟目标（毫秒）
    kws_latency_ms: int = 200           # 唤醒词检测延迟 < 200ms
    asr_latency_ms: int = 1500          # ASR 端到端延迟 < 1.5s
    nlu_latency_ms: int = 100           # NLU 推理延迟 < 100ms
    tts_latency_ms: int = 300           # TTS 播放延迟 < 300ms
    total_latency_ms: int = 2000        # 端到端总延迟 < 2s

    # 内存限制
    total_memory_mb: int = 50           # 总内存预算 50MB
    asr_model_memory_mb: int = 30       # ASR 模型 + 运行时
    kws_model_memory_mb: int = 8        # KWS 模型 + 运行时
    nlu_model_memory_mb: int = 5        # NLU 模型 + 运行时
    audio_buffer_memory_mb: int = 5     # 音频缓冲

    # 模型体积限制
    asr_model_size_mb: int = 30         # ASR 模型文件 < 30MB
    kws_model_size_mb: int = 5          # KWS 模型文件 < 5MB
    nlu_model_size_mb: int = 3          # NLU 模型文件 < 3MB
    tts_clips_total_mb: int = 5         # TTS 预录制音频 < 5MB


# ---------------------------------------------------------------------------
# 模型路径配置
# ---------------------------------------------------------------------------
@dataclass
class ModelPaths:
    """模型文件路径"""

    base_dir: str = "models"

    # KWS 唤醒词模型
    kws_model: str = "models/kws/wake_word.onnx"
    kws_tokens: str = "models/kws/tokens.txt"

    # ASR 模型 (sherpa-onnx Zipformer)
    asr_encoder: str = "models/asr/encoder.onnx"
    asr_decoder: str = "models/asr/decoder.onnx"
    asr_joiner: str = "models/asr/joiner.onnx"
    asr_tokens: str = "models/asr/tokens.txt"

    # NLU 模型
    nlu_intent_model: str = "models/nlu/intent_model.onnx"
    nlu_slot_model: str = "models/nlu/slot_model.onnx"
    nlu_vocab: str = "models/nlu/vocab.json"
    nlu_intent_labels: str = "models/nlu/intent_labels.json"
    nlu_slot_labels: str = "models/nlu/slot_labels.json"

    # TTS 预录制音频
    tts_clips_dir: str = "audio_clips"
    tts_index: str = "audio_clips/index.json"

    def validate(self) -> List[str]:
        """检查必需文件是否存在，返回缺失文件列表"""
        required = [
            self.kws_model, self.kws_tokens,
            self.asr_encoder, self.asr_decoder, self.asr_joiner, self.asr_tokens,
            self.nlu_intent_model, self.nlu_vocab, self.nlu_intent_labels,
        ]
        return [f for f in required if not os.path.exists(f)]


# ---------------------------------------------------------------------------
# 音频配置
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """音频输入/输出标准接口配置"""

    # ---- 输入参数 ----
    sample_rate: int = 16000           # 采样率 16kHz
    bit_depth: int = 16                # 位深 16bit
    channels: int = 1                  # 单声道 mono
    chunk_duration_ms: int = 30        # 每次读取分片时长 (30ms = 480 samples)
    buffer_duration_sec: float = 5.0   # 环形缓冲区长度

    # ---- 设备 ----
    input_device: Optional[str] = None     # 音频输入设备名 (None=默认)
    output_device: Optional[str] = None    # 音频输出设备名 (None=默认)

    # ---- 降噪参数 ----
    denoise_enabled: bool = True
    denoise_algorithm: str = "spectral_gate"  # "spectral_gate", "webrtc", "rnnoise"
    noise_reduction_db: float = 12.0     # 降噪强度 (dB)

    # ---- 回声消除 ----
    aec_enabled: bool = True
    aec_filter_length_ms: int = 100      # 回声消除滤波器长度

    # ---- VAD 参数 ----
    vad_enabled: bool = True
    vad_mode: int = 2                    # 0=低, 1=中, 2=高, 3=极高
    vad_silence_duration_ms: int = 800   # 静音判定阈值 (800ms)
    vad_speech_duration_ms: int = 200    # 语音触发最短长度

    @property
    def chunk_size(self) -> int:
        """单次读取采样点数"""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def buffer_size(self) -> int:
        """环形缓冲区采样点数"""
        return int(self.sample_rate * self.buffer_duration_sec)


# ---------------------------------------------------------------------------
# 管道配置
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """离线语音管道总配置"""

    # 子配置
    models: ModelPaths = field(default_factory=ModelPaths)
    audio: AudioConfig = field(default_factory=AudioConfig)
    performance: PerformanceTargets = field(default_factory=PerformanceTargets)

    # ---- 功能开关 ----
    enable_kws: bool = True             # 启用唤醒词检测
    enable_denoise: bool = True         # 启用降噪
    enable_aec: bool = True             # 启用回声消除
    enable_vad: bool = True             # 启用语音活动检测
    enable_tts_feedback: bool = True    # 启用 TTS 语音反馈

    # ---- 唤醒词配置 ----
    wake_word: str = "小T小T"           # 自定义唤醒词
    wake_word_threshold: float = 0.7    # 唤醒词检测阈值 (0-1)

    # ---- ASR 配置 ----
    asr_num_threads: int = 2            # ASR 推理线程数
    asr_max_active_paths: int = 4       # 解码最大路径数

    # ---- 设备控制 ----
    device_config_file: str = "config/devices.yaml"

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PipelineConfig":
        models = ModelPaths(**data.get("models", {}))
        audio = AudioConfig(**data.get("audio", {}))
        perf = PerformanceTargets(**data.get("performance", {}))
        config = cls(models=models, audio=audio, performance=perf)

        for key in ["enable_kws", "enable_denoise", "enable_aec",
                     "enable_vad", "enable_tts_feedback",
                     "wake_word", "wake_word_threshold",
                     "asr_num_threads", "asr_max_active_paths",
                     "device_config_file"]:
            if key in data:
                setattr(config, key, data[key])

        return config


# ---------------------------------------------------------------------------
# 默认配置实例
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PipelineConfig()
