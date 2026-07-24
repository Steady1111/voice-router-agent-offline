"""
全局配置管理

统一管理模型路径、音频参数、设备配置、性能指标。
所有配置项均有默认值，支持从 YAML/JSON 文件加载。
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 性能指标约束
# ---------------------------------------------------------------------------
@dataclass
class PerformanceTargets:
    """离线处理性能指标（与 128MB 路由器部署预算对齐）"""

    # 延迟目标（毫秒）
    kws_latency_ms: int = 200           # 唤醒词检测延迟 < 200ms
    asr_latency_ms: int = 1500          # ASR 端到端延迟 < 1.5s
    nlu_latency_ms: int = 100           # NLU 推理延迟 < 100ms
    tts_latency_ms: int = 300           # TTS 播放延迟 < 300ms
    total_latency_ms: int = 2000        # 端到端总延迟 < 2s

    # 设备总内存预算
    device_total_memory_mb: int = 128   # 目标路由器物理内存
    system_reserved_memory_mb: int = 60 # OpenWrt + Python 运行时估算

    # 引擎运行时内存估算（非 psutil RSS，见 estimate_engine_memory）
    peak_engine_memory_mb: int = 24     # 默认占位；由 estimate_engine_memory 覆写
    asr_model_memory_mb: int = 25       # ASR 子进程 / Web 常驻 sherpa 工作区峰值
    kws_model_memory_mb: int = 8        # KWS 模型 + 运行时
    nlu_model_memory_mb: int = 11       # NLU fp32 模型 + 运行时
    nlu_model_memory_mb_int8: int = 4   # NLU INT8 运行时
    audio_buffer_memory_mb: int = 5     # 环形缓冲 + 解码工作区（含 2s@16kHz PCM）

    # 模型文件体积（磁盘）
    asr_model_size_mb: int = 24         # zipformer zh 14M mobile 约 24MB
    kws_model_size_mb: int = 5          # KWS 模型文件 < 5MB
    nlu_model_size_mb: int = 11         # NLU fp32；INT8 量化目标 3MB
    tts_clips_total_mb: int = 5         # TTS 预录制音频 < 5MB

    # 兼容旧字段名
    total_memory_mb: int = 24           # = peak_engine_memory_mb


# ---------------------------------------------------------------------------
# 模型路径配置
# ---------------------------------------------------------------------------
@dataclass
class ModelPaths:
    """模型文件路径"""

    base_dir: str = "models"

    # KWS 唤醒词模型 (Transducer 架构: encoder + decoder + joiner)
    kws_encoder: str = "models/kws/encoder.onnx"
    kws_decoder: str = "models/kws/decoder.onnx"
    kws_joiner: str = "models/kws/joiner.onnx"
    kws_tokens: str = "models/kws/tokens.txt"
    kws_keywords: str = "models/kws/keywords.txt"

    # ASR 模型 (sherpa-onnx Zipformer)
    asr_encoder: str = "models/asr/encoder.onnx"
    asr_decoder: str = "models/asr/decoder.onnx"
    asr_joiner: str = "models/asr/joiner.onnx"
    asr_tokens: str = "models/asr/tokens.txt"

    # NLU 模型（intent + slot 共用同一 ONNX 文件）
    nlu_intent_model: str = "models/nlu/intent_model.onnx"
    nlu_intent_model_int8: str = "models/nlu/intent_model.int8.onnx"
    nlu_vocab: str = "models/nlu/vocab.json"
    nlu_intent_labels: str = "models/nlu/intent_labels.json"
    nlu_slot_labels: str = "models/nlu/slot_labels.json"

    # TTS 预录制音频
    tts_clips_dir: str = "audio_clips"
    tts_index: str = "audio_clips/index.json"

    def resolve_nlu_model(self, prefer_int8: bool = False) -> str:
        """路由器部署优先使用 INT8 量化模型。"""
        if prefer_int8 and os.path.exists(self.nlu_intent_model_int8):
            return self.nlu_intent_model_int8
        return self.nlu_intent_model

    def validate(self) -> List[str]:
        """检查必需文件是否存在，返回缺失文件列表"""
        required = [
            self.kws_encoder, self.kws_decoder, self.kws_joiner, self.kws_tokens,
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

    # ---- 部署模式 ----
    deployment_mode: str = "dev"        # "dev" | "router"

    # ---- 路由器内存优化（128MB 量产）----
    model_serial_exclusive: bool = False   # KWS/ASR 互斥，不同时驻留
    asr_subprocess: bool = False           # ASR 在独立子进程，退出即释放内存
    prefer_int8_nlu: bool = False          # 优先加载 NLU INT8 模型

    # ---- 功能开关 ----
    enable_kws: bool = True             # 启用唤醒词检测
    enable_denoise: bool = True         # 启用降噪
    enable_aec: bool = True             # 启用回声消除
    enable_vad: bool = True             # 启用语音活动检测
    enable_tts_feedback: bool = True    # 启用 TTS 语音反馈

    # ---- 唤醒词配置 ----
    wake_word: str = "小T小T"           # 自定义唤醒词
    wake_word_threshold: float = 0.5    # 唤醒词检测阈值 (0-1)，映射 sherpa ~0.125

    # ---- ASR 配置 ----
    asr_num_threads: int = 2            # ASR 推理线程数
    asr_max_active_paths: int = 4       # 解码最大路径数
    asr_lazy_load: bool = True          # 按需加载 ASR，降低待机内存
    asr_unload_after_use: bool = True   # 识别完成后卸载 ASR

    # ---- NLU 配置 ----
    nlu_confidence_threshold: float = 0.5  # 低于此值回退规则引擎

    # ---- 资源限制 (Linux/OpenWrt) ----
    enable_cgroups: bool = True
    cgroup_cpu_quota_pct: int = 30
    cgroup_memory_mb: int = 80

    # ---- 设备控制 ----
    device_config_file: str = "config/devices.yaml"
    use_openwrt_ubus: bool = True       # 路由器上优先走 ubus

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

        for key in ["deployment_mode", "model_serial_exclusive", "asr_subprocess",
                     "prefer_int8_nlu",
                     "enable_kws", "enable_denoise", "enable_aec",
                     "enable_vad", "enable_tts_feedback",
                     "wake_word", "wake_word_threshold",
                     "asr_num_threads", "asr_max_active_paths",
                     "asr_lazy_load", "asr_unload_after_use",
                     "nlu_confidence_threshold",
                     "enable_cgroups", "cgroup_cpu_quota_pct", "cgroup_memory_mb",
                     "device_config_file", "use_openwrt_ubus"]:
            if key in data:
                setattr(config, key, data[key])

        return config


@dataclass
class EngineMemoryEstimate:
    """离线引擎 / 全机内存快照（监控面板用）。"""

    profile: str
    nlu_variant: str
    kws_mb: float
    nlu_mb: float
    asr_mb: float
    buffer_mb: float
    current_mb: float
    engine_standby_mb: float
    engine_peak_mb: float
    device_standby_mb: float
    device_peak_mb: float
    # 运行时动态字段（tick / 状态机填充）
    engine_current_mb: float = 0.0
    device_current_mb: float = 0.0
    device_total_mb: float = 128.0
    phase: str = "standby"
    mode: str = "router_sim"
    scope: str = "router_sim"
    rss_mb: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.engine_current_mb <= 0:
            self.engine_current_mb = self.current_mb
        if self.device_current_mb <= 0:
            self.device_current_mb = (
                self.device_standby_mb
                if self.phase == "standby"
                else self.device_peak_mb
            )

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "scope": self.scope,
            "mode": self.mode,
            "phase": self.phase,
            "nlu_variant": self.nlu_variant,
            "components_mb": {
                "kws": round(self.kws_mb, 1),
                "nlu": round(self.nlu_mb, 1),
                "asr": round(self.asr_mb, 1),
                "buffer": round(self.buffer_mb, 1),
            },
            "engine_current_mb": round(self.engine_current_mb, 1),
            "device_current_mb": round(self.device_current_mb, 1),
            "device_total_mb": round(self.device_total_mb, 1),
            "current_mb": round(self.device_current_mb, 1),
            "engine_standby_mb": round(self.engine_standby_mb, 1),
            "engine_peak_mb": round(self.engine_peak_mb, 1),
            "device_standby_mb": round(self.device_standby_mb, 1),
            "device_peak_mb": round(self.device_peak_mb, 1),
            "rss_mb": round(self.rss_mb, 1) if self.rss_mb is not None else None,
            "note": self.note or "Mac 预研：按 OpenWrt 量产状态机模拟（待机↔ASR 峰值）",
        }


def resolve_nlu_runtime_mb(
    cfg: PipelineConfig,
    *,
    prefer_int8_nlu: bool | None = None,
) -> tuple[float, str]:
    """返回 NLU 运行时估算 (MB) 与变体标识。"""
    perf = cfg.performance
    use_int8 = cfg.prefer_int8_nlu if prefer_int8_nlu is None else prefer_int8_nlu
    if use_int8 and os.path.exists(cfg.models.nlu_intent_model_int8):
        return float(perf.nlu_model_memory_mb_int8), "int8"
    return float(perf.nlu_model_memory_mb), "fp32"


def estimate_engine_memory(
    cfg: PipelineConfig,
    *,
    profile: str = "web",
    web_asr_loaded: bool = False,
    prefer_int8_nlu: bool | None = None,
    phase: str = "standby",
) -> EngineMemoryEstimate:
    """按部署形态与运行阶段估算内存（与方案文档 §2.1 对齐）。

    profile:
      - web: Mac 预研 Web 服务（无 KWS；ASR 可选常驻）— 仅静态估算用
      - router: OpenWrt daemon（KWS+NLU 待机；ASR 阶段 KWS 卸载）
    phase:
      - standby: KWS + NLU + Buf
      - asr_active: NLU + ASR + Buf（互斥架构）
    """
    perf = cfg.performance
    nlu_mb, nlu_variant = resolve_nlu_runtime_mb(cfg, prefer_int8_nlu=prefer_int8_nlu)
    buffer_mb = float(perf.audio_buffer_memory_mb)
    kws_mb = float(perf.kws_model_memory_mb) if cfg.enable_kws else 0.0
    asr_model_mb = float(perf.asr_model_memory_mb)
    sys_mb = float(perf.system_reserved_memory_mb)
    asr_phase = phase == "asr_active"

    if profile == "web":
        kws_active = 0.0
        asr_active = asr_model_mb if web_asr_loaded else 0.0
        engine_current = nlu_mb + asr_active + buffer_mb
        engine_standby = nlu_mb + buffer_mb
        engine_peak = nlu_mb + asr_model_mb + buffer_mb
    elif cfg.model_serial_exclusive:
        engine_standby = kws_mb + nlu_mb + buffer_mb
        engine_peak = nlu_mb + asr_model_mb + buffer_mb
        if asr_phase:
            kws_active = 0.0
            asr_active = asr_model_mb
            engine_current = engine_peak
        else:
            kws_active = kws_mb
            asr_active = 0.0
            engine_current = engine_standby
    else:
        kws_active = kws_mb
        asr_active = 0.0
        engine_current = kws_mb + nlu_mb + buffer_mb
        engine_standby = engine_current
        engine_peak = engine_current

    device_standby = sys_mb + engine_standby
    device_peak = sys_mb + engine_peak
    device_current = device_peak if asr_phase else device_standby

    return EngineMemoryEstimate(
        profile=profile,
        nlu_variant=nlu_variant,
        kws_mb=kws_active,
        nlu_mb=nlu_mb,
        asr_mb=asr_active,
        buffer_mb=buffer_mb,
        current_mb=engine_current,
        engine_current_mb=engine_current,
        engine_standby_mb=engine_standby,
        engine_peak_mb=engine_peak,
        device_standby_mb=device_standby,
        device_peak_mb=device_peak,
        device_current_mb=device_current,
        device_total_mb=float(perf.device_total_memory_mb),
        phase=phase if profile == "router" else ("asr_active" if web_asr_loaded else "standby"),
        mode="static_estimate",
        scope="static_estimate",
    )


def router_default_config() -> PipelineConfig:
    """路由器量产默认配置（128MB OpenWrt，KWS/ASR 互斥）。"""
    cfg = PipelineConfig()
    cfg.deployment_mode = "router"
    cfg.model_serial_exclusive = True
    cfg.asr_subprocess = True
    cfg.prefer_int8_nlu = True
    cfg.asr_lazy_load = True
    cfg.asr_unload_after_use = True
    cfg.enable_cgroups = True
    cfg.cgroup_cpu_quota_pct = 30
    cfg.cgroup_memory_mb = 80
    cfg.use_openwrt_ubus = True
    cfg.audio.buffer_duration_sec = 2.0
    est = estimate_engine_memory(cfg, profile="router", web_asr_loaded=False)
    cfg.performance.peak_engine_memory_mb = int(est.engine_peak_mb)
    cfg.performance.total_memory_mb = cfg.performance.peak_engine_memory_mb
    return cfg


# ---------------------------------------------------------------------------
# 默认配置实例
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PipelineConfig()
