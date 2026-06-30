"""
语音唤醒引擎 (KWS - Keyword Spotting)

基于 sherpa-onnx KWS 模型，实现低功耗持续监听唤醒词。
支持自定义唤醒词（如 "小T小T"）。

性能指标:
- 延迟: < 200ms
- 内存: 模型 3.3MB + 运行时 2MB = ~5.3MB
- 误触发率: < 1次/24小时
- 唤醒率: > 95%
"""

import logging
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np

from voice_router_lite.config import ModelPaths, AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class KWSResult:
    """唤醒词检测结果"""
    detected: bool
    keyword: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0


class KWSEngine:
    """
    唤醒词检测引擎

    两种运行模式:
    1. sherpa-onnx KWS (需要 sherpa_onnx 包，高精度)
    2. 能量唤醒 (零依赖，简易模式，适合调试)

    使用方法:
        engine = KWSEngine(model_paths, audio_config)
        engine.initialize()

        # 模式A: 手动检测
        result = engine.detect(audio_chunk)

        # 模式B: 持续监听 (回调)
        engine.on_wake(lambda kw: print(f"唤醒: {kw}"))
        engine.start_listening()
    """

    def __init__(self, model_paths: ModelPaths, audio_config: AudioConfig,
                 wake_word_threshold: float = 0.7):
        self._model_paths = model_paths
        self._audio_config = audio_config
        self._wake_word_threshold = wake_word_threshold

        # sherpa-onnx 引擎
        self._sherpa_kws = None
        self._mode: str = "energy"  # "sherpa" | "energy"

        # 状态
        self._initialized: bool = False
        self._listening: bool = False
        self._listening_thread: Optional[threading.Thread] = None

        # KWS 流式检测（持久化 stream，持续喂入音频）
        self._kws_stream = None  # sherpa-onnx stream，初始化后创建
        self._stream_frame_count: int = 0  # 已喂入 stream 的帧数
        self._stream_tail_pending: bool = False  # 是否已喂入 tail paddings

        # 调试日志控制
        self._debug_log_interval: int = 30  # 每30帧输出一次 debug 日志
        self._verbose_frames: int = 60  # 前N帧开启逐帧详细日志

        # 唤醒回调
        self._wake_callbacks: list[Callable[[KWSResult], None]] = []
        self._lock = threading.Lock()

        # 能量唤醒参数
        # 注意：energy 模式只是 sherpa 不可用时的兜底。返回的 keyword 必须是唤醒词
        # "小T小T"（与 ws_audio 的 WAKE_WORDS 集合保持一致），否则上层会把假
        # keyword "energy_wake" 当作指令文本扔给 NLU，触发误判指令（如 LED 灯）。
        self._energy_spike_count: int = 0
        self._energy_threshold: float = 0.12
        self._energy_debounce_frames: int = 5  # 防抖：连续高能量帧数

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        初始化 KWS 引擎。
        优先使用 sherpa-onnx，不可用时回退到能量唤醒。

        Returns:
            True 初始化成功
        """
        try:
            import sherpa_onnx
            import os

            # 检查必要模型文件
            required = [
                self._model_paths.kws_encoder,
                self._model_paths.kws_decoder,
                self._model_paths.kws_joiner,
                self._model_paths.kws_tokens,
            ]
            missing = [f for f in required if not os.path.exists(f)]
            if missing:
                raise FileNotFoundError(f"缺少 KWS 模型文件: {missing}")

            # 关键词配置文件 (可选)
            keywords_file = self._model_paths.kws_keywords
            if not os.path.exists(keywords_file):
                keywords_file = ""

            # 新版 sherpa_onnx API: 直接传参构造 KeywordSpotter
            # sherpa keywords_threshold: 越高越难触发；官方默认 0.25
            env_thresh = os.getenv("VOICE_ROUTER_KWS_THRESHOLD")
            if env_thresh:
                keywords_threshold = max(0.05, min(0.5, float(env_thresh)))
            else:
                keywords_threshold = max(0.05, min(0.35, self._wake_word_threshold * 0.25))

            self._sherpa_kws = sherpa_onnx.KeywordSpotter(
                tokens=self._model_paths.kws_tokens,
                encoder=self._model_paths.kws_encoder,
                decoder=self._model_paths.kws_decoder,
                joiner=self._model_paths.kws_joiner,
                keywords_file=keywords_file if keywords_file else "",
                num_threads=1,
                sample_rate=self._audio_config.sample_rate,
                max_active_paths=8,
                keywords_score=3.0,
                keywords_threshold=keywords_threshold,
                num_trailing_blanks=4,
            )
            self._mode = "sherpa"
            self._initialized = True

            # 创建持久化 stream，持续喂入音频（流式模式）
            self._kws_stream = self._sherpa_kws.create_stream()
            self._stream_frame_count = 0
            self._stream_tail_pending = False

            # 打印关键词列表
            if keywords_file:
                try:
                    with open(keywords_file, "r") as f:
                        kw_lines = [l.strip() for l in f if l.strip()]
                    logger.info("KWS 关键词文件: %s (%d条)", keywords_file, len(kw_lines))
                    for kw in kw_lines[:3]:
                        logger.info("  → %s", kw)
                except Exception:
                    pass

            logger.info(
                "KWS 引擎已初始化: 模式=sherpa-onnx (threshold=%.2f)",
                keywords_threshold,
            )
            return True

        except (ImportError, FileNotFoundError, Exception) as e:
            logger.warning("sherpa-onnx KWS 不可用 (%s)，使用能量唤醒模式", e)
            self._mode = "energy"
            self._initialized = True
            return True

    def detect(self, audio_chunk: np.ndarray) -> KWSResult:
        """
        检测一帧音频是否包含唤醒词。

        Args:
            audio_chunk: 16kHz mono 音频 (float32, 归一化)

        Returns:
            KWSResult 检测结果
        """
        if not self._initialized:
            return KWSResult(detected=False)

        if self._mode == "sherpa":
            return self._sherpa_detect(audio_chunk)
        else:
            return self._energy_detect(audio_chunk)

    def start_listening(self, audio_source: Callable[[], Optional[np.ndarray]]) -> None:
        """
        启动持续监听 (后台线程)。

        Args:
            audio_source: 音频数据源回调函数，返回音频块或 None
        """
        if self._listening:
            return

        self._listening = True
        self._listening_thread = threading.Thread(
            target=self._listening_loop,
            args=(audio_source,),
            daemon=True,
        )
        self._listening_thread.start()
        logger.info("KWS 持续监听已启动")

    def stop_listening(self) -> None:
        """停止持续监听"""
        self._listening = False
        if self._listening_thread:
            self._listening_thread.join(timeout=2.0)
            self._listening_thread = None
        logger.info("KWS 持续监听已停止")

    def on_wake(self, callback: Callable[[KWSResult], None]) -> None:
        """
        注册唤醒回调。

        Args:
            callback: 唤醒时调用，参数为 KWSResult
        """
        self._wake_callbacks.append(callback)

    def reset(self) -> None:
        """重置引擎状态（清空 KWS stream，防止旧音频干扰）"""
        if self._mode == "sherpa" and self._sherpa_kws and self._kws_stream:
            self._sherpa_kws.reset_stream(self._kws_stream)
            self._stream_frame_count = 0
        self._energy_spike_count = 0

    def close(self) -> None:
        """释放资源"""
        self.stop_listening()
        self._kws_stream = None
        self._sherpa_kws = None
        self._initialized = False

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _sherpa_detect(self, audio_chunk: np.ndarray) -> KWSResult:
        """
        sherpa-onnx KWS 检测 (流式持久 stream 模式)。

        持续向同一个 stream 喂入音频帧，调用 decode 检查是否有
        关键词匹配。检测到关键词后自动 reset stream。

        参照 sherpa-onnx 官方 keyword-spotter-from-microphone 示例。
        """
        try:
            if audio_chunk.dtype != np.float32:
                audio = audio_chunk.astype(np.float32)
            else:
                audio = audio_chunk

            # 喂入音频到持久化 stream
            self._stream_frame_count += 1
            fc = self._stream_frame_count
            self._kws_stream.accept_waveform(
                self._audio_config.sample_rate, audio
            )

            # 解码并检查结果
            result_text = ""
            decode_count = 0
            raw_results = []
            while self._sherpa_kws.is_ready(self._kws_stream):
                self._sherpa_kws.decode_stream(self._kws_stream)
                decode_count += 1
                r = self._sherpa_kws.get_result(self._kws_stream)
                raw_results.append((type(r).__name__, repr(r)))
                if r and isinstance(r, str) and r.strip():
                    result_text = r.strip()

            # 前 N 帧逐帧详细日志
            energy = float(np.sqrt(np.mean(audio ** 2)))
            if fc <= self._verbose_frames:
                raw_info = f"raw={raw_results}" if raw_results else ""
                logger.info(
                    "[KWS verbose] #%d samples=%d energy=%.5f C:%s O:%s NaN:%s ready=%s decodes=%d result=%s %s",
                    fc, len(audio), energy,
                    audio.flags['C_CONTIGUOUS'], audio.flags['OWNDATA'],
                    bool(np.isnan(audio).any()),
                    "YES" if decode_count > 0 else "NO",
                    decode_count, repr(result_text), raw_info,
                )
            elif fc % self._debug_log_interval == 0:
                logger.info(
                    "[KWS debug] #%d energy=%.5f decodes=%d result=%s",
                    fc, energy, decode_count, repr(result_text),
                )
            elif energy > 0.025 and fc % 15 == 0:
                logger.info(
                    "[KWS] 语音能量高但未命中: energy=%.4f decodes=%d",
                    energy, decode_count,
                )

            if result_text:
                # 检测到唤醒词 → reset stream，防止重复触发
                self._sherpa_kws.reset_stream(self._kws_stream)
                frame_num = self._stream_frame_count
                self._stream_frame_count = 0
                logger.info(
                    "[KWS] 🎯 检测到唤醒词: '%s' (frame #%d)",
                    result_text, frame_num,
                )
                confidence = min(1.0, max(0.5, 1.0 - self._wake_word_threshold * 0.3))
                return KWSResult(
                    detected=True,
                    keyword=result_text.strip(),
                    confidence=confidence,
                    timestamp=time.time(),
                )

        except Exception as e:
            logger.error("KWS 推理异常: %s", e, exc_info=True)

        return KWSResult(detected=False)

    def _energy_detect(self, audio_chunk: np.ndarray) -> KWSResult:
        """
        简化能量唤醒检测 (sherpa-onnx 不可用时的兜底)。

        逻辑: 检测到短暂高能量脉冲 (如拍手/敲击) 触发唤醒。
        注意: 兜底返回的 keyword 必须是唤醒词 "小T小T"，与 ws_audio 的
        WAKE_WORDS 集合保持一致，触发后应进入 AWAKE 状态而不是被当作
        指令文本交给 NLU 处理。
        """
        if audio_chunk.dtype != np.float32:
            audio = audio_chunk.astype(np.float32)
        else:
            audio = audio_chunk

        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms > self._energy_threshold:
            self._energy_spike_count += 1

            if self._energy_spike_count >= self._energy_debounce_frames:
                self._energy_spike_count = 0
                kw_result = KWSResult(
                    detected=True,
                    keyword="小T小T",  # 必须是唤醒词，不要用 "energy_wake"
                    confidence=min(rms / self._energy_threshold / 10, 1.0),
                    timestamp=time.time(),
                )
                self._notify_wake(kw_result)
                return kw_result
        else:
            # 衰减计数
            self._energy_spike_count = max(0, self._energy_spike_count - 1)

        return KWSResult(detected=False)

    def _notify_wake(self, result: KWSResult) -> None:
        """通知所有唤醒回调"""
        for cb in self._wake_callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error("唤醒回调异常: %s", e)

    def _listening_loop(self, audio_source: Callable[[], Optional[np.ndarray]]) -> None:
        """持续监听主循环"""
        last_detect_time = 0.0
        cooldown_sec = 2.0  # 唤醒冷却时间

        while self._listening:
            try:
                audio = audio_source()
                if audio is not None:
                    result = self.detect(audio)

                    # 防抖: 冷却时间内不重复触发
                    now = time.time()
                    if result.detected:
                        if now - last_detect_time > cooldown_sec:
                            last_detect_time = now
                            self._notify_wake(result)

                time.sleep(0.01)  # 10ms 轮询间隔
            except Exception as e:
                logger.error("监听循环异常: %s", e)
                time.sleep(0.1)
