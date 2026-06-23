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

    def __init__(self, model_paths: ModelPaths, audio_config: AudioConfig):
        self._model_paths = model_paths
        self._audio_config = audio_config

        # sherpa-onnx 引擎
        self._sherpa_kws = None
        self._mode: str = "energy"  # "sherpa" | "energy"

        # 状态
        self._initialized: bool = False
        self._listening: bool = False
        self._listening_thread: Optional[threading.Thread] = None

        # 音频缓冲
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_duration_ms: int = 1500  # 1.5秒检测窗口

        # 唤醒回调
        self._wake_callbacks: list[Callable[[KWSResult], None]] = []
        self._lock = threading.Lock()

        # 能量唤醒参数
        self._energy_spike_count: int = 0
        self._energy_threshold: float = 0.05
        self._energy_debounce_frames: int = 20  # 防抖

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

            # 检查模型文件
            import os
            if not os.path.exists(self._model_paths.kws_model):
                logger.warning("KWS 模型文件不存在: %s", self._model_paths.kws_model)
                raise FileNotFoundError(self._model_paths.kws_model)

            self._sherpa_kws = sherpa_onnx.KeywordSpotter(
                model=self._model_paths.kws_model,
                tokens=self._model_paths.kws_tokens,
                num_threads=1,
                sample_rate=self._audio_config.sample_rate,
            )
            self._mode = "sherpa"
            self._initialized = True
            logger.info("KWS 引擎已初始化: 模式=sherpa-onnx")
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

        # 加入缓冲
        with self._lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

            # 保持缓冲窗口长度
            max_samples = int(
                self._audio_config.sample_rate * self._buffer_duration_ms / 1000
            )
            if len(self._audio_buffer) > max_samples:
                self._audio_buffer = self._audio_buffer[-max_samples:]

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
        """重置引擎状态"""
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)
            self._energy_spike_count = 0

    def close(self) -> None:
        """释放资源"""
        self.stop_listening()
        self._sherpa_kws = None
        self._initialized = False

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _sherpa_detect(self, audio_chunk: np.ndarray) -> KWSResult:
        """sherpa-onnx KWS 检测"""
        try:
            # 确保 float32
            if audio_chunk.dtype != np.float32:
                audio = audio_chunk.astype(np.float32)
            else:
                audio = audio_chunk

            self._sherpa_kws.accept_waveform(self._audio_config.sample_rate, audio)

            while self._sherpa_kws.is_ready():
                self._sherpa_kws.decode()

            result_text = self._sherpa_kws.get_result()
            if result_text and result_text.keyword:
                kw_result = KWSResult(
                    detected=True,
                    keyword=result_text.keyword,
                    confidence=getattr(result_text, "confidence", 0.8),
                    timestamp=time.time(),
                )
                self._notify_wake(kw_result)
                return kw_result

        except Exception as e:
            logger.debug("KWS 检测异常: %s", e)

        return KWSResult(detected=False)

    def _energy_detect(self, audio_chunk: np.ndarray) -> KWSResult:
        """
        简化能量唤醒检测。

        逻辑: 检测到短暂高能量脉冲 (如拍手/敲击) 触发唤醒。
        适合调试阶段，正式部署应使用 sherpa-onnx KWS。
        """
        if audio_chunk.dtype != np.float32:
            audio = audio_chunk.astype(np.float32)
        else:
            audio = audio_chunk

        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms > self._energy_threshold:
            self._energy_spike_count += 1

            if self._energy_spike_count >= 3:  # 连续3帧高能量
                self._energy_spike_count = 0
                kw_result = KWSResult(
                    detected=True,
                    keyword="energy_wake",
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
