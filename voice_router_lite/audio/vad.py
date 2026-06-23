"""
语音活动检测 (VAD)

基于能量的简易 VAD + WebRTC VAD 双模式。
用于判断音频帧是否包含人声，减少无效 ASR 调用和降低误触发。
"""

import logging
from typing import Tuple

import numpy as np

from voice_router_lite.config import AudioConfig

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    语音活动检测器

    VAD 模式:
    - "energy":  基于 RMS 能量阈值 (零依赖，最轻量)
    - "webrtc":  基于 WebRTC VAD (需要 webrtc 库，更准确)

    使用方法:
        vad = VoiceActivityDetector(config)
        is_speech = vad.is_speech(audio_chunk)
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._mode = "webrtc"  # 默认使用 WebRTC
        self._webrtc_vad = None

        # 能量 VAD 参数
        self._energy_threshold: float = 0.01        # RMS 阈值 (归一化)
        self._silence_frames: int = 0               # 连续静音帧数
        self._speech_frames: int = 0                # 连续语音帧数
        self._is_speaking: bool = False

        # 自适应阈值
        self._noise_floor: float = 0.001
        self._adaptation_rate: float = 0.05

        self._init_vad()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        判断音频帧是否包含语音。

        Args:
            audio_chunk: 16kHz, mono 音频数据 (int16 或 float32)

        Returns:
            True 包含人声，False 静音/噪音
        """
        if self._mode == "webrtc" and self._webrtc_vad is not None:
            return self._webrtc_is_speech(audio_chunk)
        return self._energy_is_speech(audio_chunk)

    def process_frame(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
        """
        处理音频帧，返回 (is_speech, speech_just_ended)。

        Args:
            audio_chunk: 音频数据

        Returns:
            (当前帧是否语音, 语音段是否刚结束)
        """
        is_speech = self.is_speech(audio_chunk)

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1
            if not self._is_speaking and self._speech_frames >= 5:
                self._is_speaking = True
            return True, False
        else:
            self._speech_frames = 0
            self._silence_frames += 1

            silence_threshold = int(
                self._config.vad_silence_duration_ms
                / self._config.chunk_duration_ms
            )

            if self._is_speaking and self._silence_frames >= silence_threshold:
                self._is_speaking = False
                self._silence_frames = 0
                return False, True  # 语音段结束

            return False, False

    def update_noise_floor(self, audio_chunk: np.ndarray) -> None:
        """自适应更新噪声地板"""
        rms = self._compute_rms(audio_chunk)
        if rms < self._noise_floor * 3:
            self._noise_floor = (
                (1 - self._adaptation_rate) * self._noise_floor
                + self._adaptation_rate * rms
            )

    def reset(self) -> None:
        """重置 VAD 状态"""
        self._silence_frames = 0
        self._speech_frames = 0
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _init_vad(self) -> None:
        """初始化 VAD 引擎"""
        try:
            import webrtcvad
            self._webrtc_vad = webrtcvad.Vad(self._config.vad_mode)
            self._mode = "webrtc"
            logger.info("VAD 引擎: WebRTC (mode=%d)", self._config.vad_mode)
        except ImportError:
            self._mode = "energy"
            logger.info("VAD 引擎: 能量检测 (webrtcvad 未安装)")

    def _webrtc_is_speech(self, audio_chunk: np.ndarray) -> bool:
        """WebRTC VAD 检测"""
        # 确保是 16bit PCM
        if audio_chunk.dtype == np.float32:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = np.asarray(audio_chunk, dtype=np.int16)

        # WebRTC VAD 需要 10/20/30ms 帧
        frame_size = self._config.chunk_size
        if len(audio_int16) < frame_size:
            audio_int16 = np.pad(audio_int16, (0, frame_size - len(audio_int16)))

        try:
            return self._webrtc_vad.is_speech(
                audio_int16[:frame_size].tobytes(),
                self._config.sample_rate,
            )
        except Exception:
            return self._energy_is_speech(audio_chunk)

    def _energy_is_speech(self, audio_chunk: np.ndarray) -> bool:
        """能量 VAD"""
        rms = self._compute_rms(audio_chunk)
        threshold = max(self._noise_floor * 3, self._energy_threshold)
        return rms > threshold

    @staticmethod
    def _compute_rms(audio: np.ndarray) -> float:
        """计算均方根能量"""
        if audio.dtype == np.float32:
            return float(np.sqrt(np.mean(audio ** 2)))
        audio_f32 = audio.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(audio_f32 ** 2)))
