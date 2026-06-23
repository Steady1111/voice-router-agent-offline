"""
音频播放模块

提供标准音频输出接口，用于 TTS 语音反馈播放。
支持 I2S DAC / USB 声卡 / 板载音频输出。
"""

import logging
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from voice_router_lite.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    音频播放器

    标准接口：16kHz, 16bit, mono PCM 输出。

    使用方法:
        player = AudioPlayer(config)
        player.open()
        player.play(audio_data)        # 阻塞播放
        player.play_async(audio_data)  # 异步播放
        player.close()
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._stream = None
        self._is_open = False
        self._lock = threading.Lock()
        self._play_thread: Optional[threading.Thread] = None

    def open(self, device_name: Optional[str] = None) -> None:
        """打开音频播放设备"""
        if self._is_open:
            return

        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            device_index = self._resolve_device(device_name) if device_name else None
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                output=True,
                output_device_index=device_index,
            )
            self._is_open = True
            logger.info(
                "音频播放已打开: sample_rate=%d, device=%s",
                self._config.sample_rate,
                device_name or "默认设备",
            )
        except ImportError:
            logger.warning("PyAudio 未安装，播放静音模拟")
            self._is_open = True
        except OSError as e:
            logger.error("无法打开音频播放设备: %s", e)
            raise

    def play(self, audio: np.ndarray, blocking: bool = True) -> None:
        """
        播放音频数据。

        Args:
            audio: int16 格式音频数据，16kHz mono
            blocking: True 阻塞直到播放完成
        """
        if not self._is_open:
            logger.warning("音频播放器未打开")
            return

        audio_int16 = np.asarray(audio, dtype=np.int16)

        if blocking:
            self._play_blocking(audio_int16)
        else:
            self._play_thread = threading.Thread(
                target=self._play_blocking,
                args=(audio_int16,),
                daemon=True,
            )
            self._play_thread.start()

    def play_async(self, audio: np.ndarray) -> None:
        """异步播放（非阻塞）"""
        self.play(audio, blocking=False)

    def play_wav(self, filepath: str, blocking: bool = True) -> None:
        """播放 WAV 文件"""
        audio = self._load_wav(filepath)
        if audio is not None:
            self.play(audio, blocking=blocking)

    def play_tone(self, frequency: float = 440.0,
                  duration_ms: int = 200,
                  blocking: bool = True) -> None:
        """
        播放提示音（蜂鸣）。

        Args:
            frequency: 频率 Hz
            duration_ms: 时长 ms
        """
        num_samples = int(self._config.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        tone = (np.sin(2 * np.pi * frequency * t) * 16384).astype(np.int16)

        # 添加渐入渐出
        fade_len = min(200, num_samples // 4)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        tone[:fade_len] = (tone[:fade_len] * fade_in).astype(np.int16)
        tone[-fade_len:] = (tone[-fade_len:] * fade_out).astype(np.int16)

        self.play(tone, blocking=blocking)

    def is_playing(self) -> bool:
        """检查是否正在播放"""
        return self._play_thread is not None and self._play_thread.is_alive()

    def wait_done(self) -> None:
        """等待当前播放完成"""
        if self._play_thread:
            self._play_thread.join()

    def close(self) -> None:
        """关闭播放设备"""
        self.wait_done()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if hasattr(self, "_pyaudio") and self._pyaudio:
            self._pyaudio.terminate()
        self._is_open = False
        logger.info("音频播放已关闭")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _play_blocking(self, audio: np.ndarray) -> None:
        if self._stream is None:
            return
        try:
            chunk_size = self._config.chunk_size
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                self._stream.write(chunk.tobytes())
        except Exception as e:
            logger.warning("音频播放异常: %s", e)

    @staticmethod
    def _load_wav(filepath: str) -> Optional[np.ndarray]:
        try:
            with wave.open(filepath, "rb") as wf:
                data = wf.readframes(wf.getnframes())
            return np.frombuffer(data, dtype=np.int16).copy()
        except Exception as e:
            logger.error("WAV 加载失败 %s: %s", filepath, e)
            return None

    def _resolve_device(self, name: str) -> Optional[int]:
        try:
            for i in range(self._pyaudio.get_device_count()):
                info = self._pyaudio.get_device_info_by_index(i)
                if name in info.get("name", ""):
                    return i
        except Exception:
            pass
        return None
