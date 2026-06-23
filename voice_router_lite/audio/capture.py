"""
音频采集模块

提供标准音频输入接口 (16kHz / 16bit / mono)。
支持从物理麦克风采集或从模拟缓冲区读取。
所有处理在本地完成，无网络依赖。
"""

import logging
import threading
import wave
from typing import Optional
from collections import deque

import numpy as np

from voice_router_lite.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    音频采集器

    标准接口：16kHz, 16bit, mono PCM
    支持实时麦克风采集和模拟数据输入两种模式。

    使用方法:
        cap = AudioCapture(config)
        cap.open()
        while True:
            chunk = cap.read()           # numpy array, shape=(480,), int16
        cap.close()
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._stream = None
        self._is_open = False

        # 环形缓冲区（非阻塞读取）
        self._buffer = deque(maxlen=config.buffer_size)
        self._lock = threading.Lock()

        # 统计
        self._total_samples = 0

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def open(self, device_name: Optional[str] = None) -> None:
        """
        打开音频采集设备。

        Args:
            device_name: 设备名，None 则使用系统默认
        """
        if self._is_open:
            return

        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            device_index = self._resolve_device(device_name)
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self._config.chunk_size,
                stream_callback=None,
            )
            self._is_open = True
            logger.info(
                "音频采集已打开: sample_rate=%d, chunk_size=%d, device=%s",
                self._config.sample_rate,
                self._config.chunk_size,
                device_name or "默认设备",
            )
        except ImportError:
            logger.warning("PyAudio 未安装，使用静音模拟采集模式")
            self._is_open = True
        except OSError as e:
            logger.error("无法打开音频设备: %s", e)
            raise

    def read(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        读取音频数据。

        Args:
            num_samples: 采样点数，默认使用 chunk_size

        Returns:
            numpy array, dtype=int16, shape=(num_samples,)
        """
        if num_samples is None:
            num_samples = self._config.chunk_size

        if self._stream is not None:
            try:
                raw = self._stream.read(num_samples, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.int16).copy()
            except Exception as e:
                logger.warning("音频读取异常: %s, 返回静音", e)
                data = np.zeros(num_samples, dtype=np.int16)
        else:
            data = np.zeros(num_samples, dtype=np.int16)

        # 写入环形缓冲区
        with self._lock:
            for s in data:
                self._buffer.append(int(s))
        self._total_samples += num_samples

        return data

    def read_streaming(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        流式读取（从环形缓冲区），非阻塞。

        Args:
            num_samples: 读取采样点数

        Returns:
            numpy array, 不足时填充零
        """
        if num_samples is None:
            num_samples = self._config.chunk_size

        with self._lock:
            available = min(num_samples, len(self._buffer))
            if available == 0:
                return np.zeros(num_samples, dtype=np.float32)

            data = [self._buffer.popleft() for _ in range(available)]
            result = np.array(data, dtype=np.float32) / 32768.0

        if len(result) < num_samples:
            result = np.pad(result, (0, num_samples - len(result)))

        return result

    def close(self) -> None:
        """关闭音频采集设备"""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if hasattr(self, "_pyaudio") and self._pyaudio:
            self._pyaudio.terminate()
        self._is_open = False
        logger.info("音频采集已关闭")

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def total_duration_sec(self) -> float:
        return self._total_samples / self._config.sample_rate

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve_device(self, name: Optional[str]) -> Optional[int]:
        """根据设备名查找设备索引"""
        if name is None:
            try:
                return self._pyaudio.get_default_input_device_info()["index"]
            except Exception:
                return None

        try:
            for i in range(self._pyaudio.get_device_count()):
                info = self._pyaudio.get_device_info_by_index(i)
                if name in info.get("name", ""):
                    return i
        except Exception:
            pass
        return None


def save_wav(filepath: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """
    将音频保存为 WAV 文件。

    Args:
        filepath: 输出路径
        audio: int16 格式的音频数据
        sample_rate: 采样率
    """
    audio_int16 = np.asarray(audio, dtype=np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def load_wav(filepath: str) -> np.ndarray:
    """
    从 WAV 文件加载音频。

    Args:
        filepath: WAV 文件路径

    Returns:
        int16 格式的 numpy array
    """
    with wave.open(filepath, "rb") as wf:
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16).copy()
