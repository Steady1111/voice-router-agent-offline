"""
降噪与回声消除模块

处理嘈杂环境下的降噪与回声消除逻辑。
提供频谱门降噪、回声消除、自动增益控制。
"""

import logging
from typing import Optional

import numpy as np

from voice_router_lite.config import AudioConfig

logger = logging.getLogger(__name__)


class NoiseReducer:
    """
    降噪处理器

    算法选型:
    - "spectral_gate": 频谱门降噪 (无依赖，CPU轻量，推荐)
    - "wiener": 维纳滤波
    - "none": 不降噪

    性能指标:
    - 延迟: < 5ms per frame (30ms chunk)
    - 降噪量: 10-20dB (可配置)

    使用方法:
        nr = NoiseReducer(config)
        clean_audio = nr.process(noisy_audio)
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._algorithm = config.denoise_algorithm
        self._noise_reduction_db = config.noise_reduction_db

        # 频谱门参数
        self._fft_size = 512
        self._hop_size = 256
        self._window = np.hanning(self._fft_size)
        self._noise_spectrum: Optional[np.ndarray] = None
        self._noise_spectrum_smooth: Optional[np.ndarray] = None
        self._smoothing_factor: float = 0.9

        # 噪声估计
        self._noise_frames_count: int = 0
        self._calibration_frames: int = 30   # 前30帧用于噪声校准

        # 统计
        self._frames_processed: int = 0

        logger.info("降噪器已初始化: algorithm=%s, nr_db=%.1f",
                     self._algorithm, self._noise_reduction_db)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        对音频进行降噪处理。

        Args:
            audio: 输入音频 (float32, 归一化到 [-1, 1])

        Returns:
            降噪后的音频 (float32, 同形状)
        """
        if not self._config.denoise_enabled or self._algorithm == "none":
            return audio

        self._frames_processed += 1

        if self._algorithm == "spectral_gate":
            return self._spectral_gate(audio)
        elif self._algorithm == "wiener":
            return self._wiener_filter(audio)
        return audio

    def calibrate_noise(self, noise_audio: np.ndarray) -> None:
        """
        校准噪声环境：播放一段环境噪音，调用此方法建立噪声模型。

        Args:
            noise_audio: 纯环境噪音片段 (float32)
        """
        self._noise_spectrum = self._estimate_noise_spectrum(noise_audio)
        self._noise_spectrum_smooth = self._noise_spectrum.copy()
        self._noise_frames_count = self._calibration_frames
        logger.info("噪声校准完成")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _spectral_gate(self, audio: np.ndarray) -> np.ndarray:
        """频谱门降噪"""
        audio_f32 = audio.astype(np.float32)

        # 短时傅里叶变换 (STFT)
        n_frames = (len(audio_f32) - self._fft_size) // self._hop_size + 1
        if n_frames <= 0:
            return audio

        output = np.zeros(len(audio_f32), dtype=np.float32)
        weight = np.zeros(len(audio_f32), dtype=np.float32)

        for i in range(n_frames):
            start = i * self._hop_size
            frame = audio_f32[start:start + self._fft_size] * self._window

            # FFT
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # 噪声谱估计（滑动平均）
            if self._noise_spectrum_smooth is None:
                self._noise_spectrum_smooth = magnitude * 0.1
            else:
                if self._noise_frames_count < self._calibration_frames:
                    # 校准阶段
                    self._noise_spectrum_smooth = (
                        self._smoothing_factor * self._noise_spectrum_smooth
                        + (1 - self._smoothing_factor) * magnitude
                    )
                    self._noise_frames_count += 1

            # 计算增益（频谱减法）
            noise_floor = self._noise_spectrum_smooth
            gain_db = self._noise_reduction_db / 20.0
            gain = np.maximum(
                0.01,
                1.0 - (noise_floor / (magnitude + 1e-10)) * (10 ** gain_db)
            )
            gain = np.clip(gain, 0.0, 1.0)

            # 软掩膜
            clean_magnitude = magnitude * gain

            # 逆 FFT
            clean_spectrum = clean_magnitude * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_spectrum, n=self._fft_size)
            clean_frame *= self._window

            output[start:start + self._fft_size] += clean_frame
            weight[start:start + self._fft_size] += self._window ** 2

        # 重叠相加
        epsilon = 1e-10
        output = output / (weight + epsilon)
        output = np.clip(output, -1.0, 1.0)

        return output.astype(np.float32)

    def _wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """维纳滤波降噪"""
        audio_f32 = audio.astype(np.float32)

        if self._noise_spectrum_smooth is None:
            return audio

        n_frames = (len(audio_f32) - self._fft_size) // self._hop_size + 1
        if n_frames <= 0:
            return audio

        output = np.zeros(len(audio_f32), dtype=np.float32)
        weight = np.zeros(len(audio_f32), dtype=np.float32)

        for i in range(n_frames):
            start = i * self._hop_size
            frame = audio_f32[start:start + self._fft_size] * self._window
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)

            # 维纳增益: G = max(0, (S - N) / S)
            noise_floor = self._noise_spectrum_smooth
            snr = (magnitude ** 2) / (noise_floor ** 2 + 1e-10)
            wiener_gain = snr / (snr + 1.0)
            wiener_gain = np.clip(wiener_gain, 0.01, 1.0)

            clean_spectrum = spectrum * wiener_gain
            clean_frame = np.fft.irfft(clean_spectrum, n=self._fft_size)
            clean_frame *= self._window

            output[start:start + self._fft_size] += clean_frame
            weight[start:start + self._fft_size] += self._window ** 2

        return (output / (weight + 1e-10)).astype(np.float32)

    def _estimate_noise_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """从音频片段估计噪声频谱"""
        audio_f32 = audio.astype(np.float32)
        n_frames = (len(audio_f32) - self._fft_size) // self._hop_size + 1
        if n_frames <= 0:
            return np.zeros(self._fft_size // 2 + 1, dtype=np.float32)

        noise = np.zeros(self._fft_size // 2 + 1, dtype=np.float32)
        for i in range(n_frames):
            start = i * self._hop_size
            frame = audio_f32[start:start + self._fft_size] * self._window
            noise += np.abs(np.fft.rfft(frame))
        return noise / n_frames


class EchoCanceller:
    """
    回声消除器

    基于 NLMS (Normalized Least Mean Squares) 自适应滤波器。
    适合路由器端轻量级回声消除。

    使用方法:
        aec = EchoCanceller(config)
        clean = aec.process(mic_signal, speaker_signal)
    """

    def __init__(self, config: AudioConfig):
        self._config = config

        # 滤波器长度 (样本数)
        filter_length_ms = config.aec_filter_length_ms
        self._filter_len = int(config.sample_rate * filter_length_ms / 1000)
        self._weights = np.zeros(self._filter_len, dtype=np.float32)

        # NLMS 参数
        self._step_size: float = 0.1
        self._epsilon: float = 1e-6

        # 参考信号缓冲
        self._ref_buffer = np.zeros(self._filter_len, dtype=np.float32)

        logger.info("回声消除器已初始化: filter_len=%d samples (%d ms)",
                     self._filter_len, filter_length_ms)

    def process(self, mic_signal: np.ndarray,
                speaker_signal: np.ndarray) -> np.ndarray:
        """
        处理输入信号，消除回声。

        Args:
            mic_signal: 麦克风捕获信号 (float32)
            speaker_signal: 当前播放的音频信号 (float32，回声参考)

        Returns:
            消除回声后的信号
        """
        if not self._config.aec_enabled:
            return mic_signal

        mic = mic_signal.astype(np.float32).flatten()
        ref = speaker_signal.astype(np.float32).flatten()

        n_samples = min(len(mic), len(ref))
        if n_samples == 0:
            return mic_signal

        output = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            # 更新参考信号缓冲
            self._ref_buffer[1:] = self._ref_buffer[:-1]
            self._ref_buffer[0] = ref[i]

            # 估计回声
            echo_estimate = np.dot(self._weights, self._ref_buffer)

            # 消除回声
            output[i] = mic[i] - echo_estimate

            # 自适应更新权重 (NLMS)
            ref_power = np.dot(self._ref_buffer, self._ref_buffer) + self._epsilon
            self._weights += (
                self._step_size * output[i] * self._ref_buffer / ref_power
            )

        return output.astype(np.float32)

    def reset(self) -> None:
        """重置滤波器状态"""
        self._weights = np.zeros(self._filter_len, dtype=np.float32)
        self._ref_buffer = np.zeros(self._filter_len, dtype=np.float32)


class AudioPreprocessor:
    """
    音频预处理器（统一接口）

    组合降噪 + 回声消除 + 自动增益控制。
    提供简洁的最外层调用 API。

    使用方法:
        preprocessor = AudioPreprocessor(config)
        clean = preprocessor.process(raw_mic, speaker_ref)
    """

    def __init__(self, config: AudioConfig):
        self._config = config
        self._denoiser = NoiseReducer(config) if config.denoise_enabled else None
        self._echo_canceller = EchoCanceller(config) if config.aec_enabled else None
        self._agc_gain: float = 1.0
        self._target_rms: float = 0.1

    def process(self, mic_audio: np.ndarray,
                speaker_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """
        完整的音频预处理流水线。

        Args:
            mic_audio: 麦克风输入 (int16 或 float32)
            speaker_ref: 扬声器参考信号 (用于回声消除，可选)

        Returns:
            处理后的音频 (float32, 归一化)
        """
        # 类型统一
        if mic_audio.dtype == np.int16:
            audio = mic_audio.astype(np.float32) / 32768.0
        else:
            audio = mic_audio.astype(np.float32)

        # 1. 回声消除
        if self._echo_canceller and speaker_ref is not None:
            audio = self._echo_canceller.process(audio, speaker_ref)

        # 2. 降噪
        if self._denoiser:
            audio = self._denoiser.process(audio)

        # 3. 自动增益控制
        audio = self._auto_gain_control(audio)

        return audio.astype(np.float32)

    def calibrate_noise(self, noise_audio: np.ndarray) -> None:
        """校准环境噪音"""
        if self._denoiser:
            if noise_audio.dtype == np.int16:
                noise = noise_audio.astype(np.float32) / 32768.0
            else:
                noise = noise_audio.astype(np.float32)
            self._denoiser.calibrate_noise(noise)
            logger.info("环境噪音校准完成")

    def reset(self) -> None:
        """重置所有预处理状态"""
        if self._echo_canceller:
            self._echo_canceller.reset()
        self._agc_gain = 1.0

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _auto_gain_control(self, audio: np.ndarray) -> np.ndarray:
        """简易自动增益控制"""
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-10

        # 平滑调整增益
        target_gain = self._target_rms / rms
        target_gain = np.clip(target_gain, 0.1, 10.0)
        self._agc_gain = 0.9 * self._agc_gain + 0.1 * target_gain

        return (audio * self._agc_gain).astype(np.float32)
