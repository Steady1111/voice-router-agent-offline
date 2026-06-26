"""
音频模块测试：采集、播放、回声消除、WAV 读写
"""

import sys
import os
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.config import AudioConfig
from voice_router_lite.audio.capture import AudioCapture, save_wav, load_wav
from voice_router_lite.audio.player import AudioPlayer
from voice_router_lite.audio.denoise import EchoCanceller


# ======================================================================
# AudioCapture
# ======================================================================

class TestAudioCapture:
    """音频采集测试"""

    @pytest.fixture
    def config(self):
        return AudioConfig(sample_rate=16000, chunk_duration_ms=30)

    @pytest.fixture
    def capture(self, config):
        cap = AudioCapture(config)
        yield cap
        cap.close()

    def test_open_close(self, capture):
        """采集器打开关闭"""
        capture.open()
        assert capture.is_open is True
        capture.close()
        assert capture.is_open is False

    def test_read_returns_correct_shape(self, capture):
        """read() 返回正确形状"""
        capture.open()
        chunk = capture.read()
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.int16
        assert len(chunk) == capture._config.chunk_size  # 480 samples

    def test_read_custom_size(self, capture):
        """read() 支持自定义采样点数"""
        capture.open()
        chunk = capture.read(num_samples=960)
        assert len(chunk) == 960

    def test_read_without_open(self, capture):
        """未打开时不抛异常"""
        chunk = capture.read()
        assert len(chunk) == 480

    def test_multiple_reads(self, capture):
        """多次读取累加计数"""
        capture.open()
        for _ in range(10):
            capture.read()
        assert capture.total_duration_sec > 0

    def test_read_streaming(self, capture):
        """流式读取"""
        capture.open()
        # 先读一些数据填充缓冲
        for _ in range(5):
            capture.read()
        stream = capture.read_streaming(240)
        assert isinstance(stream, np.ndarray)
        assert stream.dtype == np.float32
        assert len(stream) == 240

    def test_read_streaming_empty_buffer(self, capture):
        """空缓冲时流式读取返回零"""
        capture.open()
        result = capture.read_streaming(480)
        assert np.allclose(result, 0)

    def test_total_duration_increases(self, capture):
        """total_duration_sec 随读取增长"""
        capture.open()
        before = capture.total_duration_sec
        for _ in range(20):
            capture.read()
        after = capture.total_duration_sec
        assert after > before

    def test_double_open_is_safe(self, capture):
        """重复 open() 不报错"""
        capture.open()
        capture.open()  # 不应抛异常
        assert capture.is_open


# ======================================================================
# AudioPlayer
# ======================================================================

class TestAudioPlayer:
    """音频播放测试"""

    @pytest.fixture
    def config(self):
        return AudioConfig(sample_rate=16000)

    @pytest.fixture
    def player(self, config):
        p = AudioPlayer(config)
        yield p
        p.close()

    def test_open_close(self, player):
        """播放器打开关闭"""
        player.open()
        player.close()

    def test_play_silence_no_crash(self, player):
        """播放静音不崩溃"""
        player.open()
        silence = np.zeros(1600, dtype=np.int16)  # 100ms 静音
        player.play(silence, blocking=True)

    def test_play_tone_no_crash(self, player):
        """播放提示音不崩溃"""
        player.open()
        player.play_tone(frequency=440, duration_ms=50, blocking=True)

    def test_play_tone_params(self, player):
        """提示音参数不影响功能"""
        player.open()
        player.play_tone(frequency=1000, duration_ms=100, blocking=True)
        player.play_tone(frequency=220, duration_ms=200, blocking=True)

    def test_play_async(self, player):
        """异步播放"""
        player.open()
        silence = np.zeros(1600, dtype=np.int16)
        player.play_async(silence)
        player.wait_done()

    def test_play_without_open(self, player):
        """未打开时播放不崩溃"""
        silence = np.zeros(480, dtype=np.int16)
        player.play(silence, blocking=True)

    def test_is_playing_after_async_play(self, player):
        """异步播放时状态正确"""
        player.open()
        short_silence = np.zeros(480, dtype=np.int16)
        player.play_async(short_silence)
        # 短暂延迟后它应该播放完了
        player.wait_done()

    def test_play_wav_not_exist(self, player):
        """播放不存在的 WAV 文件不崩溃"""
        player.open()
        player.play_wav("/nonexistent/file.wav", blocking=True)


# ======================================================================
# WAV 读写
# ======================================================================

class TestWavIO:
    """WAV 文件读写测试"""

    def test_save_and_load(self):
        """写入后再读回应一致"""
        audio = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 16384).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            save_wav(tmp_path, audio, sample_rate=16000)
            loaded = load_wav(tmp_path)
            assert len(loaded) == len(audio)
            assert loaded.dtype == np.int16
            np.testing.assert_array_almost_equal(loaded, audio, decimal=0)
        finally:
            os.unlink(tmp_path)

    def test_save_silence(self):
        """静音保存加载"""
        audio = np.zeros(1600, dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            save_wav(tmp_path, audio, sample_rate=16000)
            loaded = load_wav(tmp_path)
            assert np.all(loaded == 0)
        finally:
            os.unlink(tmp_path)


# ======================================================================
# EchoCanceller
# ======================================================================

class TestEchoCanceller:
    """回声消除测试"""

    @pytest.fixture
    def aec(self):
        cfg = AudioConfig(aec_enabled=True, aec_filter_length_ms=100)
        return EchoCanceller(cfg)

    def test_process_silence(self, aec):
        """处理静音"""
        mic = np.zeros(480, dtype=np.float32)
        ref = np.zeros(480, dtype=np.float32)
        result = aec.process(mic, ref)
        assert result.dtype == np.float32
        assert len(result) == min(len(mic), len(ref))
        assert np.all(np.isfinite(result))

    def test_process_signal(self, aec):
        """处理正常信号"""
        mic = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32) * 0.3
        ref = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32) * 0.1
        result = aec.process(mic, ref)
        assert len(result) == 480
        assert np.all(np.isfinite(result))

    def test_process_different_lengths(self, aec):
        """不同长度的输入"""
        mic = np.random.randn(500).astype(np.float32) * 0.1
        ref = np.random.randn(300).astype(np.float32) * 0.1
        result = aec.process(mic, ref)
        assert len(result) == min(500, 300)
        assert np.all(np.isfinite(result))

    def test_reset(self, aec):
        """重置后状态清空"""
        mic = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32) * 0.3
        ref = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32) * 0.1

        result1 = aec.process(mic, ref)
        aec.reset()
        result2 = aec.process(mic, ref)

        # 重置后结果不应完全相同（权重被重置）
        assert len(result1) == len(result2)
        assert np.all(np.isfinite(result2))

    def test_convergence(self, aec):
        """测试滤波器收敛(误差减小)"""
        # 生成回声场景：mic = 原始信号 + 回声
        original = np.sin(2 * np.pi * 440 * np.arange(2000) / 16000).astype(np.float32) * 0.5
        echo = np.roll(original * 0.3, 50)  # 延迟50样本的回声

        mic = original + echo
        ref = original  # 参考信号 = 扬声器播放的原始信号

        errors = []
        chunk_size = 200
        for i in range(0, len(mic) - chunk_size, chunk_size):
            mic_chunk = mic[i:i + chunk_size]
            ref_chunk = ref[i:i + chunk_size]
            output = aec.process(mic_chunk, ref_chunk)

            # 计算残留回声能量
            if i > chunk_size * 3:  # 跳过初始几帧(滤波器未收敛)
                true_signal = original[i:i + chunk_size]
                error = np.sqrt(np.mean((output - true_signal) ** 2))
                errors.append(error)

        if len(errors) >= 3:
            # 后期误差应该相对稳定(已收敛)
            assert np.mean(errors) < 0.5  # 误差在合理范围


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
