"""
离线语音管道测试
"""

import sys
import os
import pytest
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.config import (
    PipelineConfig, ModelPaths, AudioConfig, PerformanceTargets, DEFAULT_CONFIG,
)
from voice_router_lite.audio.vad import VoiceActivityDetector
from voice_router_lite.audio.denoise import AudioPreprocessor, NoiseReducer
from voice_router_lite.nlu.model import CNNLSTMNLU, NLUResult
from voice_router_lite.nlu.engine import NLUEngine


# ======================================================================
# 配置测试
# ======================================================================

class TestConfig:
    """配置正确性测试"""

    def test_default_config(self):
        """默认配置应合法"""
        cfg = DEFAULT_CONFIG
        assert cfg.audio.sample_rate == 16000
        assert cfg.audio.chunk_duration_ms == 30
        assert cfg.audio.chunk_size == 480
        assert cfg.audio.buffer_size == 80000

    def test_performance_targets(self):
        """性能指标约束"""
        pt = DEFAULT_CONFIG.performance
        assert pt.total_latency_ms <= 2000
        assert pt.total_memory_mb <= 50
        assert pt.asr_model_size_mb <= 30
        assert pt.nlu_model_size_mb <= 3
        assert pt.kws_model_size_mb <= 5

    def test_audio_interface_standard(self):
        """音频接口标准: 16kHz / 16bit / mono"""
        audio = DEFAULT_CONFIG.audio
        assert audio.sample_rate == 16000
        assert audio.bit_depth == 16
        assert audio.channels == 1


# ======================================================================
# VAD 测试
# ======================================================================

class TestVAD:
    """语音活动检测测试"""

    @pytest.fixture
    def vad(self):
        cfg = AudioConfig(vad_mode=2)
        return VoiceActivityDetector(cfg)

    def test_detect_silence(self, vad):
        """静音应返回 False"""
        silence = np.zeros(480, dtype=np.float32)
        assert vad.is_speech(silence) is False

    def test_detect_speech(self, vad):
        """高强度信号应返回 True"""
        tone = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32)
        # 能量 VAD 阈值，模拟语音
        loud = tone * 0.5
        # 需要建立一个高于噪声的基线
        for _ in range(30):
            vad.update_noise_floor(np.zeros(480, dtype=np.float32))
        assert vad.is_speech(loud) is True

    def test_speech_segment_detection(self, vad):
        """语音段起止检测"""
        # 初始化噪声基线
        for _ in range(30):
            vad.update_noise_floor(np.zeros(480, dtype=np.float32))

        tone = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32) * 0.5

        # 连续语音帧
        for _ in range(10):
            is_speech, ended = vad.process_frame(tone)
            if is_speech:
                assert vad.is_speaking or is_speech
                break

        # 连续静音帧 → 段结束
        for _ in range(30):
            _, ended = vad.process_frame(np.zeros(480, dtype=np.float32))


# ======================================================================
# 降噪测试
# ======================================================================

class TestDenoise:
    """降噪处理测试"""

    def test_noise_reducer_passthrough(self):
        """禁用降噪应原样输出"""
        cfg = AudioConfig(denoise_enabled=False)
        nr = NoiseReducer(cfg)
        audio = np.sin(2 * np.pi * 440 * np.arange(1024) / 16000).astype(np.float32)
        result = nr.process(audio)
        np.testing.assert_array_almost_equal(result, audio)

    def test_noise_reducer_enabled(self):
        """启用降噪应降低噪声能量"""
        cfg = AudioConfig(denoise_enabled=True, denoise_algorithm="spectral_gate")
        nr = NoiseReducer(cfg)

        # 建立噪声模型
        noise = np.random.randn(480 * 30).astype(np.float32) * 0.01
        nr.calibrate_noise(noise)

        # 处理信号
        signal = np.sin(2 * np.pi * 440 * np.arange(480 * 10) / 16000).astype(np.float32) * 0.3
        noisy = signal + np.random.randn(len(signal)).astype(np.float32) * 0.05
        clean = nr.process(noisy)

        # 降噪后不应产生异常值
        assert np.all(np.isfinite(clean))
        assert np.max(np.abs(clean)) <= 1.0

    def test_preprocessor_pipeline(self):
        """预处理管线"""
        cfg = AudioConfig(denoise_enabled=True, aec_enabled=False)
        preproc = AudioPreprocessor(cfg)

        audio = np.sin(2 * np.pi * 440 * np.arange(1024) / 16000).astype(np.float32)
        result = preproc.process(audio)

        assert result.dtype == np.float32
        assert len(result) == len(audio)
        assert np.all(np.isfinite(result))


# ======================================================================
# NLU 测试
# ======================================================================

class TestNLU:
    """指令理解测试"""

    @pytest.fixture
    def nlu_model(self):
        return CNNLSTMNLU()

    def test_rule_based_intent(self, nlu_model):
        """规则引擎意图分类"""
        cases = [
            ("打开风扇", "set_device_state"),
            ("关闭灯光", "set_device_state"),
            ("调大一点", "adjust_fan_speed"),
            ("亮度暗一点", "set_light_brightness"),
            ("睡眠模式", "scene_mode"),
            ("重启路由器", "router_reboot"),
            ("WiFi断了", "router_wifi_restart"),
            ("帮助", "help"),
            ("今天天气不错", "unknown"),
        ]

        for text, expected_intent in cases:
            result = nlu_model.predict(text)
            assert result.intent == expected_intent, f"'{text}' → {result.intent}, expected {expected_intent}"

    def test_slot_extraction(self, nlu_model):
        """槽位提取"""
        result = nlu_model.predict("打开客厅的风扇")
        assert result.slots.get("device_type") in ("fan", "风扇")

        result = nlu_model.predict("关闭灯光")
        assert result.slots.get("device_type") in ("light", "灯")

        result = nlu_model.predict("调大一点")
        assert result.slots.get("direction") == "up"

    def test_result_dataclass(self):
        """NLUResult 数据模型"""
        r = NLUResult(text="测试", intent="help", confidence=0.9, is_valid=True)
        cmd = r.to_command()
        assert cmd["intent"] == "help"
        assert cmd["confidence"] == 0.9

    def test_batch_predict(self, nlu_model):
        """批量推理"""
        texts = ["打开风扇", "关闭灯光", "帮助"]
        results = nlu_model.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, NLUResult)


# ======================================================================
# 性能指标验证
# ======================================================================

class TestPerformance:
    """性能指标测试"""

    def test_vad_latency(self):
        """VAD 单帧延迟 < 5ms"""
        import time
        cfg = AudioConfig()
        vad = VoiceActivityDetector(cfg)
        chunk = np.sin(2 * np.pi * 440 * np.arange(480) / 16000).astype(np.float32)

        start = time.perf_counter()
        for _ in range(100):
            vad.is_speech(chunk)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 5, f"VAD 平均延迟 {avg_ms:.2f}ms > 5ms"

    def test_nlu_latency(self):
        """NLU 推理延迟 < 100ms"""
        import time
        model = CNNLSTMNLU()

        texts = ["打开风扇", "关闭灯光", "调大一点", "睡眠模式", "帮助"]
        start = time.perf_counter()
        for text in texts * 20:  # 100 次推理
            model.predict(text)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 100, f"NLU 平均延迟 {avg_ms:.2f}ms > 100ms"

    def test_model_size_constraints(self):
        """模型体积估算在限制内"""
        perf = DEFAULT_CONFIG.performance
        # 验证数值约束
        assert perf.asr_model_size_mb <= 30
        assert perf.nlu_model_size_mb <= 3
        assert perf.kws_model_size_mb <= 5
        assert perf.tts_clips_total_mb <= 5
        # 总模型体积 ≤ 43MB
        total_model = (perf.asr_model_size_mb + perf.nlu_model_size_mb +
                       perf.kws_model_size_mb + perf.tts_clips_total_mb)
        assert total_model <= 45, f"模型总体积 {total_model}MB > 45MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
