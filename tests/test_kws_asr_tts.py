"""
KWS / ASR / TTS 引擎测试
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.config import ModelPaths, AudioConfig
from voice_router_lite.kws.engine import KWSEngine, KWSResult
from voice_router_lite.asr.engine import ASREngine, ASRResult
from voice_router_lite.tts.engine import TTSEngine


# ======================================================================
# KWSResult
# ======================================================================

class TestKWSResult:
    """唤醒词检测结果"""

    def test_default(self):
        r = KWSResult(detected=False)
        assert r.detected is False
        assert r.keyword == ""
        assert r.confidence == 0.0
        assert r.timestamp == 0.0

    def test_detected(self):
        r = KWSResult(detected=True, keyword="小T小T", confidence=0.95, timestamp=123.45)
        assert r.detected is True
        assert r.keyword == "小T小T"
        assert r.confidence == 0.95
        assert r.timestamp == 123.45


# ======================================================================
# KWSEngine
# ======================================================================

class TestKWSEngine:
    """唤醒词引擎"""

    @pytest.fixture
    def kws(self):
        paths = ModelPaths()
        cfg = AudioConfig()
        k = KWSEngine(paths, cfg)
        yield k
        k.close()

    def test_initialize_energy_mode(self, kws):
        """模型文件不存在时回退能量模式"""
        ok = kws.initialize()
        assert ok is True
        assert kws._initialized is True
        assert kws._mode == "energy"

    def test_detect_silence(self, kws):
        kws.initialize()
        silence = np.zeros(480, dtype=np.float32)
        result = kws.detect(silence)
        assert isinstance(result, KWSResult)
        assert result.detected is False

    def test_detect_low_energy(self, kws):
        kws.initialize()
        low = np.random.randn(480).astype(np.float32) * 0.01
        result = kws.detect(low)
        assert result.detected is False

    def test_reset(self, kws):
        kws.initialize()
        # 触发一些检测
        for _ in range(5):
            kws.detect(np.zeros(480, dtype=np.float32))
        kws.reset()
        # 不应抛异常
        assert True

    def test_mode_property(self, kws):
        kws.initialize()
        assert kws.mode in ("sherpa", "energy")

    def test_close_idempotent(self, kws):
        kws.initialize()
        kws.close()
        kws.close()  # 重复关闭不报错

    def test_detect_without_init(self, kws):
        """未初始化时返回空结果"""
        result = kws.detect(np.zeros(480, dtype=np.float32))
        assert result.detected is False

    def test_on_wake_callback(self, kws):
        kws.initialize()
        callback_called = []

        def cb(result):
            callback_called.append(result)

        kws.on_wake(cb)
        # 能量检测不会触发（能量太低）
        silence = np.zeros(480, dtype=np.float32)
        kws.detect(silence)
        assert len(callback_called) == 0


# ======================================================================
# ASRResult
# ======================================================================

class TestASRResult:
    """ASR 结果"""

    def test_default(self):
        r = ASRResult(text="")
        assert r.text == ""
        assert r.is_final is True
        assert r.confidence == 0.0
        assert r.tokens == []

    def test_with_text(self):
        r = ASRResult(text="你好世界", is_final=True, confidence=0.95)
        assert r.text == "你好世界"
        assert r.confidence == 0.95

    def test_is_empty_true(self):
        r = ASRResult(text="")
        assert r.is_empty is True

        r = ASRResult(text="  ")
        assert r.is_empty is True

    def test_is_empty_false(self):
        r = ASRResult(text="hello")
        assert r.is_empty is False

    def test_partial_result(self):
        r = ASRResult(text="正在识别...", is_final=False)
        assert r.is_final is False


# ======================================================================
# ASREngine
# ======================================================================

class TestASREngine:
    """ASR 引擎"""

    @pytest.fixture
    def asr(self):
        paths = ModelPaths()
        cfg = AudioConfig()
        a = ASREngine(paths, cfg)
        yield a
        a.close()

    def test_initialize_mock_mode(self, asr):
        """sherpa-onnx 不可用时 mock 模式"""
        ok = asr.initialize()
        assert ok is True

    def test_transcribe_silence(self, asr):
        asr.initialize()
        silence = np.zeros(16000, dtype=np.int16)  # 1秒静音
        result = asr.transcribe(silence)
        assert isinstance(result, ASRResult)
        # mock 模式或真实模式都可能返回空
        assert result.text == ""

    def test_transcribe_noise(self, asr):
        asr.initialize()
        noise = np.random.randn(16000).astype(np.float32) * 0.01
        result = asr.transcribe(noise)
        assert isinstance(result, ASRResult)

    def test_start_stop_stream(self, asr):
        asr.initialize()
        asr.start_stream()
        assert asr.is_streaming is True

        # 发送一些数据
        for _ in range(10):
            chunk = np.zeros(480, dtype=np.float32)
            partial = asr.process_chunk(chunk)

        result = asr.stop_stream()
        assert asr.is_streaming is False
        assert isinstance(result, ASRResult)

    def test_process_chunk_when_not_streaming(self, asr):
        asr.initialize()
        result = asr.process_chunk(np.zeros(480, dtype=np.float32))
        assert result is None

    def test_stop_stream_when_not_streaming(self, asr):
        asr.initialize()
        result = asr.stop_stream()
        assert isinstance(result, ASRResult)

    def test_close_idempotent(self, asr):
        asr.initialize()
        asr.close()
        asr.close()  # 不报错

    def test_transcribe_without_init(self, asr):
        result = asr.transcribe(np.zeros(16000, dtype=np.int16))
        assert isinstance(result, ASRResult)

    def test_transcribe_file_not_exist(self, asr):
        asr.initialize()
        result = asr.transcribe_file("/nonexistent/file.wav")
        assert isinstance(result, ASRResult)
        assert result.text == ""


# ======================================================================
# TTSEngine
# ======================================================================

class TestTTSEngine:
    """TTS 引擎"""

    @pytest.fixture
    def tts(self):
        paths = ModelPaths()
        cfg = AudioConfig()
        t = TTSEngine(paths, cfg)
        yield t
        t.close()

    def test_initialize(self, tts):
        ok = tts.initialize()
        assert ok is True
        assert tts._initialized is True

    def test_beep_ok(self, tts):
        tts.initialize()
        tts.beep_ok()  # 不应崩溃

    def test_beep_error(self, tts):
        tts.initialize()
        tts.beep_error()

    def test_beep_wake(self, tts):
        tts.initialize()
        tts.beep_wake()

    def test_speak_builtin_tone(self, tts):
        tts.initialize()
        tts.speak("ok_beep", blocking=True)

    def test_speak_nonexistent_clip(self, tts):
        tts.initialize()
        tts.speak("nonexistent_clip", blocking=True)  # 不应崩溃

    def test_speak_sentence_simple(self, tts):
        tts.initialize()
        tts.speak_sentence("无法识别", blocking=True)

    def test_speak_sentence_unknown(self, tts):
        tts.initialize()
        tts.speak_sentence("某个不能合成的句子abc", blocking=True)
        # 不应崩溃

    def test_speak_feedback(self, tts):
        tts.initialize()
        tts.speak_feedback(
            "set_device_state",
            {"device_type": "fan", "state": "on"},
            blocking=True,
        )

    def test_add_clip(self, tts):
        tts.initialize()
        fake_clip = np.zeros(1600, dtype=np.int16)
        tts.add_clip("test_clip", fake_clip)
        tts.speak("test_clip", blocking=True)

    def test_generate_tone(self, tts):
        """提示音生成函数"""
        tone = tts._generate_tone(440, 100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.int16
        assert len(tone) > 0

    def test_generate_tone_double(self, tts):
        tone = tts._generate_tone_double(440, 50, 880, 50)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.int16
        assert len(tone) > 0

    def test_close_idempotent(self, tts):
        tts.initialize()
        tts.close()
        tts.close()  # 重复关闭不报错

    def test_build_feedback_text(self):
        texts = [
            ("set_device_state", {"device_type": "fan", "state": "on"}, True),
            ("set_device_state", {"device_type": "light", "state": "off"}, True),
            ("adjust_fan_speed", {}, True),
            ("set_light_brightness", {}, True),
            ("query_device_status", {"device_type": "fan"}, True),
            ("timer_setting", {}, True),
            ("scene_mode", {"mode_name": "睡眠"}, True),
            ("router_reboot", {}, True),
            ("router_wifi_restart", {}, True),
            ("control_ac", {}, True),
            ("help", {}, True),
            ("unknown_intent", {}, False),
        ]
        for intent, slots, expected_nonempty in texts:
            text = TTSEngine._build_feedback_text(intent, slots)
            if expected_nonempty:
                assert len(text) > 0, f"{intent} 反馈为空"
            # 不应报错
            assert isinstance(text, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
