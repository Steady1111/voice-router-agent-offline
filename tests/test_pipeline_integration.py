"""
VoiceRouterPipeline 集成测试

测试完整管道：初始化、process_utterance、回调、状态机
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite import VoiceRouterPipeline, PipelineConfig
from voice_router_lite.pipeline import PipelineState


# ======================================================================
# VoiceRouterPipeline 生命周期
# ======================================================================

class TestPipelineLifecycle:
    """管道生命周期"""

    @pytest.fixture
    def pipeline(self):
        p = VoiceRouterPipeline()
        yield p
        p.close()

    def test_initialize(self, pipeline):
        ok = pipeline.initialize()
        assert ok is True

    def test_start_stop(self, pipeline):
        pipeline.initialize()
        pipeline.start()
        pipeline.stop()

    def test_close(self, pipeline):
        pipeline.initialize()
        pipeline.close()

    def test_close_without_init(self, pipeline):
        pipeline.close()  # 不崩溃

    def test_start_without_init(self, pipeline):
        pipeline.start()  # 应自动创建 AudioCapture

    def test_multiple_start_stop(self, pipeline):
        pipeline.initialize()
        pipeline.start()
        pipeline.stop()
        pipeline.start()
        pipeline.stop()


# ======================================================================
# VoiceRouterPipeline process_utterance
# ======================================================================

class TestPipelineProcessUtterance:
    """管道指令处理"""

    @pytest.fixture
    def pipeline(self):
        p = VoiceRouterPipeline()
        p.initialize()
        yield p
        p.close()

    def test_process_text_open_fan(self, pipeline):
        result = pipeline.process_utterance(text="打开风扇")
        assert result["success"] is True
        assert result["intent"] == "device_control"
        assert "message" in result

    def test_process_text_close_led(self, pipeline):
        result = pipeline.process_utterance(text="关闭LED")
        assert result["success"] is True

    def test_process_text_unknown(self, pipeline):
        result = pipeline.process_utterance(text="今天天气不错")
        assert result["success"] is False
        assert "无法理解" in result["message"]

    def test_process_text_help(self, pipeline):
        result = pipeline.process_utterance(text="帮助")
        assert result["success"] is True

    def test_process_text_empty(self, pipeline):
        result = pipeline.process_utterance(text="")
        assert result["success"] is False

    def test_process_multiple_commands(self, pipeline):
        commands = [
            ("打开风扇", True),
            ("关闭LED", True),
            ("帮助", True),
            ("今天天气不错", False),
        ]
        for cmd, expect_ok in commands:
            result = pipeline.process_utterance(text=cmd)
            assert result["success"] == expect_ok, f"'{cmd}' expected {expect_ok}"

    def test_result_structure(self, pipeline):
        result = pipeline.process_utterance(text="打开风扇")
        required_keys = ["success", "text", "intent", "confidence", "slots", "message"]
        for key in required_keys:
            assert key in result, f"缺少 key: {key}"

    def test_confidence_in_result(self, pipeline):
        result = pipeline.process_utterance(text="打开风扇")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_silence_audio(self, pipeline):
        """给静音音频应该返回失败"""
        silence = np.zeros(16000, dtype=np.int16)
        result = pipeline.process_utterance(audio_data=silence)
        assert result["success"] is False


# ======================================================================
# VoiceRouterPipeline 回调
# ======================================================================

class TestPipelineCallbacks:
    """管道回调"""

    @pytest.fixture
    def pipeline(self):
        p = VoiceRouterPipeline()
        p.initialize()
        yield p
        p.close()

    def test_on_result_callback(self, pipeline):
        results = []

        def cb(data):
            results.append(data)

        pipeline.on_result(cb)
        pipeline.process_utterance(text="打开风扇")
        assert len(results) == 1
        assert results[0]["intent"] == "device_control"

    def test_on_error_callback(self, pipeline):
        errors = []

        def cb(msg):
            errors.append(msg)

        pipeline.on_error(cb)
        pipeline.process_utterance(text="打开风扇")
        # 正常情况下应该没有错误

    def test_multiple_callbacks(self, pipeline):
        c1, c2 = [], []

        pipeline.on_result(lambda d: c1.append(d))
        pipeline.on_result(lambda d: c2.append(d))
        pipeline.process_utterance(text="打开风扇")

        assert len(c1) == 1
        assert len(c2) == 1
        assert c1[0]["intent"] == c2[0]["intent"]
        assert c1[0]["intent"] == "device_control"

    def test_callback_exception_handled(self, pipeline):
        # 回调中抛异常不应影响管道
        def broken_cb(data):
            raise RuntimeError("callback error")

        pipeline.on_result(broken_cb)
        result = pipeline.process_utterance(text="打开风扇")
        assert result["success"] is True  # 回调异常不影响


# ======================================================================
# VoiceRouterPipeline 状态机
# ======================================================================

class TestPipelineState:
    """管道状态机"""

    @pytest.fixture
    def pipeline(self):
        p = VoiceRouterPipeline()
        p.initialize()
        yield p
        p.close()

    def test_initial_state(self, pipeline):
        assert pipeline._get_state() == PipelineState.IDLE

    def test_state_transition_on_process(self, pipeline):
        pipeline.process_utterance(text="打开风扇")
        assert pipeline._get_state() == PipelineState.IDLE  # 处理完回到 IDLE

    def test_set_get_state(self, pipeline):
        pipeline._set_state(PipelineState.PROCESSING)
        assert pipeline._get_state() == PipelineState.PROCESSING

        pipeline._set_state(PipelineState.IDLE)
        assert pipeline._get_state() == PipelineState.IDLE

    def test_all_states_defined(self):
        states = [
            PipelineState.IDLE,
            PipelineState.LISTENING,
            PipelineState.WAKED,
            PipelineState.RECORDING,
            PipelineState.PROCESSING,
            PipelineState.EXECUTING,
            PipelineState.SPEAKING,
        ]
        # 验证所有状态都有定义
        for s in states:
            assert isinstance(s, PipelineState)

    def test_utterance_duration(self, pipeline):
        pipeline._utterance_buffer = [
            np.zeros(480, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
        ]
        duration = pipeline._utterance_duration()
        assert duration == 960 / 16000 * 1.0  # ~0.06s

    def test_empty_utterance_duration(self, pipeline):
        pipeline._utterance_buffer = []
        duration = pipeline._utterance_duration()
        assert duration == 0.0


# ======================================================================
# VoiceRouterPipeline 配置
# ======================================================================

class TestPipelineConfig:
    """管道配置"""

    def test_custom_config(self):
        cfg = PipelineConfig()
        cfg.wake_word_threshold = 0.85
        cfg.enable_kws = False

        pipeline = VoiceRouterPipeline(config=cfg)
        assert pipeline._config.wake_word_threshold == 0.85
        assert pipeline._config.enable_kws is False

    def test_wake_word_override(self):
        cfg = PipelineConfig()
        cfg.wake_word = "你好小T"
        pipeline = VoiceRouterPipeline(config=cfg)
        assert pipeline._config.wake_word == "你好小T"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
