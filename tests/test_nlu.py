"""
NLU 模块深度测试：意图分类边界、槽位提取、引擎初始化、后处理
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.config import ModelPaths
from voice_router_lite.nlu.model import CNNLSTMNLU, NLUResult
from voice_router_lite.nlu.engine import NLUEngine


# ======================================================================
# NLUResult 数据模型
# ======================================================================

class TestNLUResult:
    """NLUResult 数据模型"""

    def test_default_values(self):
        r = NLUResult(text="")
        assert r.text == ""
        assert r.intent == "unknown"
        assert r.confidence == 0.0
        assert r.slots == {}
        assert r.is_valid is False

    def test_to_command_full(self):
        r = NLUResult(
            text="打开风扇",
            intent="device_control",
            confidence=0.95,
            slots={"device_type": "fan", "state": "on"},
            is_valid=True,
        )
        cmd = r.to_command()
        assert cmd["intent"] == "device_control"
        assert cmd["confidence"] == 0.95
        assert cmd["slots"]["device_type"] == "fan"

    def test_to_command_invalid(self):
        r = NLUResult(text="xxxx", intent="unknown", confidence=0.0, is_valid=False)
        cmd = r.to_command()
        assert cmd["intent"] == "unknown"
        assert not r.is_valid

    def test_is_empty_true(self):
        r = ASRResult_test("  ")
        assert r.is_empty is True

    def test_is_empty_false(self):
        r = ASRResult_test("hello")
        assert r.is_empty is False


def ASRResult_test(text):
    """Mock ASRResult for is_empty test"""
    from voice_router_lite.asr.engine import ASRResult
    return ASRResult(text=text)


# ======================================================================
# CNNLSTMNLU 规则引擎
# ======================================================================

class TestCNNLSTMNLU:
    """CNNLSTMNLU 模型"""

    @pytest.fixture
    def model(self):
        return CNNLSTMNLU()

    # ---- 意图分类 ----

    @pytest.mark.parametrize("text,expected", [
        ("打开风扇", "device_control"),
        ("关掉灯", "device_control"),
        ("启动空调", "device_control"),
        ("停止风扇", "device_control"),
        ("调大一点", "device_adjust"),
        ("风速小一点", "device_adjust"),
        ("亮度调高", "device_adjust"),
        ("暗一点", "device_adjust"),
        ("风扇开着吗", "device_query"),
        ("温度多少", "router_network_query"),
        ("10分钟后关闭", "timer_setting"),
        ("定时1小时", "timer_setting"),
        ("睡眠模式", "scene_mode"),
        ("工作模式", "scene_mode"),
        ("离开模式", "scene_mode"),
        ("重启路由器", "router_reboot"),
        ("网络卡了", "router_reboot"),
        ("WiFi断了", "router_wifi_restart"),
        ("重启WiFi", "router_wifi_restart"),
        ("打开空调", "device_control"),
        ("温度调高", "device_adjust"),
        ("你能做什么", "help"),
        ("帮助", "help"),
    ])
    def test_intent_classification(self, model, text, expected):
        result = model.predict(text)
        assert result.intent == expected, f"'{text}' → {result.intent}, 期望 {expected}"

    def test_unknown_for_unrelated_text(self, model):
        """无关文本返回 unknown"""
        cases = [
            "今天天气不错",
            "吃了吗",
            "hello world",
            "123456",
            "",
        ]
        for text in cases:
            result = model.predict(text)
            assert result.intent == "unknown", f"'{text}' 应返回 unknown，实际 {result.intent}"

    # ---- 槽位提取 ----

    def test_slot_device_on(self, model):
        r = model.predict("打开风扇")
        assert r.slots.get("device_type") == "fan"
        assert r.slots.get("state") == "on"

    def test_slot_device_off(self, model):
        r = model.predict("关闭灯光")
        assert r.slots.get("device_type") == "light"
        assert r.slots.get("state") == "off"

    def test_slot_direction_up(self, model):
        r = model.predict("调大一点")
        assert r.slots.get("direction") == "up"

    def test_slot_direction_down(self, model):
        r = model.predict("暗一点")
        assert r.slots.get("direction") == "down"

    def test_slot_scene_mode(self, model):
        r = model.predict("睡眠模式")
        assert r.slots.get("mode_name") == "睡眠"

    def test_slot_device_ac(self, model):
        r = model.predict("打开空调")
        assert r.slots.get("device_type") == "ac"

    def test_slot_device_relay(self, model):
        r = model.predict("打开插座")
        assert r.slots.get("device_type") == "relay"

    def test_slot_device_led(self, model):
        r = model.predict("打开LED")
        assert r.slots.get("device_type") == "led"
        assert r.slots.get("state") == "on"

    # ---- 边界 & 特殊场景 ----

    def test_empty_text(self, model):
        r = model.predict("")
        assert r.intent == "unknown"
        assert r.confidence == 0.0
        assert not r.is_valid

    def test_whitespace_text(self, model):
        r = model.predict("   ")
        assert r.intent == "unknown"
        assert not r.is_valid

    def test_case_insensitive(self, model):
        """大小写不敏感"""
        r = model.predict("重启wifi")
        assert r.intent == "router_wifi_restart"

        r = model.predict("wifi断了")
        assert r.intent == "router_wifi_restart"

    def test_multiple_keywords_first_wins(self, model):
        """多关键词匹配最高分"""
        r = model.predict("打开风扇调大")
        # "打开风扇" 匹配 device_control, "调大" 匹配 device_adjust
        assert r.intent == "device_control"

    def test_batch_predict_all_same_type(self, model):
        texts = ["打开风扇", "关闭灯光", "帮助", "睡眠模式", "调大一点"]
        results = model.predict_batch(texts)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, NLUResult)
            assert hasattr(r, "intent")
            assert hasattr(r, "confidence")

    def test_batch_predict_empty_list(self, model):
        results = model.predict_batch([])
        assert results == []

    def test_confidence_range(self, model):
        """置信度在 [0, 1] 之间"""
        texts = [
            "打开风扇", "关闭灯", "调大一点", "睡眠工作模式",
            "重启路由器", "帮助", "今天天气不错",
        ]
        for text in texts:
            r = model.predict(text)
            assert 0.0 <= r.confidence <= 1.0, f"'{text}' confidence={r.confidence}"

    def test_is_valid_when_intent_known(self, model):
        r = model.predict("打开风扇")
        assert r.is_valid is True

    def test_is_valid_when_intent_unknown(self, model):
        r = model.predict("今天天气不错")
        assert r.is_valid is False

    # ---- 内部方法 ----

    def test_tokenize(self, model):
        tokens = model._tokenize("你好")
        assert isinstance(tokens, list)
        assert len(tokens) == model.max_seq_len
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenize_long_text(self, model):
        long_text = "这是一段非常长的测试文本" * 10
        tokens = model._tokenize(long_text)
        assert len(tokens) == model.max_seq_len

    def test_softmax(self, model):
        logits = np.array([1.0, 2.0, 3.0])
        probs = model._softmax(logits)
        assert np.allclose(np.sum(probs), 1.0)
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_large_values(self, model):
        logits = np.array([1000.0, 1001.0, 0.0])
        probs = model._softmax(logits)
        assert np.allclose(np.sum(probs), 1.0)
        assert np.all(np.isfinite(probs))


# ======================================================================
# NLUEngine
# ======================================================================

class TestNLUEngine:
    """NLUEngine 引擎封装"""

    @pytest.fixture
    def engine(self):
        paths = ModelPaths()
        eng = NLUEngine(paths)
        return eng

    def test_initialize_fallback_to_rules(self, engine):
        """模型文件不存在时回退到规则引擎"""
        ok = engine.initialize()
        assert ok is True
        assert engine._initialized is True

    def test_understand_basic(self, engine):
        engine.initialize()
        result = engine.understand("打开风扇")
        assert result.intent == "device_control"
        assert result.slots.get("device_type") == "fan"
        assert result.slots.get("state") == "on"

    def test_understand_without_init(self, engine):
        """未初始化时返回空结果"""
        result = engine.understand("打开风扇")
        assert result.intent == "unknown"
        assert not result.is_valid

    def test_understand_batch(self, engine):
        engine.initialize()
        results = engine.understand_batch(["打开风扇", "关闭灯光"])
        assert len(results) == 2
        assert results[0].intent == "device_control"
        assert results[1].intent == "device_control"

    def test_device_name_normalization(self, engine):
        """设备名称标准化"""
        engine.initialize()
        # Chinese风扇 → fan
        result = engine.understand("打开风扇")
        assert result.slots.get("device_type") == "fan"

    def test_infer_missing_state(self, engine):
        """补全缺失的状态槽位"""
        engine.initialize()
        result = engine.understand("打开风扇")
        assert result.slots.get("state") == "on"

        result = engine.understand("关闭风扇")
        assert result.slots.get("state") == "off"

    def test_adjust_speed_delta(self, engine):
        """调速指令速度步长推断"""
        engine.initialize()
        result = engine.understand("调大一点")
        assert result.slots.get("speed_delta") == "1"

    def test_understand_empty_text(self, engine):
        engine.initialize()
        result = engine.understand("")
        assert result.intent == "unknown"
        assert not result.is_valid


# ======================================================================
# 综合指令测试
# ======================================================================

class TestComprehensiveNLU:
    """NLU 综合指令理解"""

    @pytest.fixture
    def model(self):
        return CNNLSTMNLU()

    # 16 个意图全覆盖
    INTENT_CASES = [
        ("打开风扇", "device_control", {"device_type": "fan", "state": "on"}),
        ("关闭LED", "device_control", {"device_type": "led", "state": "off"}),
        ("调大一点", "device_adjust", {"direction": "up"}),
        ("调小风速", "device_adjust", {"direction": "down"}),
        ("亮度调高", "device_adjust", {"direction": "up"}),
        ("暗一点", "device_adjust", {"direction": "down"}),
        ("风扇开着吗", "device_query", {}),
        ("10分钟后关闭", "timer_setting", {}),
        ("睡眠模式", "scene_mode", {"mode_name": "睡眠"}),
        ("重启路由器", "router_reboot", {}),
        ("WiFi断了", "router_wifi_restart", {}),
        ("打开空调", "device_control", {"device_type": "ac"}),
        ("帮助", "help", {}),
        ("今天天气不错", "unknown", {}),
    ]

    @pytest.mark.parametrize("text,expected_intent,expected_slots", INTENT_CASES)
    def test_comprehensive(self, model, text, expected_intent, expected_slots):
        result = model.predict(text)
        assert result.intent == expected_intent, f"'{text}' intent mismatch: {result.intent}"
        for key, val in expected_slots.items():
            assert result.slots.get(key) == val, (
                f"'{text}' slot '{key}' mismatch: {result.slots.get(key)} != {val}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
