"""ESP32 mic fan command overrides for demo ASR mishears."""

from voice_router_lite.web.service import _esp32_mic_fan_override


def test_close_fan_from_strip_wake_only():
    r = _esp32_mic_fan_override("关闭", fan_is_on=True, raw_asr="小提小提关闭")
    assert r is not None
    assert r.slots["state"] == "off"


def test_close_fan_mishears():
    for text in ("关闭一万", "第一关闭站", "李过你风", "官病"):
        r = _esp32_mic_fan_override(text, fan_is_on=True, raw_asr=text)
        assert r is not None, text
        assert r.slots["state"] == "off", text


def test_open_fan_mishears():
    for text in ("打开封了", "晓婷打扇", "打扇"):
        r = _esp32_mic_fan_override(text, fan_is_on=False, raw_asr=f"小提{text}")
        assert r is not None, text
        assert r.slots["state"] == "on", text
