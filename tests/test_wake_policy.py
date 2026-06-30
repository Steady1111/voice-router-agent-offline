"""Tests for ESP32 demo wake policy."""

from voice_router_lite.web.wake_policy import (
    VolumeWakeConfig,
    VolumeWakeDetector,
    strip_wake_prefix,
)


def test_strip_wake_prefix():
    assert strip_wake_prefix("小T小T打开风扇") == "打开风扇"
    assert strip_wake_prefix("小弟关闭风扇") == "关闭风扇"
    assert strip_wake_prefix("小提小提关闭") == "关闭"
    assert strip_wake_prefix("打开风扇") == "打开风扇"


def test_volume_wake_requires_sustained_energy():
    cfg = VolumeWakeConfig(
        rms_threshold=0.03,
        peak_threshold=0.12,
        burst_ratio=3.0,
        min_rise=0.06,
        streak_frames=3,
        cooldown_sec=1.0,
    )
    det = VolumeWakeDetector(cfg)
    t = 1000.0
    assert not det.check(0.01, t)
    assert not det.check(0.15, t + 0.1)
    assert not det.check(0.15, t + 0.2)
    assert det.check(0.15, t + 0.3)
