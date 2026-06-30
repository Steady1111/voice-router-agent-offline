"""ESP32 演示唤醒策略（方案 1：音量触发 + 单次 ASR + NLU）。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_WAKE_T_LIKE = r"[tT弟地提婷题题徐听天稀希西洗喜细隙夕曦锡体替岐姨晓]"


_WAKE_PREFIX = re.compile(rf"^(?:小{_WAKE_T_LIKE}?|[听替李])+", re.IGNORECASE)


def strip_wake_prefix(text: str) -> str:
    """去掉句首唤醒词，保留后续指令（不全局删「小提」以免误伤）。"""
    t = (text or "").strip()
    for prefix in ("我替", "我是", "五体", "一体", "有事"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    while True:
        m = _WAKE_PREFIX.match(t)
        if not m:
            break
        t = t[m.end():].strip()
    t = re.sub(r"[\s，,。.、]+", " ", t).strip()
    return t


@dataclass
class VolumeWakeConfig:
    rms_threshold: float = 0.035
    peak_threshold: float = 0.12
    burst_ratio: float = 3.0
    min_rise: float = 0.06
    streak_frames: int = 5
    cooldown_sec: float = 3.0

    @classmethod
    def from_env(cls) -> VolumeWakeConfig:
        return cls(
            rms_threshold=float(os.getenv("VOICE_ROUTER_VOLUME_WAKE_RMS", "0.035")),
            peak_threshold=float(os.getenv("VOICE_ROUTER_VOLUME_WAKE_PEAK", "0.12")),
            burst_ratio=float(os.getenv("VOICE_ROUTER_VOLUME_WAKE_BURST", "3.0")),
            min_rise=float(os.getenv("VOICE_ROUTER_VOLUME_WAKE_MIN_RISE", "0.06")),
            streak_frames=int(os.getenv("VOICE_ROUTER_VOLUME_WAKE_FRAMES", "5")),
            cooldown_sec=float(os.getenv("VOICE_ROUTER_WAKE_COOLDOWN_SEC", "3.0")),
        )


class VolumeWakeDetector:
    """音量唤醒：连续高能量帧 + 峰值/突增判定，带自适应底噪。"""

    def __init__(self, config: VolumeWakeConfig | None = None) -> None:
        self.config = config or VolumeWakeConfig.from_env()
        self.noise_floor = 0.008
        self.noise_adapt_rate = 0.03
        self.streak = 0
        self.peak = 0.0
        self.cooldown_until = 0.0

    def update_noise_floor(self, energy: float) -> None:
        if energy < self.noise_floor * 2.5:
            self.noise_floor = (
                (1.0 - self.noise_adapt_rate) * self.noise_floor
                + self.noise_adapt_rate * energy
            )

    def check(self, energy: float, now: float) -> bool:
        if now <= self.cooldown_until:
            return False

        self.update_noise_floor(energy)
        wake_threshold = max(self.config.rms_threshold, self.noise_floor * 2.5)

        if energy > wake_threshold:
            self.streak += 1
            self.peak = max(self.peak, energy)
        else:
            self.reset_streak()

        if self.streak < self.config.streak_frames:
            return False

        burst_ratio = self.peak / max(self.noise_floor, 1e-6)
        rise_above_floor = self.peak - self.noise_floor
        if (
            self.peak >= self.config.peak_threshold
            and burst_ratio >= self.config.burst_ratio
            and rise_above_floor >= self.config.min_rise
        ):
            return True
        return False

    def reset_streak(self) -> None:
        self.streak = 0
        self.peak = 0.0

    def set_cooldown(self, now: float) -> None:
        self.cooldown_until = now + self.config.cooldown_sec
        self.reset_streak()
