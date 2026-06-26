"""硬件控制模块 — 风扇调速、数码管显示、OLED 屏幕、ESP32 桥接。

从 voice-router-agent 移植的核心硬件控制逻辑，可直接复用。
"""

from voice_router_lite.hardware.fan import (
    SPEED_LEVELS,
    speed_level_to_pwm,
    pwm_to_speed_level,
)
from voice_router_lite.hardware.display import (
    DisplayState,
    DISP_SEGMENTS,
    format_display_command,
    format_oled_command,
)
from voice_router_lite.hardware.esp32 import (
    ESP32Bridge,
    FanCommand,
    DisplayCommand,
    OledCommand,
)

__all__ = [
    "SPEED_LEVELS",
    "speed_level_to_pwm",
    "pwm_to_speed_level",
    "DisplayState",
    "DISP_SEGMENTS",
    "format_display_command",
    "format_oled_command",
    "ESP32Bridge",
    "FanCommand",
    "DisplayCommand",
    "OledCommand",
]
