"""数码管显示 + OLED 屏幕命令格式化。

从 voice-router-agent/firmware/src/main.cpp 移植显示逻辑。
ESP32 硬件：
  - 四位数码管：GPIO 10-17 (段), GPIO 18-21 (位选) — 共阴直驱，250Hz ISR 扫描
  - 0.96" OLED SSD1306：I2C SDA=GPIO41, SCL=GPIO42, 地址 0x3C
"""

from dataclasses import dataclass, field
from typing import Optional


# ── 数码管段码 (共阴) ──
# 段顺序: A B C D E F G DP
DISP_SEGMENTS: dict[str, int] = {
    "0": 0b11111100,
    "1": 0b01100000,
    "2": 0b11011010,
    "3": 0b11110010,
    "4": 0b01100110,
    "5": 0b10110110,
    "6": 0b10111110,
    "7": 0b11100000,
    "8": 0b11111110,
    "9": 0b11110110,
    "-": 0b00000010,
    " ": 0b00000000,
    "A": 0b11101110,
    "E": 0b10011110,
    "F": 0b10001110,
    "H": 0b01101110,
    "L": 0b00011100,
    "P": 0b11001110,
    "U": 0b01111100,
}


@dataclass
class DisplayState:
    """数码管显示状态。"""

    mode: str = "idle"  # "idle" | "recording" | "level" | "error" | "custom"
    value: int = 0       # 显示数值
    text: str = "----"   # 自定义文本 (最多4字符)
    brightness: int = 8  # 亮度 (0-15)


def format_display_command(state: DisplayState) -> dict:
    """格式化数码管显示命令 (发往 ESP32)。

    ESP32 固件中的显示规则：
    - idle: 显示 ----
    - recording: 显示录音秒数 (如 "  5s")
    - level: 显示风扇档位 (如 "F  3")
    - error: 显示 4位全亮闪烁
    - custom: 显示自定义文本
    """
    if state.mode == "recording":
        # 录音中显示秒数
        text = f"{' ' * max(0, 3 - len(str(state.value)))}{state.value}s"
    elif state.mode == "level":
        # 风扇档位显示
        text = f"F{' ' * (3 - len(str(state.value)))}{state.value}"
    elif state.mode == "error":
        text = "8888"
    elif state.mode == "custom":
        text = state.text[:4].ljust(4)
    else:
        text = "----"

    return {
        "event": "display",
        "text": text,
        "brightness": state.brightness,
    }


def format_oled_command(
    lines: Optional[list[str]] = None,
    show_loading: bool = False,
    show_recording: bool = False,
    fan_state: Optional[str] = None,
    fan_level: Optional[int] = None,
) -> dict:
    """格式化 OLED 屏幕显示命令 (发往 ESP32)。

    ESP32 SSD1306 128x64 显示布局：
    ┌──────────────────┐
    │ 📅 日期/时间      │  ← 第 1 行
    │ 🔊 录音状态 + VU  │  ← 第 2 行
    │ 📶 WiFi 信号     │  ← 第 3 行
    │ 🌀 风扇状态+转速  │  ← 第 4 行
    └──────────────────┘
    """
    if lines:
        return {"event": "oled", "lines": lines}

    payload: dict = {"event": "oled"}
    if show_loading:
        payload["loading"] = True
    if show_recording:
        payload["recording"] = True
    if fan_state:
        payload["fan"] = fan_state
    if fan_level is not None:
        payload["fan_level"] = fan_level
    return payload
