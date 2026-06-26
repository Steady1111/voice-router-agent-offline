"""ESP32 WebSocket 桥接 — 管理 ESP32 设备连接与指令广播。

从 voice-router-agent/server/transport/ws_audio.py 移植核心逻辑：
- 维护 ESP32 会话列表
- 风扇指令广播 (broadcast_to_esp32)
- 状态同步 (sync_device_state_to_esp32)
- 显示/OLED 指令下发

数据类定义:
  FanCommand:     风扇控制 (开关/档位/PWM)
  DisplayCommand: 数码管显示 (文本/亮度)
  OledCommand:    OLED 屏幕 (行文本/状态图标)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Awaitable, Callable

from voice_router_lite.hardware.fan import SPEED_LEVELS, speed_level_to_pwm
from voice_router_lite.hardware.display import DisplayState, format_display_command, format_oled_command

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  命令数据类
# ------------------------------------------------------------------

@dataclass
class FanCommand:
    """风扇控制命令。"""
    action: str = "off"       # "on" | "off" | "set_speed"
    speed_level: int = 3      # 1-5
    speed: int = 178          # PWM 0-255

    def to_payload(self) -> dict:
        return {
            "event": "fan",
            "action": self.action,
            "speed": self.speed,
            "speed_level": self.speed_level,
        }


@dataclass
class DisplayCommand:
    """数码管显示命令。"""
    mode: str = "idle"
    value: int = 0
    text: str = "----"
    brightness: int = 8

    def to_payload(self) -> dict:
        return format_display_command(DisplayState(
            mode=self.mode,
            value=self.value,
            text=self.text,
            brightness=self.brightness,
        ))


@dataclass
class OledCommand:
    """OLED 显示命令。"""
    lines: Optional[list[str]] = None
    fan_state: Optional[str] = None
    fan_level: Optional[int] = None
    loading: bool = False
    recording: bool = False

    def to_payload(self) -> dict:
        return format_oled_command(
            lines=self.lines,
            show_loading=self.loading,
            show_recording=self.recording,
            fan_state=self.fan_state,
            fan_level=self.fan_level,
        )


# ------------------------------------------------------------------
#  ESP32 桥接
# ------------------------------------------------------------------

# 类型别名：async 发送回调
JsonSender = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ESP32Session:
    """一个 ESP32 设备连接。"""
    client_host: str
    send_json: JsonSender
    connected_at: float = 0.0


class ESP32Bridge:
    """管理所有 ESP32 设备连接，广播控制指令。

    用法:
        bridge = ESP32Bridge()
        bridge.register(host, send_json_func)
        bridge.broadcast_fan(FanCommand(action="on", speed_level=3))
        bridge.broadcast_display(DisplayCommand(mode="level", value=3))
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ESP32Session] = []
        self._on_device_state: Optional[Callable[[str, dict], None]] = None
        self._on_audio_frame: Optional[Callable[[bytes, str], None]] = None

    # ------------------------------------------------------------------
    #  会话管理
    # ------------------------------------------------------------------

    def register(self, client_host: str, send_json: JsonSender) -> None:
        """注册 ESP32 连接（自动踢掉同 IP 旧会话）。"""
        self._sessions = [s for s in self._sessions if s.client_host != client_host]
        self._sessions.append(ESP32Session(
            client_host=client_host,
            send_json=send_json,
        ))
        logger.info("[ESP32] 设备已注册: %s (当前 %d 台)", client_host, len(self._sessions))

    def unregister(self, client_host: str) -> None:
        """注销 ESP32 连接。"""
        before = len(self._sessions)
        self._sessions = [s for s in self._sessions if s.client_host != client_host]
        logger.info("[ESP32] 设备已注销: %s (剩余 %d 台)", client_host, len(self._sessions))

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def sessions(self) -> list[ESP32Session]:
        return list(self._sessions)  # 返回副本防止并发修改

    # ------------------------------------------------------------------
    #  风扇指令广播
    # ------------------------------------------------------------------

    async def broadcast_fan(self, cmd: FanCommand) -> int:
        """广播风扇指令到所有 ESP32，返回成功发送的设备数。"""
        payload = cmd.to_payload()
        logger.info("[ESP32] 广播风扇指令: %s (目标 %d 台)", payload, len(self._sessions))
        sent = 0
        dead: list[ESP32Session] = []
        for session in self._sessions:
            try:
                await session.send_json(payload)
                sent += 1
            except Exception as e:
                logger.warning("[ESP32] 发送失败: %s → %s", session.client_host, e)
                dead.append(session)
        for s in dead:
            self.unregister(s.client_host)
        return sent

    async def sync_device_state(self, fan_on: bool, speed_level: int) -> int:
        """ESP32 重连后同步当前设备状态。"""
        speed = speed_level_to_pwm(speed_level)
        cmd = FanCommand(
            action="on" if fan_on else "off",
            speed_level=speed_level,
            speed=speed,
        )
        return await self.broadcast_fan(cmd)

    # ------------------------------------------------------------------
    #  显示/OLED 指令广播
    # ------------------------------------------------------------------

    async def broadcast_display(self, cmd: DisplayCommand) -> int:
        """广播数码管显示指令。"""
        payload = cmd.to_payload()
        logger.debug("[ESP32] 广播显示: %s", payload)
        return await self._broadcast(payload)

    async def broadcast_oled(self, cmd: OledCommand) -> int:
        """广播 OLED 显示指令。"""
        payload = cmd.to_payload()
        logger.debug("[ESP32] 广播 OLED: %s", payload)
        return await self._broadcast(payload)

    async def set_display_idle(self) -> None:
        """数码管回到空闲状态 (----)。"""
        await self.broadcast_display(DisplayCommand(mode="idle"))

    async def set_display_fan_level(self, level: int) -> None:
        """数码管显示风扇档位。"""
        await self.broadcast_display(DisplayCommand(mode="level", value=level))

    async def set_display_error(self) -> None:
        """数码管显示全亮 (错误状态)。"""
        await self.broadcast_display(DisplayCommand(mode="error"))

    async def update_oled_fan_status(self, state: str, level: int) -> None:
        """更新 OLED 风扇状态栏。"""
        await self.broadcast_oled(OledCommand(fan_state=state, fan_level=level))

    # ------------------------------------------------------------------
    #  回调
    # ------------------------------------------------------------------

    def on_device_state_change(self, callback: Callable[[str, dict], None]) -> None:
        """注册设备状态变化回调（ESP32 上报状态时调用）。"""
        self._on_device_state = callback

    def on_audio_frame(self, callback: Callable[[bytes, str], None]) -> None:
        """注册音频帧回调（ESP32 上报 PCM 时调用）。"""
        self._on_audio_frame = callback

    # ------------------------------------------------------------------
    #  内部
    # ------------------------------------------------------------------

    async def _broadcast(self, payload: dict) -> int:
        sent = 0
        dead: list[ESP32Session] = []
        for session in self._sessions:
            try:
                await session.send_json(payload)
                sent += 1
            except Exception:
                dead.append(session)
        for s in dead:
            self.unregister(s.client_host)
        return sent
