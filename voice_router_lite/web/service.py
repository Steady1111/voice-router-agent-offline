"""Small web-facing command service.

This module intentionally avoids the heavy audio pipeline. The web console's
text path only needs NLU and device control, so it stays usable on thin-device
development targets without opening microphones, ASR models, or TTS outputs.

ESP32 桥接: 通过 esp32_bridge 实现风扇/显示/OLED 的硬件控制透传。
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Optional

from voice_router_lite.config import DEFAULT_CONFIG, PipelineConfig
from voice_router_lite.device import DeviceManager, create_default_device_manager
from voice_router_lite.hardware.esp32 import ESP32Bridge
from voice_router_lite.nlu import NLUEngine
from voice_router_lite.nlu.engine import NLUResult


def _repair_voice_command(text: str) -> str:
    """修正 ASR 常见误识别，便于 NLU 命中风扇等指令。"""
    cleaned = text.strip()
    exact = {
        "拍风声": "打开风扇",
        "打风扇": "打开风扇",
        "大风扇": "打开风扇",
        "打开风": "打开风扇",
        "管扇": "关风扇",
        "光扇": "关风扇",
        "观扇": "关风扇",
        "官扇": "关风扇",
        "官兵": "关风扇",
        "过敏率": "关风扇",
        "本币一份": "关风扇",
        "关扇": "关风扇",
        "风山": "风扇",
        "开封": "打开风扇",
    }
    if cleaned in exact:
        return exact[cleaned]
    for wrong, right in (
        ("分扇", "风扇"),
        ("风散", "风扇"),
        ("丰扇", "风扇"),
        ("打开分", "打开风"),
        ("开分扇", "开风扇"),
    ):
        cleaned = cleaned.replace(wrong, right)
    return cleaned


# ESP32 麦克风 ASR 常见误识别 → 打开风扇（预研兜底）
_FAN_ASR_MISHEARD = {
    "开封": "打开风扇",
    "开风": "打开风扇",
    "开扇": "打开风扇",
    "打开": "打开风扇",
    "当然": "打开风扇",
    "开窗": "打开风扇",
    "开放": "打开风扇",
    "开方": "打开风扇",
    "分扇": "打开风扇",
    "风散": "打开风扇",
    "大功率": "打开风扇",
    "拍风声": "打开风扇",
    "打风扇": "打开风扇",
}

# ASR 误识别 → 关闭风扇
_FAN_CLOSE_ASR_MISHEARD = {
    "管扇": "关闭风扇",
    "光扇": "关闭风扇",
    "观扇": "关闭风扇",
    "官扇": "关闭风扇",
    "关风": "关闭风扇",
    "关扇": "关闭风扇",
    "关了": "关闭风扇",
    "关掉": "关闭风扇",
    "官兵": "关闭风扇",
    "过敏率": "关闭风扇",
    "本币一份": "关闭风扇",
    "关风扇": "关闭风扇",
}


def _fan_control_result(state: str, text: str, confidence: float = 0.9) -> NLUResult:
    return NLUResult(
        text=text,
        intent="device_control",
        intent_id=0,
        confidence=confidence,
        slots={"device_type": "fan", "state": state},
        is_valid=True,
    )


def _voice_wake_fan_guess(text: str, *, fan_is_on: bool = False) -> NLUResult | None:
    """唤醒后语音指令：ASR 误识别时尽量落到风扇控制。"""
    t = text.strip()
    if not t:
        return None
    if t in _FAN_CLOSE_ASR_MISHEARD:
        return _fan_control_result("off", _FAN_CLOSE_ASR_MISHEARD[t])
    if t in _FAN_ASR_MISHEARD:
        t = _FAN_ASR_MISHEARD[t]
    if "关" in t and any(w in t for w in ("风扇", "风", "扇")):
        return _fan_control_result("off", t)
    if any(w in t for w in ("风扇", "风机", "风", "扇")) and any(
        w in t for w in ("开", "打开", "开启", "启动")
    ):
        return _fan_control_result("on", t)
    if t in ("风扇", "风机"):
        return _fan_control_result("off" if fan_is_on else "on", t)
    if t in _FAN_ASR_MISHEARD.values() or (
        len(t) <= 6 and "开" in t and "关" not in t
    ):
        return _fan_control_result("on", "打开风扇", confidence=0.85)
    # 风扇已开时，短句里带「风/扇」且不含「开」→ 倾向关风扇（ASR 常漏掉「关」）
    if fan_is_on and len(t) <= 4 and ("风" in t or "扇" in t) and "开" not in t:
        return _fan_control_result("off", "关闭风扇", confidence=0.8)
    # 风扇关着时，短句里带「风/扇」且不含「关」→ 倾向开风扇（如「拍风声」）
    if not fan_is_on and len(t) <= 5 and ("风" in t or "扇" in t) and "关" not in t:
        return _fan_control_result("on", "打开风扇", confidence=0.75)
    return None


_ROUTER_HINTS = ("路由", "网络", "wifi", "WiFi", "密码", "防火墙", "重启", "宽带", "IP")


def _is_likely_router_command(text: str) -> bool:
    t = text.lower()
    return any(h.lower() in t for h in _ROUTER_HINTS)


def _looks_like_open_command(text: str) -> bool:
    return any(w in text for w in ("开", "打开", "开启", "启动", "开一下"))


def _apply_voice_wake_fan_correction(
    text: str,
    nlu_result: NLUResult,
    *,
    fan_is_on: bool,
    voice_wake: bool,
    had_keyword: bool,
) -> NLUResult:
    """唤醒后 ASR 乱识别时，尽量落到风扇开关（尤其风扇已开 → 关）。"""
    if not voice_wake or had_keyword:
        return nlu_result
    if _is_likely_router_command(text):
        return nlu_result

    guessed = _voice_wake_fan_guess(text, fan_is_on=fan_is_on)
    if guessed is not None:
        if (
            not nlu_result.is_valid
            or nlu_result.intent.startswith("router_")
            or nlu_result.intent in ("help", "unknown")
            or (
                nlu_result.intent == "device_control"
                and guessed.slots.get("state") == "off"
                and fan_is_on
            )
        ):
            return guessed
        return nlu_result

    # 风扇已开：短句且不像「打开」→ 默认关风扇（ESP32 麦 ASR 常把「关闭风扇」听成胡话）
    if (
        fan_is_on
        and len(text) <= 8
        and not _looks_like_open_command(text)
        and (
            nlu_result.intent.startswith("router_")
            or not nlu_result.is_valid
            or nlu_result.intent in ("help", "unknown")
            or nlu_result.intent == "device_control"
        )
    ):
        return _fan_control_result("off", "关闭风扇", confidence=0.65)
    return nlu_result


def _keyword_fan_override(text: str, *, fan_is_on: bool = False) -> NLUResult | None:
    """关键词兜底：文本含「风扇+开/关」时强制 device_control。"""
    if "风扇" not in text and "风机" not in text:
        return None
    # 含档位/调速语义时交给 _fan_level_command_override
    if any(w in text for w in ("档", "调到", "调大", "调小", "最低", "最高", "风速")):
        return None
    if any(w in text for w in ("打开", "开启", "启动", "开一下")) or text.endswith("开"):
        return _fan_control_result("on", text, confidence=0.95)
    if any(w in text for w in ("关闭", "关掉", "停止")) or (
        "关" in text and any(w in text for w in ("风扇", "风机", "风", "扇"))
    ):
        return _fan_control_result("off", text, confidence=0.95)
    if text in ("风扇", "风机", "关风扇", "关风机"):
        return _fan_control_result("off" if fan_is_on else "on", text, confidence=0.9)
    return None


_CN_LEVEL = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}


def _fan_level_command_override(text: str) -> NLUResult | None:
    """关键词兜底：风扇档位调节（NLU 模型常把「调到N档」解析错）。"""
    if "风扇" not in text and "风机" not in text:
        return None
    if not any(w in text for w in ("档", "调到", "调大", "调小", "最低", "最高", "风速", "微风", "全速")):
        return None

    import re

    def _adjust(slots: dict[str, str]) -> NLUResult:
        return NLUResult(
            text=text,
            intent="device_adjust",
            intent_id=0,
            confidence=0.95,
            slots=slots,
            is_valid=True,
        )

    if any(w in text for w in ("最低", "最慢", "微风", "最小")):
        return _adjust({"device_type": "fan", "direction": "min"})
    if any(w in text for w in ("最高", "最快", "全速", "最大")):
        return _adjust({"device_type": "fan", "direction": "max"})
    if any(w in text for w in ("调大", "大一点", "加大", "快一点", "调快")):
        return _adjust({"device_type": "fan", "direction": "up"})
    if any(w in text for w in ("调小", "小一点", "减小", "慢一点", "调慢")):
        return _adjust({"device_type": "fan", "direction": "down"})

    level_match = re.search(r"(\d+)\s*档", text)
    if level_match:
        return _adjust({
            "device_type": "fan",
            "direction": "set",
            "level": level_match.group(1),
        })

    for cn, num in _CN_LEVEL.items():
        if f"{cn}档" in text:
            return _adjust({
                "device_type": "fan",
                "direction": "set",
                "level": str(num),
            })

    return None


class WebCommandService:
    """Owns the lightweight NLU -> device-command path for the web console."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.nlu = NLUEngine(
            self.config.models,
            confidence_threshold=self.config.nlu_confidence_threshold,
        )
        self.esp32_bridge = ESP32Bridge()
        self.devices: DeviceManager = create_default_device_manager(
            esp32_bridge=self.esp32_bridge,
        )
        self.last_command: str = ""
        self.last_ack_at: str | None = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        prefer_int8 = (
            self.config.prefer_int8_nlu
            or os.getenv("VOICE_ROUTER_PREFER_INT8_NLU", "").lower() in {"1", "true", "yes"}
        )
        self.nlu.initialize(prefer_int8=prefer_int8)
        self.devices.initialize()
        self._initialized = True

        # 同步引擎内存估算到监控面板（仅离线引擎核心模块）
        self._sync_memory_to_monitor()

    def _sync_memory_to_monitor(self) -> None:
        """将配置中的引擎内存估算写入监控面板。"""
        try:
            from voice_router_lite.web.monitor import get_monitor
            perf = self.config.performance
            total = (
                perf.kws_model_memory_mb
                + perf.asr_model_memory_mb
                + perf.nlu_model_memory_mb
                + perf.audio_buffer_memory_mb
            )
            get_monitor().update_engine_memory(float(total))
        except Exception:
            pass

    def close(self) -> None:
        if self._initialized:
            self.devices.close()
        self._initialized = False

    def list_devices(self, *, include_mock: bool = False) -> list[dict[str, Any]]:
        self.initialize()
        items = [
            _device_state_to_api(device_id, state)
            for device_id, state in self.devices.get_all_states().items()
        ]
        if include_mock:
            if not any(d.get("type") == "router" for d in items):
                items.append({
                    "id": "router",
                    "name": "路由器",
                    "type": "router",
                    "state": "on",
                    "level": None,
                    "updated_at": None,
                })
            return items
        items = [d for d in items if d.get("type") in ("fan", "router")]
        if not any(d.get("type") == "router" for d in items):
            items.append({
                "id": "router",
                "name": "路由器",
                "type": "router",
                "state": "on",
                "level": None,
                "updated_at": None,
            })
        return items

    def handle_text(self, text: str, *, voice_wake: bool = False) -> dict[str, Any]:
        self.initialize()
        asr_text = text.strip()
        clean_text = _repair_voice_command(asr_text)
        if not clean_text:
            return {
                "success": False,
                "asr_text": asr_text,
                "transcript": "",
                "command": "",
                "reply": "请输入指令。",
                "intent": "unknown",
                "slots": {},
            }

        fan_is_on = bool(self.devices._fan_is_on) if voice_wake else False
        keyword_result = (
            _fan_level_command_override(clean_text)
            or _keyword_fan_override(clean_text, fan_is_on=fan_is_on)
            or (
                _voice_wake_fan_guess(clean_text, fan_is_on=fan_is_on)
                if voice_wake
                else None
            )
        )
        nlu_result = keyword_result or self.nlu.understand(clean_text)
        nlu_result = _apply_voice_wake_fan_correction(
            clean_text,
            nlu_result,
            fan_is_on=fan_is_on,
            voice_wake=voice_wake,
            had_keyword=keyword_result is not None,
        )

        strict_router = (
            self.config.deployment_mode == "router"
            or os.getenv("VOICE_ROUTER_STRICT_ROUTER_NLU", "").lower() in {"1", "true", "yes"}
        )
        if (
            strict_router
            and nlu_result.intent.startswith("router_")
            and nlu_result.confidence < self.config.nlu_confidence_threshold
        ):
            return {
                "success": False,
                "asr_text": asr_text,
                "transcript": clean_text,
                "command": clean_text,
                "corrected": clean_text != asr_text,
                "reply": "没听清，请再说一次",
                "intent": nlu_result.intent,
                "confidence": nlu_result.confidence,
                "slots": nlu_result.slots,
            }

        if nlu_result.is_valid:
            exec_result = self.devices.execute_command(
                nlu_result.intent,
                nlu_result.slots,
            )
        else:
            exec_result = {"success": False, "message": "无法理解指令"}

        success = bool(exec_result.get("success", False))
        command = _resolve_display_command(
            asr_text, clean_text, nlu_result, success=success,
        )
        reply = _format_reply(clean_text, nlu_result.intent, nlu_result.slots, exec_result)
        if success and command:
            self.last_command = command
        return {
            "success": success,
            "asr_text": asr_text,
            "transcript": clean_text,
            "command": command,
            "corrected": command != asr_text,
            "reply": reply,
            "intent": nlu_result.intent,
            "confidence": nlu_result.confidence,
            "slots": nlu_result.slots,
            "device_state": _maybe_device_state(exec_result.get("device_state")),
        }


def _resolve_display_command(
    asr_text: str,
    clean_text: str,
    nlu_result: NLUResult,
    *,
    success: bool,
) -> str:
    """生成前端展示的指令文本（与最终执行语义一致）。"""
    if not success:
        return clean_text or asr_text

    slots = nlu_result.slots
    intent = nlu_result.intent
    device_names = {
        "fan": "风扇", "led": "LED", "light": "灯光", "relay": "继电器", "ac": "空调",
    }

    if intent in ("device_control", "set_device_state"):
        device = device_names.get(slots.get("device_type", ""), slots.get("device_type", "设备"))
        state = slots.get("state", "")
        if state in ("on", "打开", "开启", "开"):
            return f"打开{device}"
        if state in ("off", "关闭", "关掉", "关"):
            return f"关闭{device}"

    if intent == "device_adjust":
        device = device_names.get(slots.get("device_type", "fan"), "风扇")
        level = slots.get("level")
        if level:
            return f"{device}调到{level}档"
        direction = slots.get("direction", "")
        if direction:
            return f"{device}{direction}"

    resolved = (nlu_result.text or "").strip()
    if resolved and len(resolved) >= 2:
        return resolved

    return clean_text or asr_text


def _device_state_to_api(device_id: str, state: Any) -> dict[str, Any]:
    return {
        "id": state.device_type or device_id,
        "name": state.name or device_id,
        "type": state.device_type,
        "state": "on" if state.is_on else "off",
        "level": state.level,
        "updated_at": None,
        **state.extra,
    }


def _maybe_device_state(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    try:
        return asdict(state)
    except TypeError:
        return None


def _format_reply(
    text: str,
    intent: str,
    slots: dict[str, str],
    exec_result: dict[str, Any],
) -> str:
    if not exec_result.get("success"):
        return exec_result.get("message", "无法理解指令")

    # 设备控制类意图
    if intent in ("device_control", "set_device_state"):
        device_state = exec_result.get("device_state")
        name = getattr(device_state, "name", slots.get("device_type", "设备"))
        state = slots.get("state", "")
        if state in ("on", "打开"):
            return f"已打开{name}"
        if state in ("off", "关闭"):
            return f"已关闭{name}"

    # 优先使用 manager 返回的 message
    return exec_result.get("message") or f"已执行：{text}"
