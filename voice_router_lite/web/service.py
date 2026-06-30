"""Small web-facing command service.

This module intentionally avoids the heavy audio pipeline. The web console's
text path only needs NLU and device control, so it stays usable on thin-device
development targets without opening microphones, ASR models, or TTS outputs.

ESP32 桥接: 通过 esp32_bridge 实现风扇/显示/OLED 的硬件控制透传。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from voice_router_lite.config import DEFAULT_CONFIG, PipelineConfig
from voice_router_lite.device import DeviceManager, create_default_device_manager
from voice_router_lite.hardware.esp32 import ESP32Bridge
from voice_router_lite.nlu import NLUEngine
from voice_router_lite.nlu.engine import NLUResult


def _repair_voice_command(text: str) -> str:
    """修正 ASR 常见误识别，便于 NLU 命中风扇/路由器等指令。"""
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
        "封山": "风扇",
        "封善": "风扇",
        "封生": "风扇",
        "开封": "打开风扇",
        "关闭封山": "关闭风扇",
        "关闭封善": "关闭风扇",
        "关闭封生": "关闭风扇",
        "官兵电风扇": "关闭风扇",
        "官兵风扇": "关闭风扇",
        # 路由器 ASR 误听
        "wifi秘密": "WiFi密码",
        "WIFI秘密": "WiFi密码",
        "微费密码": "WiFi密码",
        "微费秘密": "WiFi密码",
        "威飞密码": "WiFi密码",
        "崇启": "重启",
        "崇启路由器": "重启路由器",
        "充启路由器": "重启路由器",
        "冲启路由器": "重启路由器",
        "陆由器": "路由器",
        "陆游器": "路由器",
        "路由七": "路由器",
        "防客网络": "访客网络",
        "防客": "访客",
        "水连了wifi": "谁连了WiFi",
        "水连了网络": "谁连了网络",
        # 网页按住说话 — 2026-06-29 实测误听
        "重启路游戏": "重启路由器",
        "密码多少": "WiFi密码多少",
        "外卖密码多少": "WiFi密码多少",
        "外卖密码": "WiFi密码",
        "晚上密码": "WiFi密码",
        "晚上密码多少": "WiFi密码多少",
        "大开房客网络": "打开访客网络",
        "重启外犯": "重启WiFi",
        "重洗录": "重启路由",
        "重洗路由器": "重启路由器",
        "录新状态": "路由状态",
        "关闭缝": "关闭风扇",
        "关闭封": "关闭风扇",
        "大开吧": "打开风扇",
        # 47 条路由器朗读实录 (2026-06-29)
        "崇喜网络": "重启网络",
        "落又期袭击了重启一下": "路由器死机了重启一下",
        "重启歪饭": "重启WiFi",
        "打开歪歪": "打开WiFi",
        "关闭外发": "关闭WiFi",
        "也连不上重启一下": "WiFi连不上重启一下",
        "麻烦密码多少": "WiFi密码多少",
        "改外犯密码": "改WiFi密码",
        "外犯改革名字": "WiFi改个名字",
        "打开房客网络": "打开访客网络",
        "切换到五级频段": "切换到5G频段",
        "指示多少": "IP地址是多少",
        "十一连的外犯": "谁连了WiFi",
        "略有期运行多久了": "路由器运行多久了",
        "内存上多少": "内存剩多少",
        "皮油附载多少": "CPU负载多少",
        "关闭录系灯": "关闭路由器灯",
        "打开洛器灯": "打开路由器灯",
        "提掉这个设备": "踢掉这个设备",
        "聘一下": "Ping一下",
        "电台是正常吗": "DNS正常吗",
        "检查顾健更新": "检查固件更新",
        "配置": "备份配置",
        "查看系统日制": "查看系统日志",
        "设定定时重启": "设置定时重启",
        "打开课玩死": "打开QoS",
        "关闭科外": "关闭QoS",
        "这台电脑的网速": "限制这台电脑的网速",
        "打开奖场控制": "打开家长控制",
        "打开微偏方": "打开VPN",
        "寸网": "防蹭网",
        "水平要负载多多": "CPU负载多少",
        "外犯密码": "WiFi密码多少",
        "同起落": "重启路由器",
        "同启落": "重启路由器",
        "重洗落": "重启路由器",
    }
    if cleaned in exact:
        return exact[cleaned]
    router_ctx = _is_likely_router_command(cleaned)
    for wrong, right in (
        ("电风扇", "风扇"),
        ("电扇", "风扇"),
        ("分扇", "风扇"),
        ("风散", "风扇"),
        ("丰扇", "风扇"),
        ("封山", "风扇"),
        ("封善", "风扇"),
        ("封生", "风扇"),
        ("烽扇", "风扇"),
        ("打开分", "打开风"),
        ("开分扇", "开风扇"),
        ("崇启", "重启"),
        ("充启", "重启"),
        ("冲启", "重启"),
        ("陆由", "路由"),
        ("陆游", "路由"),
        ("防客", "访客"),
        ("微费", "WiFi"),
        ("威飞", "WiFi"),
        ("wifi秘密", "WiFi密码"),
        ("无线秘密", "无线密码"),
        ("WIFI秘密", "WiFi密码"),
        ("路游戏", "路由器"),
        ("外卖", "WiFi"),
        ("房客", "访客"),
        ("外犯", "WiFi"),
        ("大开", "打开"),
        ("重洗", "重启"),
        ("录新", "路由"),
        ("崇喜", "重启"),
        ("落又期", "路由器"),
        ("袭击", "死机"),
        ("歪饭", "WiFi"),
        ("歪歪", "WiFi"),
        ("外发", "WiFi"),
        ("麻烦", "WiFi"),
        ("五级", "5G"),
        ("指示", "IP地址"),
        ("十一连的", "谁连了"),
        ("略有期", "路由器"),
        ("内存上", "内存剩"),
        ("皮油", "CPU"),
        ("附载", "负载"),
        ("录系", "路由器"),
        ("洛器", "路由器"),
        ("提掉", "踢掉"),
        ("聘", "Ping"),
        ("电台", "DNS"),
        ("顾健", "固件"),
        ("日制", "日志"),
        ("设定", "设置"),
        ("课玩死", "QoS"),
        ("科外", "QoS"),
        ("奖场", "家长"),
        ("微偏", "VPN"),
        ("寸网", "防蹭网"),
    ):
        cleaned = cleaned.replace(wrong, right)
    if not router_ctx:
        for wrong, right in (
            ("官兵", "关闭"),
            ("官扇", "关"),
            ("观扇", "关"),
            ("光扇", "关"),
            ("管扇", "关"),
        ):
            cleaned = cleaned.replace(wrong, right)
    if router_ctx:
        cleaned = cleaned.replace("秘密", "密码")
    return cleaned


def _mentions_fan(text: str) -> bool:
    if "风扇" in text or "风机" in text:
        return True
    return any(h in text for h in (
        "封山", "封善", "封生", "封了", "丰扇", "风散", "分扇", "风山",
    ))


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
    "打开封了": "打开风扇",
    "打扇": "打开风扇",
    "晓婷打扇": "打开风扇",
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
    "关闭": "关闭风扇",
    "官兵": "关闭风扇",
    "官病": "关闭风扇",
    "过敏率": "关闭风扇",
    "本币一份": "关闭风扇",
    "关风扇": "关闭风扇",
    "关闭一万": "关闭风扇",
    "第一关闭站": "关闭风扇",
    "过你风": "关闭风扇",
    "李过你风": "关闭风扇",
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


def _esp32_allows_fan_control(text: str, *, fan_is_on: bool) -> bool:
    """ESP32 麦：放宽风扇指令校验（ASR 常漏掉「风扇」二字）。"""
    if _mentions_fan(text):
        return True
    if text in _FAN_ASR_MISHEARD or text in _FAN_CLOSE_ASR_MISHEARD:
        return True
    if text in _FAN_ASR_MISHEARD.values():
        return True
    if fan_is_on and text in ("关闭", "关掉", "关了", "停", "停止"):
        return True
    if fan_is_on and "关" in text and "开" not in text:
        return True
    if not fan_is_on and "关" not in text:
        if text in _FAN_ASR_MISHEARD:
            return True
        if ("开" in text or "打" in text) and any(w in text for w in ("风", "扇", "封")):
            return True
        if "打扇" in text:
            return True
    return False


def _esp32_mic_fan_override(
    text: str,
    *,
    fan_is_on: bool,
    raw_asr: str = "",
) -> NLUResult | None:
    """ESP32 麦克风专用风扇兜底（演示：ASR 误听「关闭风扇」极多）。"""
    candidates: list[str] = []
    for s in (text, raw_asr):
        s = (s or "").strip()
        if s and s not in candidates:
            candidates.append(s)

    for t in candidates:
        if t in _FAN_CLOSE_ASR_MISHEARD:
            return _fan_control_result("off", _FAN_CLOSE_ASR_MISHEARD[t], confidence=0.92)
        if fan_is_on and t in ("关闭", "关掉", "关了", "停", "停止"):
            return _fan_control_result("off", "关闭风扇", confidence=0.9)
        if "关" in t and "开" not in t:
            if fan_is_on or any(w in t for w in ("风", "扇", "万", "站", "病", "官")):
                return _fan_control_result("off", "关闭风扇", confidence=0.88)
        if "关" not in raw_asr and "关" not in t:
            if t in _FAN_ASR_MISHEARD:
                return _fan_control_result("on", _FAN_ASR_MISHEARD[t], confidence=0.88)
            if ("开" in t or "打" in t) and any(w in t for w in ("风", "扇", "封")):
                return _fan_control_result("on", "打开风扇", confidence=0.88)
            if "打扇" in t or (t.endswith("扇") and "打" in t):
                return _fan_control_result("on", "打开风扇", confidence=0.87)
            open_guess = _voice_wake_fan_guess(t, fan_is_on=fan_is_on)
            if open_guess and open_guess.slots.get("state") == "on":
                return open_guess
    return None


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
        w in t for w in ("开", "打开", "开启", "启动", "打")
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


_ROUTER_HINTS = ("路由", "网络", "wifi", "WiFi", "密码", "秘密", "防火墙", "重启", "宽带", "IP",
                 "访客", "防客", "房客", "微费", "威飞", "无线", "崇启", "充启", "冲启", "陆由", "陆游",
                 "外犯", "外发", "外卖", "路游", "歪饭", "歪歪", "麻烦", "崇喜", "聘", "ping", "Ping",
                 "电台", "DNS", "dns", "顾健", "固件", "QoS", "qos", "科外", "课玩", "VPN", "vpn",
                 "微偏", "奖场", "家长", "防蹭", "寸网", "洛器", "录系", "皮油", "CPU", "负载", "踢",
                 "提掉", "黑名单", "测速", "延迟", "诊断", "备份", "日志", "出厂", "优先", "游戏机")


def _is_likely_router_command(text: str) -> bool:
    t = text.lower()
    return any(h.lower() in t for h in _ROUTER_HINTS)


def _router_result(intent: str, text: str, slots: dict[str, str], confidence: float = 0.95) -> NLUResult:
    return NLUResult(
        text=text,
        intent=intent,
        intent_id=0,
        confidence=confidence,
        slots=slots,
        is_valid=True,
    )


def _is_password_query(text: str) -> bool:
    """查 WiFi 密码（非改密码）。"""
    t = text.strip()
    if not t or not ("密码" in t or "秘密" in t):
        return False
    if any(w in t for w in ("改", "修改", "设置", "换成", "重置")):
        return False
    if any(w in t for w in ("多少", "是什么", "啥", "查询", "查看")):
        return True
    if t in ("密码", "WiFi密码", "wifi密码", "无线密码", "WIFI密码"):
        return True
    tl = t.lower()
    if any(w in tl for w in ("wifi", "wi-fi", "无线", "微费", "威飞", "外卖", "麻烦", "外犯")):
        return True
    return len(t) <= 8


def _keyword_router_override(text: str) -> NLUResult | None:
    """关键词兜底：网页/ESP32 ASR 误听后仍落到路由器意图。"""
    t = text.strip()
    if not t or not _is_likely_router_command(t):
        return None
    tl = t.lower()

    if _is_password_query(t):
        return _router_result(
            "router_network_query", t,
            {"query_type": "wifi_password"},
        )

    if "连不上" in t and "重启" in t:
        return _router_result("router_wifi_restart", t, {"action": "restart"})

    if any(w in t for w in ("重启", "崇启", "充启", "冲启")):
        if any(w in tl for w in ("wifi", "wi-fi", "无线", "微费", "威飞", "外犯", "歪饭", "外发")):
            return _router_result("router_wifi_restart", t, {"action": "restart"})
        if any(w in t for w in ("路由", "陆由", "陆游", "网络", "路游")) or t in ("重启", "重启一下"):
            return _router_result("router_reboot", t, {"action": "reboot"})

    if ("密码" in t or "秘密" in t) and any(w in t for w in ("多少", "是什么", "啥", "查询", "查看")):
        if any(w in tl for w in ("wifi", "wi-fi", "无线", "微费", "威飞", "路由", "外卖", "麻烦")):
            return _router_result(
                "router_network_query", t,
                {"query_type": "wifi_password"},
            )
        if t in ("密码多少", "WiFi密码多少", "无线密码多少") or len(t) <= 8:
            return _router_result(
                "router_network_query", t,
                {"query_type": "wifi_password"},
            )

    if any(w in t for w in ("谁连了", "水连了", "谁连接", "连了几个", "多少设备", "连接设备", "十一连")):
        return _router_result("router_network_query", t, {"query_type": "devices"})

    if any(w in t for w in ("IP地址", "指示多少")) or t in ("IP地址是多少",):
        return _router_result("router_network_query", t, {"query_type": "ip"})

    if any(w in t for w in ("CPU负载", "皮油", "内存剩", "内存上", "运行多久", "略有期")):
        return _router_result("router_network_query", t, {"query_type": t})

    if any(w in t for w in ("路由状态", "网络状态", "运行状态")):
        return _router_result("router_network_query", t, {"query_type": "status"})

    if any(w in t for w in ("测一下网速", "测速", "网速怎么样")):
        return _router_result("router_network_diag", t, {"diag_type": "speed"})

    if any(w in tl for w in ("ping", "聘一下", "延迟多少")):
        return _router_result("router_network_diag", t, {"diag_type": "latency"})

    if any(w in t for w in ("DNS", "dns", "电台是正常吗")):
        return _router_result("router_network_diag", t, {"diag_type": "dns"})

    if "网络诊断" in t:
        return _router_result("router_network_diag", t, {"diag_type": "diagnosis"})

    if any(w in t for w in ("路由器灯", "录系灯", "洛器灯", "指示灯")):
        state = "off" if any(w in t for w in ("关", "关闭", "关掉", "熄灭")) else "on"
        return _router_result("router_led_control", t, {"state": state})

    if any(w in t for w in ("踢掉", "提掉", "拉黑", "黑名单", "取消拉黑")):
        return _router_result("router_device_manage", t, {"manage_action": t})

    if any(w in t for w in ("限制", "游戏机")) and "网速" in t:
        return _router_result("router_qos", t, {"qos_action": t})

    if "优先" in t:
        return _router_result("router_qos", t, {"qos_action": t})

    if any(w in t for w in ("检查固件", "固件", "顾健", "备份配置", "备份", "恢复出厂", "系统日志", "查看系统", "定时重启", "设定定时", "设置定时")):
        return _router_result("router_system", t, {"sys_action": t})

    if any(w in tl for w in ("qos", "课玩死", "科外")) or "限速" in t:
        return _router_result("router_qos", t, {"qos_action": t})

    if any(w in t for w in ("防火墙", "家长", "奖场", "VPN", "vpn", "微偏", "防蹭", "寸网")):
        return _router_result("router_security", t, {"security_type": t})

    if any(w in t for w in ("访客", "防客", "房客")):
        if any(w in t for w in ("打开", "开启", "启动", "开一下")):
            return _router_result(
                "router_wifi_config", t,
                {"config_type": "访客", "state": "on"},
            )
        if any(w in t for w in ("关闭", "关掉", "关")):
            return _router_result(
                "router_wifi_config", t,
                {"config_type": "访客", "state": "off"},
            )

    if any(w in tl for w in ("wifi密码", "wifi 密码", "无线密码")) and not any(
        w in t for w in ("改", "修改", "设置", "换成")
    ):
        if "多少" in t or "是什么" in t or t.endswith("密码"):
            return _router_result(
                "router_network_query", t,
                {"query_type": "wifi_password"},
            )

    if any(w in t for w in ("打开WiFi", "打开wifi", "打开无线", "关闭WiFi", "关闭wifi", "关闭无线")):
        return _router_result("router_wifi_restart", t, {"action": "restart"})

    if "重启" in t and any(w in tl for w in ("wifi", "wi-fi", "无线", "微费", "威飞")):
        return _router_result("router_wifi_restart", t, {"action": "restart"})

    return None


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
    if not voice_wake and not _mentions_fan(text) and not fan_is_on:
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
    if not _mentions_fan(text):
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
        self.esp32_temperature_c: float | None = None
        self._initialized = False
        # 持久化状态文件（解决重启后温度丢失问题）
        self._state_dir = Path(__file__).resolve().parent.parent.parent / "data"
        self._state_file = self._state_dir / "esp32_state.json"

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

        # 从持久化文件恢复温度（重启不丢）
        self._load_persisted_state()

        # 同步引擎内存估算到监控面板（仅离线引擎核心模块）
        self._sync_memory_to_monitor()

    def update_esp32_temperature(self, temp_c: float) -> None:
        """更新 ESP32 温度并持久化，避免重启丢失。"""
        self.esp32_temperature_c = round(temp_c, 1)
        try:
            from voice_router_lite.web.monitor import get_monitor
            get_monitor().update_esp32_temperature(temp_c)
        except Exception:
            pass
        self._save_persisted_state()

    # ------------------------------------------------------------------
    # 持久化（温度在重启后恢复，避免前端空白）
    # ------------------------------------------------------------------
    def _load_persisted_state(self) -> None:
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                temp = data.get("temperature_c")
                if temp is not None:
                    self.esp32_temperature_c = round(float(temp), 1)
                    logger.info("从持久化恢复温度: %.1f°C", self.esp32_temperature_c)
        except Exception:
            pass

    def _save_persisted_state(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            data = {"temperature_c": self.esp32_temperature_c}
            self._state_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _sync_memory_to_monitor(self) -> None:
        """将配置中的引擎内存估算写入监控面板（KWS 直驱，无 ASR）。"""
        try:
            from voice_router_lite.web.monitor import get_monitor
            perf = self.config.performance
            total = (
                perf.kws_model_memory_mb
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

    def handle_text(
        self,
        text: str,
        *,
        voice_wake: bool = False,
        esp32_mic: bool = False,
        raw_asr: str = "",
    ) -> dict[str, Any]:
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

        fan_is_on = bool(self.devices._fan_is_on)
        raw_for_esp32 = (raw_asr or asr_text).strip()
        allow_fan_guess = (voice_wake or _mentions_fan(clean_text)) and not esp32_mic
        esp32_fan = (
            _esp32_mic_fan_override(
                clean_text, fan_is_on=fan_is_on, raw_asr=raw_for_esp32,
            )
            if esp32_mic
            else None
        )
        keyword_result = (
            esp32_fan
            or _fan_level_command_override(clean_text)
            or _keyword_router_override(clean_text)
            or _keyword_fan_override(clean_text, fan_is_on=fan_is_on)
            or (
                _voice_wake_fan_guess(clean_text, fan_is_on=fan_is_on)
                if allow_fan_guess
                else None
            )
        )
        nlu_result = keyword_result or self.nlu.understand(clean_text)
        if (
            nlu_result.intent == "router_wifi_config"
            and "密码" in clean_text
            and _is_password_query(clean_text)
        ):
            nlu_result = _router_result(
                "router_network_query", clean_text,
                {"query_type": "wifi_password"},
            )
        nlu_result = _apply_voice_wake_fan_correction(
            clean_text,
            nlu_result,
            fan_is_on=fan_is_on,
            voice_wake=voice_wake and not esp32_mic,
            had_keyword=keyword_result is not None,
        )
        state = nlu_result.slots.get("state", "")
        if (
            nlu_result.intent == "device_control"
            and state not in ("on", "off", "打开", "开启", "开", "关闭", "关掉", "关")
        ):
            fix = (
                _keyword_router_override(clean_text)
                or _keyword_fan_override(clean_text, fan_is_on=fan_is_on)
                or _voice_wake_fan_guess(clean_text, fan_is_on=fan_is_on)
            )
            if fix is not None:
                nlu_result = fix
        elif (
            not nlu_result.is_valid
            or (
                nlu_result.intent.startswith("router_")
                and nlu_result.confidence < 0.5
            )
        ):
            fix = _keyword_router_override(clean_text)
            if fix is not None:
                nlu_result = fix

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

        esp32_min_conf = float(os.getenv("VOICE_ROUTER_ESP32_MIN_NLU_CONF", "0.55"))
        if esp32_mic and nlu_result.is_valid:
            if nlu_result.confidence < esp32_min_conf:
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
            if (
                nlu_result.intent == "device_control"
                and not _esp32_allows_fan_control(clean_text, fan_is_on=fan_is_on)
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
            if (
                nlu_result.intent.startswith("router_")
                and not _is_likely_router_command(clean_text)
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
        self.last_command = (command or clean_text or asr_text).strip()
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


def _canonical_display_command(
    intent: str,
    slots: dict[str, str],
    clean_text: str,
) -> str | None:
    """按意图返回下拉框标准话术，供界面展示（非 raw ASR）。"""
    if not intent or intent in ("unknown", "help"):
        return None
    t = clean_text.strip()
    if intent == "router_reboot":
        return "重启路由器"
    if intent == "router_wifi_restart":
        if any(w in t for w in ("关闭", "关掉", "关")):
            return "关闭WiFi"
        if any(w in t for w in ("打开", "开启", "开")):
            return "打开WiFi"
        return "重启WiFi"
    if intent == "router_wifi_config":
        if any(w in t for w in ("访客", "房客", "防客")):
            return "关闭访客网络" if any(w in t for w in ("关", "关闭")) else "打开访客网络"
        if any(w in t for w in ("5G", "5g", "五级")):
            return "切换到5G频段"
        if any(w in t for w in ("名字", "名称", "改名", "ssid")):
            return "WiFi改个名字"
        if "密码" in t and any(w in t for w in ("改", "修改", "设置", "换")):
            return "改WiFi密码"
        if "密码" in t or "秘密" in t:
            return "WiFi密码多少"
        return "改WiFi密码"
    if intent == "router_network_query":
        qt = str(slots.get("query_type", "")).lower()
        if qt in ("wifi_password", "password", "wifi") or ("密码" in t and "多少" in t):
            return "WiFi密码多少"
        if qt == "devices" or any(w in t for w in ("谁连", "连了几个", "设备")):
            return "谁连了WiFi" if "wifi" in t.lower() or "WiFi" in t else "连了几个设备"
        if qt == "ip" or "IP" in t or "地址" in t:
            return "IP地址是多少"
        if qt == "speed" or "网速" in t:
            return "网速怎么样"
        if "内存" in t:
            return "内存剩多少"
        if "cpu" in t.lower() or "负载" in t:
            return "CPU负载多少"
        if "多久" in t or "运行" in t:
            return "路由器运行多久了"
        return "网速怎么样"
    if intent == "router_led_control":
        return "关闭路由器灯" if any(w in t for w in ("关", "关闭", "熄灭")) else "打开路由器灯"
    if intent == "router_device_manage":
        if "取消" in t or "解除" in t:
            return "取消拉黑这个设备"
        if "黑名单" in t or "查看" in t:
            return "查看黑名单"
        if "手机" in t:
            return "拉黑这部手机"
        return "踢掉这个设备"
    if intent == "router_network_diag":
        if "dns" in t.lower() or "DNS" in t or "电台" in t:
            return "DNS正常吗"
        if "ping" in t.lower() or "聘" in t:
            return "Ping一下"
        if "延迟" in t:
            return "延迟多少"
        if "诊断" in t:
            return "网络诊断"
        return "测一下网速"
    if intent == "router_system":
        if "固件" in t or "顾健" in t or "更新" in t:
            return "检查固件更新"
        if "备份" in t or t == "配置":
            return "备份配置"
        if "出厂" in t or "恢复" in t:
            return "恢复出厂设置"
        if "日志" in t or "日制" in t:
            return "查看系统日志"
        return "设置定时重启"
    if intent == "router_qos":
        if any(w in t for w in ("关闭", "关掉", "关")):
            return "关闭QoS"
        if "优先" in t or "游戏机" in t:
            return "给游戏机优先"
        if "电脑" in t or "限制" in t or "网速" in t:
            return "限制这台电脑的网速"
        return "打开QoS"
    if intent == "router_security":
        if "vpn" in t.lower() or "VPN" in t or "微偏" in t:
            return "打开VPN"
        if "家长" in t or "奖场" in t:
            return "打开家长控制"
        if "蹭" in t or "寸网" in t:
            return "防蹭网"
        return "关闭防火墙" if any(w in t for w in ("关", "关闭")) else "打开防火墙"
    if intent in ("device_control", "set_device_state"):
        device = {"fan": "风扇", "led": "LED", "light": "灯光"}.get(
            slots.get("device_type", ""), slots.get("device_type", "设备"),
        )
        if slots.get("state") in ("off", "关闭", "关掉", "关") or any(
            w in t for w in ("关", "关闭")
        ):
            return f"关闭{device}"
        return f"打开{device}"
    return None


def _resolve_display_command(
    asr_text: str,
    clean_text: str,
    nlu_result: NLUResult,
    *,
    success: bool,
) -> str:
    """生成前端展示的指令文本（规范话术，非 raw ASR）。"""
    slots = nlu_result.slots or {}
    intent = nlu_result.intent

    canonical = _canonical_display_command(intent, slots, clean_text)
    if canonical:
        return canonical

    if clean_text and clean_text != asr_text:
        return clean_text

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
