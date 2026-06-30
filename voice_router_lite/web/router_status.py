"""Router status snapshot for web console (task 1D)."""

from __future__ import annotations

import os
import time
from typing import Any

from voice_router_lite.web.monitor import get_monitor
from voice_router_lite.web.service import WebCommandService


def _daemon_state() -> str:
    mode = os.getenv("VOICE_ROUTER_DEPLOYMENT_MODE", "web").lower()
    if mode == "web":
        return "web"
    # Future: read /tmp/voice-router-state.json from voice-routerd
    return "running"


def _lan_client_count(service: WebCommandService) -> int | None:
    """下联设备数。量产 ubus；Mac 预研与 NLU 返回统一数据。"""
    try:
        from voice_router_lite.platform.openwrt import get_ubus_client

        ubus = get_ubus_client()
        if ubus.available:
            # TODO: OpenWrt DHCP 租约计数（量产接入）
            return None
    except Exception:
        pass

    # Mac 预研：与 NLU「连接设备」使用同一数据源
    from voice_router_lite.device.manager import DeviceManager
    info = DeviceManager._get_router_info()
    return int(info.get("connected_devices", 0))


def get_router_status(service: WebCommandService) -> dict[str, Any]:
    """Build GET /api/router-status payload (pre-research mock + monitor)."""
    snap = get_monitor().get_snapshot()
    ram_used = float(snap.get("current_ram_mb") or 0)
    ram_total = int(snap.get("target_ram_mb") or 128)
    cpu_pct = snap.get("cpu_percent")

    return {
        "wan_up": True,
        "lan_clients": _lan_client_count(service),
        "voice_daemon": _daemon_state(),
        "ram_used_mb": round(ram_used, 1),
        "ram_total_mb": ram_total,
        "cpu_pct": round(float(cpu_pct), 1) if cpu_pct is not None else None,
        "last_command": service.last_command or "",
        "updated_at": time.time(),
    }


def get_esp32_status_extended(service: WebCommandService) -> dict[str, Any]:
    bridge = service.esp32_bridge
    if bridge is None:
        return {
            "connected": False,
            "count": 0,
            "sessions": [],
            "fan": {"on": False, "level": 0},
            "last_ack_at": None,
            "last_command": service.last_command or "",
        }

    sessions = bridge.sessions
    mgr = service.devices
    last_ack = service.last_ack_at
    return {
        "connected": len(sessions) > 0,
        "count": len(sessions),
        "sessions": [{"host": s.client_host} for s in sessions],
        "fan": {
            "on": bool(mgr._fan_is_on),
            "level": int(mgr._fan_speed_level),
        },
        "temperature_c": service.esp32_temperature_c,
        "last_ack_at": last_ack,
        "last_command": service.last_command or "",
    }
