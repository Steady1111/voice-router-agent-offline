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


def get_router_status(service: WebCommandService) -> dict[str, Any]:
    """Build GET /api/router-status payload (pre-research mock + monitor)."""
    snap = get_monitor().get_snapshot()
    mem = snap.get("memory") or {}
    ram_used = float(mem.get("rss_mb") or mem.get("used_mb") or 0)
    ram_total = int(mem.get("total_mb") or 128)
    cpu_pct = snap.get("cpu_percent")

    return {
        "wan_up": True,
        "lan_clients": 0,
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
        "last_ack_at": last_ack,
        "last_command": service.last_command or "",
    }
