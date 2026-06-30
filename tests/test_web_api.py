"""Web API routes for OpenSpec 1D."""

from __future__ import annotations

from fastapi.testclient import TestClient

from voice_router_lite.web.app import create_app


def test_router_status_endpoint():
    with TestClient(create_app()) as client:
        resp = client.get("/api/router-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "wan_up" in data
        assert "voice_daemon" in data
        assert "ram_used_mb" in data


def test_devices_default_fan_and_router_only():
    with TestClient(create_app()) as client:
        resp = client.get("/api/devices")
        assert resp.status_code == 200
        types = {d["type"] for d in resp.json()}
        assert "fan" in types
        assert "router" in types
        assert "led" not in types


def test_devices_debug_includes_mock():
    with TestClient(create_app()) as client:
        resp = client.get("/api/devices?debug=true")
        types = {d["type"] for d in resp.json()}
        assert "led" in types


def test_esp32_status_extended_fields():
    with TestClient(create_app()) as client:
        resp = client.get("/api/esp32-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "fan" in data
        assert "last_command" in data
        assert "last_ack_at" in data


def test_web_voice_router_asr_repair():
    """网页按住说话 / 47条朗读实录误听 → repair + keyword 兜底。"""
    from voice_router_lite.web.service import WebCommandService

    svc = WebCommandService()
    cases = [
        ("重启路游戏", "router_reboot", True),
        ("密码多少", "router_network_query", True),
        ("外卖密码多少", "router_network_query", True),
        ("大开房客网络", "router_wifi_config", True),
        ("录新状态", "router_network_query", True),
        ("重启外犯", "router_wifi_restart", True),
        ("关闭缝", "device_control", True),
        # 47条朗读实录抽样
        ("重启歪饭", "router_wifi_restart", True),
        ("麻烦密码多少", "router_network_query", True),
        ("十一连的外犯", "router_network_query", True),
        ("打开课玩死", "router_qos", True),
        ("打开微偏方", "router_security", True),
        ("寸网", "router_security", True),
        ("同起落", "router_reboot", True),
        ("密码", "router_network_query", True),
    ]
    for raw, intent, success in cases:
        result = svc.handle_text(raw)
        assert result["success"] is success, raw
        assert result["intent"] == intent, f"{raw} -> {result['intent']}"
        assert result["command"] != raw or raw in ("给游戏机优先",), raw
