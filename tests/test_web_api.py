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
