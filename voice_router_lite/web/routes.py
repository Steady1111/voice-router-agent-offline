"""HTTP routes for the lightweight web console."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, WebSocket

from voice_router_lite.web.monitor import get_monitor
from voice_router_lite.web.ws_audio import run_audio_session

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/devices")
async def list_devices(request: Request) -> list[dict]:
    return request.app.state.command_service.list_devices()


@router.get("/api/network")
async def network_status() -> dict:
    return {
        "online": True,
        "consecutive_failures": 0,
        "last_target_ok": "local",
        "in_recovery": False,
    }


@router.get("/api/monitor")
async def monitor_snapshot() -> dict:
    """Current performance metrics snapshot."""
    return get_monitor().get_snapshot()


@router.post("/api/text")
async def text_command(request: Request) -> dict:
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    t0 = time.perf_counter()
    try:
        result = request.app.state.command_service.handle_text(text)
        elapsed = (time.perf_counter() - t0) * 1000
        success = bool(result.get("success", False))
        get_monitor().record_request(elapsed, success)
        # 记录 NLU 置信度
        if result.get("confidence") is not None:
            get_monitor().record_accuracy(float(result["confidence"]))
        return result
    except Exception:
        get_monitor().record_error()
        raise


@router.get("/api/esp32-status")
async def esp32_status(request: Request) -> dict:
    """查询 ESP32 连接状态。"""
    bridge = request.app.state.command_service.esp32_bridge
    if bridge is None:
        return {"connected": False, "count": 0, "sessions": []}
    sessions = bridge.sessions
    return {
        "connected": len(sessions) > 0,
        "count": len(sessions),
        "sessions": [{"host": s.client_host} for s in sessions],
    }


@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    await run_audio_session(
        websocket,
        command_service=websocket.app.state.command_service,
        asr=websocket.app.state.asr,
    )
