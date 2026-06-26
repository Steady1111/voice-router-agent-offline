"""Broadcast ESP32 / pipeline events to browser WebSocket observers."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebConsoleHub:
    """Fan-out JSON/binary events to all connected browser consoles."""

    def __init__(self) -> None:
        self._browsers: set[WebSocket] = set()

    def register(self, websocket: WebSocket) -> None:
        self._browsers.add(websocket)
        logger.info("[ConsoleHub] 浏览器已连接 (当前 %d)", len(self._browsers))

    def unregister(self, websocket: WebSocket) -> None:
        self._browsers.discard(websocket)
        logger.info("[ConsoleHub] 浏览器已断开 (剩余 %d)", len(self._browsers))

    @property
    def browser_count(self) -> int:
        return len(self._browsers)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        if not self._browsers:
            return
        text = json.dumps(payload, ensure_ascii=False)
        dead: list[WebSocket] = []
        for ws in list(self._browsers):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    async def broadcast_bytes(self, data: bytes) -> None:
        if not self._browsers:
            return
        dead: list[WebSocket] = []
        for ws in list(self._browsers):
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)
