"""FastAPI entrypoint for the Voice Router Lite web console."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from voice_router_lite.asr import ASREngine
from voice_router_lite.config import DEFAULT_CONFIG
from voice_router_lite.tts import TTSEngine
from voice_router_lite.web.console_hub import WebConsoleHub
from voice_router_lite.web.monitor import get_monitor, shutdown_monitor
from voice_router_lite.web.routes import router
from voice_router_lite.web.service import WebCommandService

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = WebCommandService(DEFAULT_CONFIG)
    service.initialize()
    if service.config.enable_cgroups:
        try:
            from voice_router_lite.platform.cgroups import apply_resource_limits
            apply_resource_limits(
                cpu_quota_pct=service.config.cgroup_cpu_quota_pct,
                memory_mb=service.config.cgroup_memory_mb,
            )
        except Exception:
            pass
    app.state.command_service = service
    app.state.asr = _build_optional_asr()
    app.state.console_hub = WebConsoleHub()
    app.state.tts = _build_web_tts()
    get_monitor().start()
    try:
        yield
    finally:
        shutdown_monitor()
        tts = getattr(app.state, "tts", None)
        if tts is not None:
            tts.close()
        service.close()


def create_app() -> FastAPI:
    app = FastAPI(title="voice-router-lite", lifespan=lifespan)
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _build_optional_asr() -> ASREngine | None:
    if os.getenv("VOICE_ROUTER_WEB_ASR", "").lower() not in {"1", "true", "yes"}:
        return None
    asr = ASREngine(DEFAULT_CONFIG.models, DEFAULT_CONFIG.audio)
    return asr if asr.initialize() else None


def _build_web_tts() -> TTSEngine | None:
    if os.getenv("VOICE_ROUTER_WEB_TTS", "1").lower() in {"0", "false", "no"}:
        return None
    tts = TTSEngine(DEFAULT_CONFIG.models, DEFAULT_CONFIG.audio)
    return tts if tts.initialize(open_player=False) else None


app = create_app()


def main() -> None:
    host = os.getenv("VOICE_ROUTER_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("VOICE_ROUTER_WEB_PORT", "28080"))
    uvicorn.run(
        "voice_router_lite.web.app:app",
        host=host,
        port=port,
        log_level=os.getenv("VOICE_ROUTER_WEB_LOG_LEVEL", "info"),
        reload=False,
    )


if __name__ == "__main__":
    main()
