"""Voice + UI feedback helpers for the web console."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from voice_router_lite.tts.engine import TTSEngine
    from voice_router_lite.web.console_hub import WebConsoleHub

logger = logging.getLogger(__name__)

_UI_EVENTS = frozenset({
    "wake_detected",
    "transcript",
    "reply",
    "error",
    "collecting",
})


async def emit_ui_event(
    hub: Optional["WebConsoleHub"],
    payload: dict[str, Any],
    *,
    source: str = "esp32",
) -> None:
    """Push a user-visible event to all browser consoles."""
    if hub is None or payload.get("event") not in _UI_EVENTS:
        return
    ui = {**payload, "source": source}
    await hub.broadcast_json(ui)


async def push_voice_feedback(
    hub: Optional["WebConsoleHub"],
    tts: Optional["TTSEngine"],
    reply_text: str,
    *,
    success: bool,
) -> None:
    """Send short WAV confirmation audio to browser clients (optional)."""
    if hub is None or hub.browser_count == 0:
        return
    if tts is None:
        return
    try:
        from voice_router_lite.tts.engine import pcm_to_wav_bytes

        audio = tts.render_sentence(reply_text)
        if audio is None:
            audio = tts.render_tone("ok_beep" if success else "error_beep")
        if audio is None:
            return
        wav = pcm_to_wav_bytes(audio, sample_rate=tts.sample_rate)
        await hub.broadcast_json({
            "event": "tts_audio",
            "format": "wav",
            "sample_rate": tts.sample_rate,
        })
        await hub.broadcast_bytes(wav)
        await hub.broadcast_json({"event": "done"})
    except Exception as exc:
        logger.debug("TTS 推送到浏览器失败: %s", exc)
