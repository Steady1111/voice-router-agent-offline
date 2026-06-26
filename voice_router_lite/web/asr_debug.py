"""Persist ESP32 / voice command WAV + metadata for NLU tuning (task 1A)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from voice_router_lite.audio.capture import save_wav

_SAMPLE_RATE = 16000


def asr_debug_enabled() -> bool:
    return os.getenv("VOICE_ROUTER_SAVE_ASR_WAV", "").lower() in {"1", "true", "yes"}


def asr_debug_dir() -> Path:
    return Path(os.getenv("VOICE_ROUTER_ASR_DEBUG_DIR", "./asr_debug"))


def save_asr_debug_sample(
    audio_bytes: bytes | bytearray,
    *,
    raw_asr: str = "",
    repaired: str = "",
    command: str = "",
    intent: str = "",
    success: bool = False,
    wake_source: str = "",
    reason: str = "",
    peak: int = 0,
) -> Path | None:
    """Write WAV + sidecar JSON; returns JSON path or None if disabled."""
    if not asr_debug_enabled():
        return None
    if len(audio_bytes) < 3200:
        return None

    out_dir = asr_debug_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%d_%H%M%S_%f")
    base = out_dir / f"cmd_{stamp}"
    wav_path = base.with_suffix(".wav")
    json_path = base.with_suffix(".json")

    audio_np = np.frombuffer(bytes(audio_bytes), dtype=np.int16).copy()
    save_wav(str(wav_path), audio_np, _SAMPLE_RATE)

    duration_sec = len(audio_bytes) / 2 / _SAMPLE_RATE
    meta: dict[str, Any] = {
        "timestamp": ts.isoformat(),
        "wav": wav_path.name,
        "raw_asr": raw_asr,
        "repaired": repaired,
        "command": command,
        "intent": intent,
        "success": success,
        "wake_source": wake_source,
        "reason": reason,
        "peak": peak,
        "duration_sec": round(duration_sec, 3),
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path
