"""Browser + ESP32 PCM WebSocket endpoint for the web console.

Browser: 按钮录音 → ASR → NLU → 回复
ESP32:   持续流 → 音量唤醒 → 单次采集 → ASR → NLU → 设备控制（方案 1）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from voice_router_lite.asr import ASREngine
from voice_router_lite.config import DEFAULT_CONFIG
from voice_router_lite.web.feedback import emit_ui_event, push_voice_feedback
from voice_router_lite.web.monitor import get_monitor
from voice_router_lite.web.service import WebCommandService
from voice_router_lite.web.wake_policy import (
    VolumeWakeConfig,
    VolumeWakeDetector,
    strip_wake_prefix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _emit_wake_detected(
    websocket: WebSocket,
    hub,
    keyword: str,
    confidence: float,
    *,
    wake_source: str = "volume",
) -> None:
    payload = {
        "event": "wake_detected",
        "keyword": keyword,
        "confidence": confidence,
        "wake_source": wake_source,
    }
    await _send_json(websocket, payload)
    await emit_ui_event(hub, payload, source="esp32")
    await emit_ui_event(
        hub,
        {"event": "collecting", "text": "正在听…"},
        source="esp32",
    )


async def run_audio_session(
    websocket: WebSocket,
    command_service: WebCommandService,
    asr: ASREngine | None,
) -> None:
    role = websocket.query_params.get("role", "").lower()
    client_host = websocket.client.host if websocket.client else "unknown"
    is_esp32 = role == "esp32" or (
        role != "browser"
        and client_host not in ("127.0.0.1", "::1", "localhost")
        and os.getenv("VOICE_ROUTER_FORCE_BROWSER_WS", "").lower()
        not in {"1", "true", "yes"}
    )

    if is_esp32:
        hub = getattr(websocket.app.state, "console_hub", None)
        tts = getattr(websocket.app.state, "tts", None)
        await _run_esp32_session(
            websocket, command_service, client_host, asr=asr, hub=hub, tts=tts,
        )
        return

    await websocket.accept()
    hub = getattr(websocket.app.state, "console_hub", None)
    if hub is not None:
        hub.register(websocket)
    sample_rate = 16000
    buffer = bytearray()
    started = False

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if started:
                    buffer.extend(data)
                continue

            text = message.get("text")
            if text is None:
                continue

            payload = json.loads(text)
            event = payload.get("event")
            if event == "start":
                sample_rate = int(payload.get("sample_rate", 16000))
                buffer.clear()
                started = True
            elif event == "stop":
                started = False
                tts = getattr(websocket.app.state, "tts", None)
                await _process_audio_turn(
                    websocket, command_service, asr, bytes(buffer), sample_rate,
                    hub=hub, tts=tts,
                )
                buffer.clear()
            elif event == "ping":
                await _send_json(websocket, {"event": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("[WS Browser] 异常: %s", exc)
        await _send_json(websocket, {"event": "error", "message": str(exc)})
    finally:
        if hub is not None:
            hub.unregister(websocket)


# ==================================================================
# ESP32 演示链路（方案 1：IDLE → RECORDING → ASR → NLU）
# ==================================================================

async def _run_esp32_session(
    websocket: WebSocket,
    command_service: WebCommandService,
    client_host: str,
    *,
    asr: ASREngine | None = None,
    hub=None,
    tts=None,
) -> None:
    """ESP32：音量/按键唤醒 → 单次采集 → ASR → NLU → 设备控制。"""
    await websocket.accept()

    if command_service.esp32_bridge:
        command_service.esp32_bridge.register(client_host, lambda p: _send_json(websocket, p))
        logger.info("[ESP32] 设备已注册: %s", client_host)
        mgr = command_service.devices
        asyncio.create_task(
            command_service.esp32_bridge.sync_device_state(
                mgr._fan_is_on,
                mgr._fan_speed_level,
            )
        )

    esp32_asr = asr
    if esp32_asr is None:
        esp32_asr = ASREngine(DEFAULT_CONFIG.models, DEFAULT_CONFIG.audio)
        if esp32_asr.initialize():
            logger.info("[ESP32] ASR 已加载")
        else:
            esp32_asr = None
            logger.warning("[ESP32] ASR 初始化失败")

    class State:
        IDLE = "IDLE"
        RECORDING = "RECORDING"

    wake_cfg = VolumeWakeConfig.from_env()
    volume_wake = VolumeWakeDetector(wake_cfg)

    state = State.IDLE
    sample_rate = DEFAULT_CONFIG.audio.sample_rate
    utterance_buffer: list[np.ndarray] = []
    last_speech_time = 0.0
    silence_start = 0.0
    recording_started_at = 0.0

    speech_threshold = float(os.getenv("VOICE_ROUTER_SPEECH_THRESHOLD", "0.012"))
    silence_timeout = float(os.getenv("VOICE_ROUTER_SILENCE_SEC", "0.8"))
    max_collect_sec = float(os.getenv("VOICE_ROUTER_COLLECT_SEC", "5.0"))
    pre_roll_sec = float(os.getenv("VOICE_ROUTER_PRE_ROLL_SEC", "0.45"))
    pre_roll_samples = int(pre_roll_sec * sample_rate)
    pre_roll_buffer = np.zeros(0, dtype=np.float32)

    frame_count = 0
    last_debug_log = time.time()

    await _send_json(websocket, {"event": "start_kws"})
    logger.info(
        "[ESP32] 方案1 已启动 peak=%.3f rms=%.3f cooldown=%.1fs max_collect=%.1fs",
        wake_cfg.peak_threshold,
        wake_cfg.rms_threshold,
        wake_cfg.cooldown_sec,
        max_collect_sec,
    )

    async def _begin_recording(now: float, *, wake_source: str, confidence: float) -> None:
        nonlocal state, utterance_buffer, last_speech_time, silence_start, recording_started_at
        state = State.RECORDING
        utterance_buffer = []
        if pre_roll_buffer.size > 0:
            utterance_buffer.append(pre_roll_buffer.copy())
        last_speech_time = 0.0
        silence_start = 0.0
        recording_started_at = now
        await _emit_wake_detected(
            websocket, hub, "小T小T", confidence, wake_source=wake_source,
        )
        logger.info("[ESP32] 唤醒 source=%s conf=%.4f", wake_source, confidence)

    async def _finish_recording(now: float) -> None:
        nonlocal state, utterance_buffer, last_speech_time, silence_start
        chunks = utterance_buffer
        utterance_buffer = []
        last_speech_time = 0.0
        silence_start = 0.0
        state = State.IDLE
        volume_wake.set_cooldown(now)
        if not chunks:
            logger.info("[ESP32] 采集结束（无音频）")
            return
        await _process_esp32_utterance(
            websocket,
            command_service,
            esp32_asr,
            chunks,
            hub=hub,
            tts=tts,
        )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                payload = json.loads(text)
                event = payload.get("event", "")
                if event == "ping":
                    await _send_json(websocket, {"event": "pong"})
                elif event == "button_wake":
                    btn_now = time.time()
                    if state == State.IDLE and btn_now > volume_wake.cooldown_until:
                        await _begin_recording(btn_now, wake_source="button", confidence=1.0)
                        logger.info("[ESP32] 按键唤醒")
                elif event == "device_state":
                    _handle_esp32_state(command_service, payload)
                elif event == "telemetry":
                    _handle_esp32_telemetry(command_service, payload)
                continue

            data = message.get("bytes")
            if data is None or len(data) < 100:
                continue

            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            now = time.time()
            energy = float(np.sqrt(np.mean(audio_float32 ** 2)))

            frame_count += 1
            if now - last_debug_log > 5.0:
                logger.info(
                    "[ESP32] frames=%d energy=%.5f state=%s floor=%.5f",
                    frame_count, energy, state, volume_wake.noise_floor,
                )
                last_debug_log = now
                frame_count = 0

            get_monitor().update_audio_energy(energy, state)

            if state == State.IDLE:
                pre_roll_buffer = np.concatenate([pre_roll_buffer, audio_float32])
                if pre_roll_buffer.size > pre_roll_samples:
                    pre_roll_buffer = pre_roll_buffer[-pre_roll_samples:].copy()

                if volume_wake.check(energy, now):
                    await _begin_recording(now, wake_source="volume", confidence=min(energy, 1.0))
                    volume_wake.reset_streak()

            if state == State.RECORDING:
                utterance_buffer.append(audio_float32.copy())
                if energy > speech_threshold:
                    last_speech_time = now
                    silence_start = 0.0
                elif silence_start == 0.0 and last_speech_time > 0:
                    silence_start = now

                total_samples = sum(len(c) for c in utterance_buffer)
                total_dur = total_samples / sample_rate
                silence_dur = (now - silence_start) if silence_start > 0 else 0.0
                timed_out = (now - recording_started_at) > max_collect_sec
                end_collect = (
                    (silence_dur > silence_timeout and last_speech_time > 0)
                    or total_dur > max_collect_sec
                    or timed_out
                )
                if end_collect:
                    await _finish_recording(now)

    except WebSocketDisconnect:
        logger.info("[ESP32] 客户端断开: %s", client_host)
    except Exception as exc:
        logger.exception("[ESP32] 异常: %s", exc)
    finally:
        if command_service.esp32_bridge:
            command_service.esp32_bridge.unregister(client_host)
            logger.info("[ESP32] 已注销: %s", client_host)


async def _process_esp32_utterance(
    websocket: WebSocket,
    command_service: WebCommandService,
    asr: ASREngine | None,
    chunks: list[np.ndarray],
    *,
    hub=None,
    tts=None,
) -> None:
    """单次采集：ASR → 去唤醒词 → NLU → 前端 + ESP32 反馈。"""
    if not chunks:
        return

    audio_f = np.concatenate(chunks)
    audio_int16 = np.clip(audio_f * 32767.0, -32768, 32767).astype(np.int16)

    if asr is None:
        msg = "音频识别不可用，请确认 models/asr 已下载"
        await emit_ui_event(hub, {"event": "error", "message": msg}, source="esp32")
        await _send_json(websocket, {"event": "error", "message": msg})
        return

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    monitor = get_monitor()
    monitor.enter_router_asr()
    try:
        asr_t0 = time.perf_counter()
        asr_result = await loop.run_in_executor(None, asr.transcribe, audio_int16)
        get_monitor().record_inference_time("asr", (time.perf_counter() - asr_t0) * 1000)

        raw_asr = (asr_result.text or "").strip()
        command_text = strip_wake_prefix(raw_asr)
        text_for_nlu = command_text or raw_asr
        logger.info("[ESP32] ASR raw=%r command=%r", raw_asr, text_for_nlu)

        if not text_for_nlu:
            result = {
                "success": False,
                "asr_text": raw_asr,
                "transcript": "",
                "command": "",
                "reply": "没听清，请再说一次",
                "intent": "unknown",
                "slots": {},
            }
        else:
            nlu_t0 = time.perf_counter()
            result = command_service.handle_text(
                text_for_nlu,
                voice_wake=True,
                esp32_mic=True,
                raw_asr=raw_asr,
            )
            get_monitor().record_inference_time(
                "nlu", (time.perf_counter() - nlu_t0) * 1000,
            )

        success = bool(result.get("success", False))
        get_monitor().record_request((time.perf_counter() - t0) * 1000, success)
        if result.get("confidence") is not None:
            get_monitor().record_accuracy(float(result["confidence"]))

        display_cmd = result.get("command") or result.get("transcript") or text_for_nlu
        transcript_payload: dict[str, Any] = {
            "event": "transcript",
            "text": display_cmd,
        }
        if display_cmd != raw_asr and raw_asr:
            transcript_payload["raw_asr"] = raw_asr
            transcript_payload["corrected"] = True
        await _send_json(websocket, transcript_payload)
        await emit_ui_event(hub, transcript_payload, source="esp32")

        reply_text = result.get("reply", "")
        reply_payload = {
            "event": "reply",
            "text": reply_text,
            "result": result,
        }
        await _send_json(websocket, reply_payload)
        await emit_ui_event(hub, reply_payload, source="esp32")
        await push_voice_feedback(hub, tts, reply_text, success=success)

        logger.info(
            "[ESP32] NLU intent=%s success=%s reply=%s",
            result.get("intent"), success, reply_text,
        )

        try:
            from voice_router_lite.web.asr_debug import save_asr_debug_sample

            save_asr_debug_sample(
                audio_int16.tobytes(),
                raw_asr=raw_asr,
                repaired=display_cmd if display_cmd != raw_asr else "",
                command=display_cmd or text_for_nlu,
                intent=str(result.get("intent") or ""),
                success=success,
                wake_source="volume",
                peak=int(np.max(np.abs(audio_int16))) if len(audio_int16) else 0,
            )
        except Exception:
            logger.exception("[ESP32] asr_debug 落盘失败")
    except Exception:
        get_monitor().record_error()
        raise
    finally:
        monitor.exit_router_asr()


# ==================================================================
# 共享工具函数
# ==================================================================

def _handle_esp32_state(command_service: WebCommandService, payload: dict) -> None:
    """处理 ESP32 上报的设备状态。"""
    import datetime

    device_type = payload.get("device_type", "")
    state = payload.get("state", {})
    if device_type == "fan":
        on = bool(state.get("on", False))
        level = int(state.get("level", 0))
        command_service.last_ack_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("[ESP32 KWS] 风扇 ACK: on=%s level=%d", on, level)
    if device_type and command_service.esp32_bridge and command_service.esp32_bridge._on_device_state:
        try:
            command_service.esp32_bridge._on_device_state(device_type, state)
        except Exception:
            pass


def _handle_esp32_telemetry(command_service: WebCommandService, payload: dict) -> None:
    """处理 ESP32 周期性遥测（温度等）。"""
    temp = payload.get("temperature_c")
    if temp is None:
        return
    try:
        temp_f = float(temp)
    except (TypeError, ValueError):
        return
    if not (-55.0 <= temp_f <= 125.0):
        return
    command_service.update_esp32_temperature(temp_f)
    logger.info("[ESP32 KWS] 温度上报: %.1f°C", temp_f)


async def _process_audio_turn(
    websocket: WebSocket,
    command_service: WebCommandService,
    asr: ASREngine | None,
    audio_bytes: bytes,
    sample_rate: int,
    *,
    hub=None,
    tts=None,
) -> None:
    t0 = time.perf_counter()

    if not audio_bytes:
        await _send_json(websocket, {
            "event": "error",
            "message": "没录到声音，请按住按钮至少 1 秒",
        })
        await _send_json(websocket, {"event": "done"})
        return

    if asr is None:
        await _send_json(websocket, {
            "event": "error",
            "message": "音频识别不可用；请确认 models/asr 已下载，或重启服务查看日志。",
        })
        await _send_json(websocket, {"event": "done"})
        return

    audio = np.frombuffer(audio_bytes, dtype=np.int16).copy()
    target_rate = asr._audio_config.sample_rate
    if sample_rate != target_rate:
        src_len = audio.shape[0]
        dst_len = max(1, int(round(src_len * target_rate / sample_rate)))
        idx = np.linspace(0, src_len - 1, dst_len)
        audio = np.interp(idx, np.arange(src_len), audio.astype(np.float64)).astype(np.int16)
        logger.info("[WS Browser] 重采样 %dHz → %dHz", sample_rate, target_rate)

    try:
        monitor = get_monitor()
        monitor.enter_router_asr()
        asr_t0 = time.perf_counter()
        asr_result = asr.transcribe(audio)
        asr_elapsed = (time.perf_counter() - asr_t0) * 1000
        get_monitor().record_inference_time("asr", asr_elapsed)

        transcript = asr_result.text.strip()
        raw_asr = transcript
        if transcript:
            logger.info("[WS Browser] ASR: %r", raw_asr)
        nlu_t0 = time.perf_counter()
        result = command_service.handle_text(transcript)
        nlu_elapsed = (time.perf_counter() - nlu_t0) * 1000
        get_monitor().record_inference_time("nlu", nlu_elapsed)
        if transcript:
            logger.info(
                "[WS Browser] NLU: intent=%s conf=%.2f success=%s text=%r",
                result.get("intent"),
                float(result.get("confidence") or 0),
                result.get("success"),
                result.get("transcript") or raw_asr,
            )

        display_cmd = result.get("command") or result.get("transcript") or raw_asr
        await _send_json(websocket, {
            "event": "transcript",
            "text": display_cmd,
            **({"raw_asr": raw_asr, "corrected": True} if display_cmd != raw_asr else {}),
        })

        reply_text = result.get("reply", "")
        success = bool(result.get("success", False))
        await _send_json(websocket, {
            "event": "reply",
            "text": reply_text,
            "result": result,
        })

        elapsed = (time.perf_counter() - t0) * 1000
        get_monitor().record_request(elapsed, success)
        if result.get("confidence") is not None:
            get_monitor().record_accuracy(float(result["confidence"]))

        try:
            from voice_router_lite.web.asr_debug import save_asr_debug_sample

            save_asr_debug_sample(
                audio.tobytes(),
                raw_asr=raw_asr,
                repaired=display_cmd if display_cmd != raw_asr else "",
                command=display_cmd or raw_asr,
                intent=str(result.get("intent") or ""),
                success=success,
                wake_source="browser",
                peak=int(np.max(np.abs(audio))) if len(audio) else 0,
            )
        except Exception:
            logger.exception("[WS Browser] asr_debug 落盘失败")
    except Exception:
        get_monitor().record_error()
        raise
    finally:
        get_monitor().exit_router_asr()
        await _send_json(websocket, {"event": "done"})


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
