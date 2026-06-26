"""Browser + ESP32 PCM WebSocket endpoint for the web console.

Browser: 按钮录音 → 语音识别 → NLU → 回复
ESP32:   持续音频流 → KWS唤醒检测 → ASR识别 → NLU → 设备控制（风扇/显示/OLED）
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
from voice_router_lite.kws.engine import KWSEngine, KWSResult
from voice_router_lite.web.feedback import emit_ui_event, push_voice_feedback
from voice_router_lite.web.monitor import get_monitor
from voice_router_lite.web.service import WebCommandService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        await _run_esp32_kws_session(
            websocket, command_service, client_host, hub=hub, tts=tts,
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
# ESP32 唤醒词监听链路
# ==================================================================

async def _run_esp32_kws_session(
    websocket: WebSocket,
    command_service: WebCommandService,
    client_host: str,
    *,
    hub=None,
    tts=None,
) -> None:
    """ESP32 专用：持续 KWS 唤醒词检测 → 唤醒后 ASR → NLU → 设备控制。"""
    await websocket.accept()

    # ---- 1. 注册到桥接器 ----
    if command_service.esp32_bridge:
        command_service.esp32_bridge.register(client_host, lambda p: _send_json(websocket, p))
        logger.info("[ESP32 KWS] 设备已注册: %s", client_host)
        mgr = command_service.devices
        asyncio.create_task(
            command_service.esp32_bridge.sync_device_state(
                mgr._fan_is_on,
                mgr._fan_speed_level,
            )
        )

    # ---- 2. 初始化 KWS 引擎 ----
    kws_engine = KWSEngine(
        DEFAULT_CONFIG.models,
        DEFAULT_CONFIG.audio,
        wake_word_threshold=DEFAULT_CONFIG.wake_word_threshold,
    )
    kws_ok = kws_engine.initialize()
    logger.info("[ESP32 KWS] KWS 引擎初始化: ok=%s mode=%s", kws_ok, kws_engine.mode)

    # ---- 3. 初始化 ASR ----
    asr = ASREngine(
        DEFAULT_CONFIG.models,
        DEFAULT_CONFIG.audio,
        num_threads=DEFAULT_CONFIG.asr_num_threads,
    )
    asr_ok = asr.initialize()
    logger.info("[ESP32 KWS] ASR 引擎: ok=%s", asr_ok)

    # ---- 4. 状态变量 ----
    class State:
        LISTENING = "LISTENING"    # KWS 静默监听
        COLLECTING = "COLLECTING"  # 唤醒后收集语音指令
    state = State.LISTENING
    asr_buffer = bytearray()
    last_speech_time = 0.0
    silence_start = 0.0
    wake_cooldown_until = 0.0
    skip_frames_until = 0.0       # 唤醒后跳过缓冲区的旧帧

    # 常量
    SILENCE_TIMEOUT = 2.0
    MAX_UTTERANCE = 8.0
    MAX_COLLECT_SEC = 10.0
    WAKE_SKIP_SEC = 0.25
    ENERGY_WAKE_SKIP_SEC = 0.25
    COOLDOWN_SEC = 4.0
    COOLDOWN_FAIL_SEC = 0.8
    KWS_SKIP_SEC = 0.3
    SPEECH_THRESHOLD = 0.05
    MIN_ASR_BYTES = 16000
    MIN_COMMAND_BYTES = 48000       # 至少 ~1.5s 指令音频再结束
    USE_ESP32_RECORD_MODE = os.getenv("VOICE_ROUTER_ESP32_RECORD_MODE", "0").lower() in {
        "1", "true", "yes",
    }

    # KWS 状态
    kws_last_detect_time = 0.0
    energy_wake_streak = 0
    energy_wake_rms = float(os.getenv("VOICE_ROUTER_ENERGY_WAKE_RMS", "0.18"))
    energy_wake_frames = int(os.getenv("VOICE_ROUTER_ENERGY_WAKE_FRAMES", "4"))
    use_energy_wake = os.getenv("VOICE_ROUTER_ENERGY_WAKE", "1").lower() in {
        "1", "true", "yes",
    }

    # 调试计数器
    frame_count = 0
    last_debug_log = time.time()

    # ---- 5. 通知 ESP32 进入 KWS 静默监听模式 ----
    await _send_json(websocket, {"event": "start_kws"})
    logger.info("[ESP32 KWS] 已发送 start_kws，进入静默监听...")

    collect_start_time = 0.0
    collect_skip_sec = 0.0
    esp32_record_mode = False

    async def _notify_wake(keyword: str, confidence: float) -> None:
        payload = {
            "event": "wake_detected",
            "keyword": keyword,
            "confidence": confidence,
        }
        await _send_json(websocket, payload)
        await emit_ui_event(hub, payload, source="esp32")
        await emit_ui_event(
            hub,
            {"event": "collecting", "text": "正在听指令…"},
            source="esp32",
        )

    async def _notify_transcript(
        text: str,
        *,
        raw_asr: str | None = None,
        corrected: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"event": "transcript", "text": text}
        if raw_asr and raw_asr != text:
            payload["raw_asr"] = raw_asr
            payload["corrected"] = True
        elif corrected:
            payload["corrected"] = True
        await _send_json(websocket, payload)
        await emit_ui_event(hub, payload, source="esp32")

    async def _notify_reply(reply_text: str, result: dict, *, success: bool) -> None:
        payload = {
            "event": "reply",
            "text": reply_text,
            "result": result,
        }
        await _send_json(websocket, payload)
        await emit_ui_event(hub, payload, source="esp32")
        await push_voice_feedback(hub, tts, reply_text, success=success)

    async def finish_collecting(reason: str) -> None:
        nonlocal state, collect_start_time, collect_skip_sec, esp32_record_mode, wake_cooldown_until
        total_dur = len(asr_buffer) / 2 / 16000
        logger.info(
            "[ESP32 KWS] 指令结束(%s): dur=%.1fs bytes=%d",
            reason, total_dur, len(asr_buffer),
        )

        if esp32_record_mode:
            await _send_json(websocket, {"event": "stop_record"})
            esp32_record_mode = False

        transcript = ""
        success = False
        if asr_ok and len(asr_buffer) >= MIN_ASR_BYTES:
            try:
                audio_np = np.frombuffer(bytes(asr_buffer), dtype=np.int16).copy()
                loop = asyncio.get_event_loop()
                asr_t0 = time.perf_counter()
                transcript = await loop.run_in_executor(None, _do_asr, asr, audio_np)
                asr_elapsed = (time.perf_counter() - asr_t0) * 1000
                get_monitor().record_inference_time("asr", asr_elapsed)
            except Exception as e:
                logger.exception("[ESP32 KWS] ASR 异常: %s", e)

        if transcript:
            raw_asr = _normalize_command_text(transcript)
            logger.info("[ESP32 KWS] ASR 识别: '%s'", raw_asr)

            t0 = time.perf_counter()
            result = command_service.handle_text(raw_asr, voice_wake=True)
            elapsed = (time.perf_counter() - t0) * 1000
            success = bool(result.get("success", False))
            get_monitor().record_request(elapsed, success)
            get_monitor().record_inference_time("nlu", elapsed)
            if result.get("confidence") is not None:
                get_monitor().record_accuracy(float(result["confidence"]))

            display_cmd = result.get("command") or result.get("transcript") or raw_asr
            await _notify_transcript(
                display_cmd,
                raw_asr=raw_asr if display_cmd != raw_asr else None,
                corrected=bool(result.get("corrected")),
            )

            reply_text = result.get("reply", "")
            if not success and len(raw_asr) <= 5 and "风扇" not in reply_text:
                reply_text = f"没听清（识别为「{raw_asr}」），请再说一次，如「打开风扇」"

            await _notify_reply(reply_text, result, success=success)
            logger.info(
                "[ESP32 KWS] NLU: intent=%s command=%s reply=%s",
                result.get("intent"), display_cmd, reply_text,
            )
        else:
            guess_fan = (
                os.getenv("VOICE_ROUTER_VOICE_GUESS_FAN", "1").lower() in {"1", "true", "yes"}
                and len(asr_buffer) >= MIN_COMMAND_BYTES
            )
            if guess_fan:
                peak = int(np.max(np.abs(
                    np.frombuffer(bytes(asr_buffer), dtype=np.int16)
                ))) if len(asr_buffer) >= 2 else 0
                if peak > 2500:
                    fan_on = bool(command_service.devices._fan_is_on)
                    transcript = "关闭风扇" if fan_on else "打开风扇"
                    logger.info(
                        "[ESP32 KWS] ASR 空但有语音能量 peak=%d fan_on=%s → 猜测「%s」",
                        peak, fan_on, transcript,
                    )
                    result = command_service.handle_text(transcript, voice_wake=True)
                    display_cmd = result.get("command") or transcript
                    await _notify_transcript(display_cmd, corrected=True)
                    reply_text = result.get("reply", "")
                    await _notify_reply(
                        reply_text, result,
                        success=bool(result.get("success", False)),
                    )
                    logger.info(
                        "[ESP32 KWS] NLU(guess): intent=%s reply=%s",
                        result.get("intent"), reply_text,
                    )
                else:
                    fail_reply = "没听清，请再说一次"
                    await _notify_reply(
                        fail_reply, {"success": False},
                        success=False,
                    )
                    logger.warning(
                        "[ESP32 KWS] ASR 无结果 (reason=%s bytes=%d peak=%d)",
                        reason, len(asr_buffer), peak,
                    )
            else:
                fail_reply = "没听清，请再说一次"
                await _notify_reply(fail_reply, {"success": False}, success=False)
                logger.warning(
                    "[ESP32 KWS] ASR 无结果 (reason=%s bytes=%d need>=%d)",
                    reason, len(asr_buffer), MIN_ASR_BYTES,
                )

        await asyncio.sleep(0.5)
        await _send_json(websocket, {"event": "start_kws"})
        kws_engine.reset()
        state = State.LISTENING
        asr_buffer.clear()
        collect_start_time = 0.0
        fail_cooldown = (
            not transcript
            or (transcript and not success)
        ) if transcript else True
        cd = COOLDOWN_FAIL_SEC if fail_cooldown else COOLDOWN_SEC
        wake_cooldown_until = time.time() + cd

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # 文本消息
            text = message.get("text")
            if text is not None:
                payload = json.loads(text)
                event = payload.get("event", "")
                if event == "stop":
                    # ESP32 固件 15s 超时：仍需处理已采集音频
                    if state == State.COLLECTING and len(asr_buffer) > 0:
                        await finish_collecting("esp32_timeout")
                    else:
                        state = State.LISTENING
                        asr_buffer.clear()
                        kws_engine.reset()
                        collect_start_time = 0.0
                    logger.info("[ESP32 KWS] ESP32 录音超时，已自动回到 KWS 监听")
                elif event == "ping":
                    await _send_json(websocket, {"event": "pong"})
                elif event == "button_wake":
                    btn_now = time.time()
                    if state == State.LISTENING and btn_now > wake_cooldown_until:
                        state = State.COLLECTING
                        collect_start_time = btn_now
                        wake_cooldown_until = btn_now + COOLDOWN_SEC
                        kws_engine.reset()
                        skip_frames_until = btn_now + KWS_SKIP_SEC
                        collect_skip_sec = KWS_SKIP_SEC
                        asr_buffer.clear()
                        last_speech_time = 0.0
                        silence_start = 0.0
                        await _notify_wake("按键", 1.0)
                        logger.info("[ESP32 KWS] 🔘 按键唤醒 → 开始采集指令")
                elif event == "device_state":
                    _handle_esp32_state(command_service, payload)
                continue

            # 二进制音频帧
            data = message.get("bytes")
            if data is None or len(data) < 100:
                continue

            # 转 float32 (无需 copy，astype 会创建新数组)
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            now = time.time()

            # 调试: 每5秒输出一次音频流状态
            frame_count += 1
            energy = float(np.sqrt(np.mean(audio_float32 ** 2)))
            if now - last_debug_log > 5.0:
                logger.info("[ESP32 KWS] 📡 音频流活跃: frames=%d energy=%.5f state=%s len=%d",
                            frame_count, energy, state, len(data))
                last_debug_log = now
                frame_count = 0

            # 实时推送音频电平到监控面板
            get_monitor().update_audio_energy(energy, state)

            if state == State.LISTENING:
                if now < wake_cooldown_until:
                    energy_wake_streak = 0
                # === KWS 唤醒词检测 (线程池运行，避免阻塞事件循环) ===
                loop = asyncio.get_event_loop()
                kws_t0 = time.perf_counter()
                kws_result = await loop.run_in_executor(
                    None, kws_engine.detect, audio_float32,
                ) if kws_ok else KWSResult(detected=False)
                kws_elapsed = (time.perf_counter() - kws_t0) * 1000
                get_monitor().record_inference_time("kws", kws_elapsed)

                detected = kws_result.detected
                from_energy_wake = False
                if detected:
                    logger.info(
                        "[ESP32 KWS] 🎯 唤醒词检测到! keyword='%s' confidence=%.4f",
                        kws_result.keyword, kws_result.confidence,
                    )
                elif use_energy_wake:
                    if energy > energy_wake_rms:
                        energy_wake_streak += 1
                    else:
                        energy_wake_streak = max(0, energy_wake_streak - 1)
                    if energy_wake_streak >= energy_wake_frames:
                        detected = True
                        from_energy_wake = True
                        kws_result = KWSResult(
                            detected=True,
                            keyword="小T小T",
                            confidence=min(energy, 1.0),
                        )
                        energy_wake_streak = 0
                        logger.info(
                            "[ESP32 KWS] ⚡ 能量唤醒(预研) energy=%.4f",
                            energy,
                        )

                bypass_cooldown = detected and not from_energy_wake
                if detected and (now > wake_cooldown_until or bypass_cooldown):
                    state = State.COLLECTING
                    collect_start_time = now
                    wake_cooldown_until = now + COOLDOWN_SEC
                    kws_engine.reset()
                    skip_sec = ENERGY_WAKE_SKIP_SEC if from_energy_wake else KWS_SKIP_SEC
                    skip_frames_until = now + skip_sec
                    collect_skip_sec = skip_sec
                    asr_buffer.clear()
                    last_speech_time = 0.0
                    silence_start = 0.0
                    if USE_ESP32_RECORD_MODE:
                        esp32_record_mode = True
                        await _send_json(websocket, {"event": "start_record"})
                    await _notify_wake(
                        kws_result.keyword or "小T小T",
                        kws_result.confidence,
                    )
                    logger.info(
                        "[ESP32 KWS] 🎤 唤醒 → 开始采集指令 (skip=%.1fs energy=%s)",
                        skip_sec, from_energy_wake,
                    )
                elif detected:
                    logger.info(
                        "[ESP32 KWS] 唤醒被冷却拦截 (剩余 %.1fs，请再说一次唤醒词)",
                        wake_cooldown_until - now,
                    )

            elif state == State.COLLECTING:
                # 跳过唤醒后缓冲区残留帧
                if now < skip_frames_until:
                    continue

                asr_buffer.extend(data)
                energy = float(np.sqrt(np.mean(audio_float32 ** 2)))

                if energy > SPEECH_THRESHOLD:
                    last_speech_time = now
                    silence_start = 0.0
                elif silence_start == 0.0:
                    silence_start = now

                total_dur = len(asr_buffer) / 2 / 16000
                silence_dur = now - silence_start if silence_start > 0 else 0
                collect_elapsed = now - collect_start_time if collect_start_time > 0 else 0

                min_collect_elapsed = collect_skip_sec + 1.8
                should_finish = (
                    (
                        silence_dur > SILENCE_TIMEOUT
                        and last_speech_time > 0
                        and len(asr_buffer) >= MIN_COMMAND_BYTES
                        and collect_elapsed >= min_collect_elapsed
                    )
                    or total_dur > MAX_UTTERANCE
                    or (
                        collect_elapsed > MAX_COLLECT_SEC
                        and len(asr_buffer) >= MIN_COMMAND_BYTES
                    )
                )
                if should_finish:
                    if silence_dur > SILENCE_TIMEOUT and last_speech_time > 0:
                        finish_reason = "silence"
                    elif total_dur > MAX_UTTERANCE:
                        finish_reason = "max_utterance"
                    else:
                        finish_reason = "max_collect"
                    await finish_collecting(finish_reason)
                    continue

    except WebSocketDisconnect:
        logger.info("[ESP32 KWS] 客户端断开: %s", client_host)
    except Exception as exc:
        logger.exception("[ESP32 KWS] 异常: %s", exc)
    finally:
        if command_service.esp32_bridge:
            command_service.esp32_bridge.unregister(client_host)
            logger.info("[ESP32 KWS] ESP32 已注销: %s", client_host)


def _normalize_command_text(text: str) -> str:
    """去掉唤醒词残留，便于 NLU 识别指令。"""
    cleaned = text.strip()
    for wake in ("小T小T", "小t小t", "小提小提", "小爱同学", "真是", "然后"):
        cleaned = cleaned.replace(wake, "")
    return cleaned.strip(" ，。！？\t")


def _prepare_asr_audio(audio_np: np.ndarray) -> np.ndarray:
    """归一化音量，修复削波导致的 ASR 乱识别。"""
    peak = int(np.max(np.abs(audio_np))) if len(audio_np) else 0
    if peak <= 0:
        return audio_np
    clip_count = int(np.sum(np.abs(audio_np) >= 32000))
    clip_ratio = clip_count / len(audio_np)
    if peak >= 30000 or clip_ratio > 0.01:
        target = 6000
        logger.warning(
            "ASR 音频削波: peak=%d clip_ratio=%.2f%% → 衰减到 %d",
            peak, clip_ratio * 100, target,
        )
    elif peak > 20000:
        target = 10000
    elif peak < 15000:
        target = 14000
    else:
        target = 0
    if target:
        audio_np = (audio_np.astype(np.float32) * (target / peak)).astype(np.int16)
    tail = np.zeros(int(0.5 * 16000), dtype=np.int16)
    return np.concatenate([audio_np, tail])


def _do_asr(asr: ASREngine, audio_np: np.ndarray) -> str:
    """在线程池中执行 ASR 识别 (同步函数)。"""
    try:
        peak = int(np.max(np.abs(audio_np))) if len(audio_np) else 0
        logger.info("ASR 输入: samples=%d peak=%d", len(audio_np), peak)
        audio_np = _prepare_asr_audio(audio_np)
        asr_result = asr.transcribe(audio_np)
        text = asr_result.text.strip()
        if not text:
            logger.warning("ASR 返回空文本 (samples=%d peak=%d)", len(audio_np), peak)
        return text
    except Exception as e:
        logger.exception("ASR 执行失败: %s", e)
        return ""


# ==================================================================
# 共享工具函数
# ==================================================================

def _handle_esp32_state(command_service: WebCommandService, payload: dict) -> None:
    """处理 ESP32 上报的设备状态。"""
    device_type = payload.get("device_type", "")
    state = payload.get("state", {})
    if device_type == "fan":
        on = bool(state.get("on", False))
        level = int(state.get("level", 0))
        logger.info("[ESP32 KWS] 风扇 ACK: on=%s level=%d", on, level)
    if device_type and command_service.esp32_bridge and command_service.esp32_bridge._on_device_state:
        try:
            command_service.esp32_bridge._on_device_state(device_type, state)
        except Exception:
            pass


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
        await _send_json(websocket, {"event": "done"})
        return

    if asr is None:
        await _send_json(websocket, {
            "event": "error",
            "message": "音频识别未启用；请先使用文本调试，或设置 VOICE_ROUTER_WEB_ASR=1。",
        })
        await _send_json(websocket, {"event": "done"})
        return

    audio = np.frombuffer(audio_bytes, dtype=np.int16).copy()
    if sample_rate != asr._audio_config.sample_rate:
        await _send_json(websocket, {
            "event": "error",
            "message": f"暂不支持 {sample_rate}Hz 音频，请使用 16000Hz。",
        })
        await _send_json(websocket, {"event": "done"})
        return

    try:
        asr_t0 = time.perf_counter()
        asr_result = asr.transcribe(audio)
        asr_elapsed = (time.perf_counter() - asr_t0) * 1000
        get_monitor().record_inference_time("asr", asr_elapsed)

        transcript = asr_result.text.strip()
        raw_asr = transcript
        nlu_t0 = time.perf_counter()
        result = command_service.handle_text(transcript)
        nlu_elapsed = (time.perf_counter() - nlu_t0) * 1000
        get_monitor().record_inference_time("nlu", nlu_elapsed)

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
    except Exception:
        get_monitor().record_error()
        raise
    finally:
        await _send_json(websocket, {"event": "done"})


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))
