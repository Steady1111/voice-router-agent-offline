"""
离线语音处理管道 - 核心编排器

将 KWS、ASR、NLU、TTS、设备控制组合为完整的语音交互管线。
所有处理均在本地完成，无网络依赖。

管线流程:
    麦克风采集 → VAD → 降噪 → KWS唤醒 → ASR识别 → NLU理解 → 设备控制 → TTS反馈

使用方法:
    pipeline = VoiceRouterPipeline()
    pipeline.initialize()

    # 模式A: 手动处理
    pipeline.start()
    result = pipeline.process_utterance(audio_data)
    pipeline.stop()

    # 模式B: 持续监听 (后台线程)
    pipeline.start_listening()
    # ... 说指令 ...
    pipeline.stop_listening()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable, Dict, Any
from enum import Enum, auto

import numpy as np

from voice_router_lite.config import PipelineConfig, DEFAULT_CONFIG, estimate_engine_memory
from voice_router_lite.audio.capture import AudioCapture, save_wav
from voice_router_lite.audio.player import AudioPlayer
from voice_router_lite.audio.vad import VoiceActivityDetector
from voice_router_lite.audio.denoise import AudioPreprocessor
from voice_router_lite.kws.engine import KWSEngine, KWSResult
from voice_router_lite.asr.engine import ASREngine, ASRResult
from voice_router_lite.nlu.engine import NLUEngine
from voice_router_lite.nlu.model import NLUResult
from voice_router_lite.tts.engine import TTSEngine
from voice_router_lite.device.manager import DeviceManager, create_default_device_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 管道状态
# ---------------------------------------------------------------------------
class PipelineState(Enum):
    """管道状态机"""
    IDLE = auto()           # 空闲，等待唤醒
    LISTENING = auto()      # 监听中
    WAKED = auto()          # 已唤醒，等待指令
    RECORDING = auto()      # 正在录音
    PROCESSING = auto()     # 处理中 (ASR+NLU)
    EXECUTING = auto()      # 执行设备指令
    SPEAKING = auto()       # 播放 TTS 反馈


class VoiceRouterPipeline:
    """
    离线语音控制管道

    组件:
    - AudioCapture:    音频采集 (16kHz mono)
    - AudioPreprocessor: 降噪 + 回声消除 + AGC
    - VoiceActivityDetector: 语音活动检测
    - KWSEngine:       语音唤醒 (最小 3.3MB)
    - ASREngine:       语音识别 (最小 25MB)
    - NLUEngine:       指令理解 (最小 2MB)
    - TTSEngine:       语音反馈 (预录制拼接)
    - DeviceManager:   设备控制

    性能指标:
    - KWS 延迟: < 200ms
    - ASR 延迟: < 1.5s
    - NLU 延迟: < 100ms
    - 端到端: < 2s
    - 内存: 峰值引擎 ~48MB（ASR 按需加载/卸载）
    - 模型体积: ~41MB（ASR 25 + KWS 5 + NLU 11）

    使用方法:
        pipeline = VoiceRouterPipeline()
        pipeline.initialize()
        pipeline.start_listening()
        # 说 "小T小T，打开风扇"
        pipeline.stop_listening()
        pipeline.close()
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or DEFAULT_CONFIG

        # 音频层
        self._capture: Optional[AudioCapture] = None
        self._preprocessor: Optional[AudioPreprocessor] = None
        self._vad: Optional[VoiceActivityDetector] = None

        # AI 引擎层
        self._kws: Optional[KWSEngine] = None
        self._asr: Optional[ASREngine] = None
        self._nlu: Optional[NLUEngine] = None
        self._tts: Optional[TTSEngine] = None

        # 设备控制层
        self._device_manager: Optional[DeviceManager] = None
        self._scheduler = None

        # 状态管理
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # 事件回调 (支持多个)
        self._on_wake_callbacks: list[Callable[[KWSResult], None]] = []
        self._on_result_callbacks: list[Callable[[Dict[str, Any]], None]] = []
        self._on_error_callbacks: list[Callable[[str], None]] = []

        # 后台监听
        self._listening_thread: Optional[threading.Thread] = None
        self._running: bool = False

        # 录音缓冲（KWS 唤醒后收集指令）
        self._utterance_buffer: list[np.ndarray] = []
        self._max_utterance_duration_sec: float = 10.0
        self._wake_cooldown_until: float = 0.0
        self._skip_frames_until: float = 0.0
        self._last_speech_time: float = 0.0
        self._silence_start: float = 0.0
        self._speech_threshold: float = 0.008
        self._wake_cooldown_sec: float = 2.0

    # ==================================================================
    # 公开 API: 生命周期
    # ==================================================================

    def initialize(self) -> bool:
        """
        初始化管道中的所有组件。

        Returns:
            True 全部初始化成功
        """
        logger.info("=" * 50)
        logger.info("VoiceRouterPipeline 初始化开始...")
        logger.info("=" * 50)

        # 同步音频预处理开关
        self._config.audio.denoise_enabled = self._config.enable_denoise
        self._config.audio.aec_enabled = self._config.enable_aec

        # 可选 cgroup 限制（OpenWrt/Linux）
        if self._config.enable_cgroups:
            try:
                from voice_router_lite.platform.cgroups import apply_resource_limits
                apply_resource_limits(
                    cpu_quota_pct=self._config.cgroup_cpu_quota_pct,
                    memory_mb=self._config.cgroup_memory_mb,
                )
            except Exception as exc:
                logger.debug("cgroup 限制跳过: %s", exc)

        # 1. 音频捕获
        self._capture = AudioCapture(self._config.audio)
        self._capture.open()

        # 2. 降噪预处理
        self._preprocessor = AudioPreprocessor(self._config.audio)

        # 3. VAD
        if self._config.enable_vad:
            self._vad = VoiceActivityDetector(self._config.audio)
        else:
            self._vad = None

        # 4. KWS 唤醒词
        if self._config.enable_kws and self._config.wake_word_threshold >= 0:
            self._kws = KWSEngine(
                self._config.models,
                self._config.audio,
                wake_word_threshold=self._config.wake_word_threshold,
            )
            self._kws.initialize()
        else:
            self._kws = None

        # 5. ASR 语音识别（支持按需加载）
        self._asr = ASREngine(
            self._config.models,
            self._config.audio,
            num_threads=self._config.asr_num_threads,
        )
        if not self._config.asr_lazy_load:
            self._asr.initialize()

        # 6. NLU 指令理解
        self._nlu = NLUEngine(
            self._config.models,
            confidence_threshold=self._config.nlu_confidence_threshold,
        )
        self._nlu.initialize(prefer_int8=self._config.prefer_int8_nlu)

        # 7. TTS 语音反馈
        self._tts = TTSEngine(self._config.models, self._config.audio)
        self._tts.initialize()

        # 8. 设备管理
        self._device_manager = create_default_device_manager(
            use_openwrt_ubus=self._config.use_openwrt_ubus,
            include_mock_devices=(self._config.deployment_mode != "router"),
        )
        self._device_manager.initialize()

        # 9. 路由器量产：分时模型调度
        if self._config.deployment_mode == "router" or (
            self._config.model_serial_exclusive or self._config.asr_subprocess
        ):
            from voice_router_lite.router.model_scheduler import ModelScheduler
            self._scheduler = ModelScheduler(self._config, self._kws, self._asr)
            logger.info(
                "ModelScheduler 已启用: serial=%s subprocess=%s",
                self._config.model_serial_exclusive,
                self._config.asr_subprocess,
            )
        else:
            self._scheduler = None

        logger.info("VoiceRouterPipeline 初始化完成 ✅")
        self._log_memory_estimate()

    def _publish_daemon_memory(self, phase: str | None = None) -> None:
        """守护进程向快照文件写入当前 RSS（供 Web 控制台读取）。"""
        try:
            from voice_router_lite.web.router_memory import (
                PHASE_ASR,
                PHASE_STANDBY,
                process_tree_rss_mb,
                write_daemon_memory_snapshot,
            )
            from voice_router_lite.config import estimate_engine_memory

            ph = phase or (
                PHASE_ASR if self._router_memory_phase_asr() else PHASE_STANDBY
            )
            est = estimate_engine_memory(
                self._config,
                profile="router",
                prefer_int8_nlu=self._config.prefer_int8_nlu,
                phase=ph,
            )
            write_daemon_memory_snapshot(
                rss_mb=process_tree_rss_mb(),
                phase=ph,
                estimate=est,
            )
        except Exception:
            pass

    def _router_memory_phase_asr(self) -> bool:
        try:
            from voice_router_lite.web.monitor import get_monitor
            return get_monitor()._router_memory.phase == "asr_active"
        except Exception:
            return False

    def start(self) -> None:
        """启动管道（开始音频采集）"""
        if not self._capture or not self._capture.is_open:
            self._capture = AudioCapture(self._config.audio)
            self._capture.open()
        self._running = True
        self._set_state(PipelineState.IDLE)
        logger.info("管道已启动，等待唤醒...")

    def stop(self) -> None:
        """停止管道"""
        self._running = False
        if self._listening_thread:
            self._listening_thread.join(timeout=3.0)
            self._listening_thread = None
        self._set_state(PipelineState.IDLE)
        logger.info("管道已停止")

    def close(self) -> None:
        """释放所有资源"""
        self.stop()

        components = [
            ("KWS", self._kws),
            ("ASR", self._asr),
            ("TTS", self._tts),
            ("Capture", self._capture),
        ]
        for name, comp in components:
            if comp:
                try:
                    if name == "ASR" and hasattr(comp, "unload"):
                        comp.unload()
                    comp.close()
                except Exception as e:
                    logger.warning("关闭 %s 异常: %s", name, e)

        if self._device_manager:
            self._device_manager.close()

        logger.info("管道已完全关闭")

    # ==================================================================
    # 公开 API: 回调注册
    # ==================================================================

    def on_wake(self, callback: Callable[[KWSResult], None]) -> None:
        """注册唤醒回调 (收到唤醒词时调用，支持多次注册)"""
        self._on_wake_callbacks.append(callback)

    def on_result(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册结果回调 (指令执行完成时调用，支持多次注册)"""
        self._on_result_callbacks.append(callback)

    def on_error(self, callback: Callable[[str], None]) -> None:
        """注册错误回调 (支持多次注册)"""
        self._on_error_callbacks.append(callback)

    # ==================================================================
    # 公开 API: 工作模式
    # ==================================================================

    def process_utterance(self,
                          audio_data: Optional[np.ndarray] = None,
                          text: Optional[str] = None) -> Dict[str, Any]:
        """
        处理一条完整语音指令。

        Args:
            audio_data: 16kHz int16 音频数据，None 则直接使用 text
            text: 直接提供文本（跳过 ASR）

        Returns:
            {"success": bool, "intent": str, "slots": dict, "response": str, ...}
        """
        # 状态: 开始处理
        self._set_state(PipelineState.PROCESSING)

        # 1. ASR 识别
        if text is None and audio_data is not None:
            mem_monitor = None
            try:
                from voice_router_lite.web.monitor import get_monitor
                mem_monitor = get_monitor()
                mem_monitor.enter_router_asr()
            except Exception:
                mem_monitor = None
            try:
                if self._scheduler is not None:
                    text = self._scheduler.transcribe(audio_data)
                else:
                    self._ensure_asr()
                    asr_result = self._asr.transcribe(audio_data)
                    text = asr_result.text
                    if self._config.asr_unload_after_use:
                        self._asr.unload()
            finally:
                if mem_monitor is not None:
                    mem_monitor.exit_router_asr()
                if self._config.deployment_mode == "router":
                    self._publish_daemon_memory()
            logger.info("ASR 识别结果: '%s'", text)

        if not text or not text.strip():
            self._set_state(PipelineState.IDLE)
            return {"success": False, "message": "未识别到语音内容"}

        # 2. NLU 理解
        nlu_result = self._nlu.understand(text)

        if (
            self._config.deployment_mode == "router"
            and nlu_result.intent.startswith("router_")
            and nlu_result.confidence < self._config.nlu_confidence_threshold
        ):
            self._set_state(PipelineState.IDLE)
            msg = "没听清，请再说一次"
            if self._config.enable_tts_feedback:
                self._tts.speak_sentence(msg, blocking=False)
            return {
                "success": False,
                "text": text,
                "intent": nlu_result.intent,
                "confidence": nlu_result.confidence,
                "slots": nlu_result.slots,
                "message": msg,
            }

        # 3. 执行指令
        if nlu_result.is_valid:
            exec_result = self._device_manager.execute_command(
                nlu_result.intent, nlu_result.slots
            )

            # 4. TTS 反馈
            self._set_state(PipelineState.EXECUTING)
            if self._config.enable_tts_feedback:
                reply = exec_result.get("message") or text
                if exec_result.get("success"):
                    self._tts.speak_sentence(reply, blocking=False)
                else:
                    self._tts.speak_feedback(
                        nlu_result.intent, nlu_result.slots, blocking=False
                    )
        else:
            exec_result = {"success": False, "message": "无法理解指令"}
            self._set_state(PipelineState.IDLE)
            if self._config.enable_tts_feedback:
                self._tts.speak_sentence("无法识别", blocking=False)

        # 5. 组装结果
        result = {
            "success": exec_result.get("success", False),
            "text": text,
            "intent": nlu_result.intent,
            "confidence": nlu_result.confidence,
            "slots": nlu_result.slots,
            "message": exec_result.get("message", ""),
        }

        # 6. 回调通知
        for cb in self._on_result_callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error("结果回调异常: %s", e)

        self._set_state(PipelineState.IDLE)
        return result

    def start_listening(self) -> None:
        """
        启动持续监听模式（后台线程）。

        管线流程:
        1. 持续采集音频，做 VAD
        2. 检测到语音 → KWS 确认唤醒词
        3. 唤醒后 → 录音 → ASR → NLU → 执行 → TTS
        4. 回到监听状态
        """
        if self._listening_thread and self._listening_thread.is_alive():
            return

        self.start()
        self._listening_thread = threading.Thread(
            target=self._listening_loop,
            daemon=True,
        )
        self._listening_thread.start()
        logger.info("持续监听模式已启动")

    def stop_listening(self) -> None:
        """停止持续监听"""
        self._running = False
        if self._listening_thread:
            self._listening_thread.join(timeout=5.0)
            self._listening_thread = None

    # ==================================================================
    # 公开 API: 批量处理
    # ==================================================================

    def process_file(self, wav_path: str) -> Dict[str, Any]:
        """处理 WAV 文件"""
        from voice_router_lite.audio.capture import load_wav
        audio = load_wav(wav_path)
        return self.process_utterance(audio_data=audio)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """仅转写，不执行指令"""
        result = self._asr.transcribe(audio_data)
        return result.text

    # ==================================================================
    # 后台监听循环
    # ==================================================================

    def _listening_loop(self) -> None:
        """后台监听主循环"""
        use_kws = (
            self._config.enable_kws
            and self._config.wake_word_threshold >= 0
        )
        silence_timeout = self._config.audio.vad_silence_duration_ms / 1000.0

        if use_kws:
            self._set_state(PipelineState.LISTENING)
            logger.info("持续监听: KWS 唤醒模式")
        else:
            logger.info("持续监听: VAD 直通模式 (KWS 已禁用)")

        while self._running:
            try:
                chunk = self._capture.read()
                clean_chunk = self._preprocessor.process(chunk)

                if use_kws:
                    self._kws_listening_step(clean_chunk, silence_timeout)
                else:
                    self._vad_listening_step(clean_chunk, silence_timeout)

            except Exception as e:
                logger.error("监听循环异常: %s", e)
                for cb in self._on_error_callbacks:
                    try:
                        cb(str(e))
                    except Exception:
                        pass
                time.sleep(0.1)

    def _kws_listening_step(self, clean_chunk: np.ndarray,
                            silence_timeout: float) -> None:
        """KWS 唤醒 → 录音 → ASR 流程。"""
        state = self._get_state()
        now = time.time()
        sample_rate = self._config.audio.sample_rate

        kws = self._scheduler.kws if self._scheduler else self._kws
        if kws is None:
            return

        if state in (PipelineState.IDLE, PipelineState.LISTENING):
            self._set_state(PipelineState.LISTENING)
            kws_result = kws.detect(clean_chunk)
            if kws_result.detected and now > self._wake_cooldown_until:
                self._wake_cooldown_until = now + self._wake_cooldown_sec
                kws.reset()
                self._utterance_buffer = [clean_chunk.copy()]
                self._last_speech_time = now
                self._silence_start = 0.0
                self._skip_frames_until = now + 0.2
                self._set_state(PipelineState.RECORDING)
                for cb in self._on_wake_callbacks:
                    try:
                        cb(kws_result)
                    except Exception as exc:
                        logger.error("唤醒回调异常: %s", exc)
                logger.info("唤醒词 '%s' 已检测，开始录音", kws_result.keyword)

        elif state == PipelineState.RECORDING:
            if now < self._skip_frames_until:
                return

            self._utterance_buffer.append(clean_chunk)
            energy = float(np.sqrt(np.mean(clean_chunk ** 2)))
            if energy > self._speech_threshold:
                self._last_speech_time = now
                self._silence_start = 0.0
            elif self._silence_start == 0.0:
                self._silence_start = now

            total_dur = self._utterance_duration()
            silence_dur = now - self._silence_start if self._silence_start > 0 else 0.0

            if (
                (silence_dur > silence_timeout and self._last_speech_time > 0)
                or total_dur > self._max_utterance_duration_sec
            ):
                self._process_utterance_buffer()
                kws_after = self._scheduler.kws if self._scheduler else self._kws
                if kws_after:
                    kws_after.reset()
                self._set_state(PipelineState.LISTENING)

    def _vad_listening_step(self, clean_chunk: np.ndarray,
                            silence_timeout: float) -> None:
        """无 KWS 时的 VAD 直通流程。"""
        if self._vad is None:
            is_speech, speech_ended = True, False
        else:
            is_speech, speech_ended = self._vad.process_frame(clean_chunk)

        current_state = self._get_state()

        if current_state == PipelineState.IDLE:
            if is_speech:
                self._set_state(PipelineState.LISTENING)
                self._utterance_buffer = []

        elif current_state == PipelineState.LISTENING:
            self._utterance_buffer.append(clean_chunk)

            if speech_ended:
                self._process_utterance_buffer()
            elif self._utterance_duration() > self._max_utterance_duration_sec:
                self._process_utterance_buffer()

    def _ensure_asr(self) -> None:
        if self._asr is not None:
            self._asr.ensure_loaded()

    def _process_utterance_buffer(self) -> None:
        """处理录音缓冲"""
        if not self._utterance_buffer:
            self._set_state(PipelineState.IDLE)
            return

        # 合并音频
        full_audio = np.concatenate(self._utterance_buffer, dtype=np.float32)
        # 转回 int16
        audio_int16 = (full_audio * 32767).astype(np.int16)
        self._utterance_buffer = []

        # 进入处理
        self.process_utterance(audio_data=audio_int16)

    # ==================================================================
    # 状态管理
    # ==================================================================

    def _set_state(self, state: PipelineState) -> None:
        with self._state_lock:
            self._state = state

    def _get_state(self) -> PipelineState:
        with self._state_lock:
            return self._state

    # ==================================================================
    # 工具方法
    # ==================================================================

    def _utterance_duration(self) -> float:
        """计算当前录音缓冲时长 (秒)"""
        total_samples = sum(len(c) for c in self._utterance_buffer)
        return total_samples / self._config.audio.sample_rate

    def _log_memory_estimate(self) -> None:
        """估算内存占用，并同步到监控面板。

        仅计算离线引擎核心模块（KWS/ASR/NLU + 音频缓冲），
        不包含 Python 解释器、系统库等与路由器部署无关的开销。
        """
        est = estimate_engine_memory(
            self._config,
            profile="router",
            web_asr_loaded=False,
            prefer_int8_nlu=self._config.prefer_int8_nlu,
        )
        perf = self._config.performance
        if self._config.model_serial_exclusive:
            logger.info(
                "内存估算(互斥): 待机 KWS+NLU+Buf=%.0fMB | 峰值 NLU+ASR+Buf=%.0fMB",
                est.engine_standby_mb,
                est.engine_peak_mb,
            )
        else:
            logger.info(
                "内存估算: KWS=%.0f + NLU=%.0f + Buf=%.0f = %.0fMB",
                est.kws_mb,
                est.nlu_mb,
                est.buffer_mb,
                est.current_mb,
            )
        logger.info(
            "设备预算: 系统~%dMB + 引擎待机~%.0fMB / 峰值~%.0fMB ≈ %.0f / %.0fMB / %dMB",
            perf.system_reserved_memory_mb,
            est.engine_standby_mb,
            est.engine_peak_mb,
            est.device_standby_mb,
            est.device_peak_mb,
            perf.device_total_memory_mb,
        )
        logger.info("性能目标: KWS<%dms, NLU<%dms, 端到端<%dms",
                     perf.kws_latency_ms,
                     perf.nlu_latency_ms,
                     perf.total_latency_ms)

        # 同步引擎内存估算到监控面板
        try:
            from voice_router_lite.web.monitor import get_monitor
            get_monitor().configure_router_memory(
                self._config,
                prefer_int8=self._config.prefer_int8_nlu,
            )
            if self._config.deployment_mode == "router":
                self._publish_daemon_memory()
        except Exception:
            pass
