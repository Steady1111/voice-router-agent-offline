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

import logging
import threading
import time
from typing import Optional, Callable, Dict, Any
from enum import Enum, auto

import numpy as np

from voice_router_lite.config import PipelineConfig, DEFAULT_CONFIG
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
    - 内存: < 50MB
    - 模型体积: < 35MB

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

        # 状态管理
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()

        # 事件回调
        self._on_wake: Optional[Callable[[KWSResult], None]] = None
        self._on_result: Optional[Callable[[Dict[str, Any]], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None

        # 后台监听
        self._listening_thread: Optional[threading.Thread] = None
        self._running: bool = False

        # 录音缓冲
        self._utterance_buffer: list[np.ndarray] = []
        self._max_utterance_duration_sec: float = 10.0  # 最长录音 10 秒

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

        # 1. 音频捕获
        self._capture = AudioCapture(self._config.audio)
        self._capture.open()

        # 2. 降噪预处理
        self._preprocessor = AudioPreprocessor(self._config.audio)

        # 3. VAD
        self._vad = VoiceActivityDetector(self._config.audio)

        # 4. KWS 唤醒词
        if self._config.enable_kws:
            self._kws = KWSEngine(self._config.models, self._config.audio)
            self._kws.initialize()

        # 5. ASR 语音识别
        self._asr = ASREngine(self._config.models, self._config.audio)
        self._asr.initialize()

        # 6. NLU 指令理解
        self._nlu = NLUEngine(self._config.models)
        self._nlu.initialize()

        # 7. TTS 语音反馈
        self._tts = TTSEngine(self._config.models, self._config.audio)
        self._tts.initialize()

        # 8. 设备管理
        self._device_manager = create_default_device_manager()
        self._device_manager.initialize()

        logger.info("VoiceRouterPipeline 初始化完成 ✅")
        self._log_memory_estimate()
        return True

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
        """注册唤醒回调 (收到唤醒词时调用)"""
        self._on_wake = callback

    def on_result(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册结果回调 (指令执行完成时调用)"""
        self._on_result = callback

    def on_error(self, callback: Callable[[str], None]) -> None:
        """注册错误回调"""
        self._on_error = callback

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
            asr_result = self._asr.transcribe(audio_data)
            text = asr_result.text
            logger.info("ASR 识别结果: '%s'", text)

        if not text or not text.strip():
            self._set_state(PipelineState.IDLE)
            return {"success": False, "message": "未识别到语音内容"}

        # 2. NLU 理解
        nlu_result = self._nlu.understand(text)

        # 3. 执行指令
        if nlu_result.is_valid:
            exec_result = self._device_manager.execute_command(
                nlu_result.intent, nlu_result.slots
            )

            # 4. TTS 反馈
            self._set_state(PipelineState.EXECUTING)
            if self._config.enable_tts_feedback:
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
        if self._on_result:
            try:
                self._on_result(result)
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
        has_wake_word = self._kws is not None
        skip_kws = self._config.wake_word_threshold < 0  # 跳过唤醒词

        while self._running:
            try:
                # 读取音频块
                chunk = self._capture.read()

                # 降噪预处理
                clean_chunk = self._preprocessor.process(chunk)

                # VAD 检测
                is_speech, speech_ended = self._vad.process_frame(clean_chunk)

                current_state = self._get_state()

                if current_state == PipelineState.IDLE:
                    if is_speech:
                        # 检测到语音，进入监听
                        self._set_state(PipelineState.LISTENING)
                        self._utterance_buffer = []

                elif current_state == PipelineState.LISTENING:
                    self._utterance_buffer.append(clean_chunk)

                    if speech_ended:
                        # 语音段结束，开始处理
                        self._process_utterance_buffer()

                    elif self._utterance_duration() > self._max_utterance_duration_sec:
                        # 超时，强制处理
                        self._process_utterance_buffer()

            except Exception as e:
                logger.error("监听循环异常: %s", e)
                if self._on_error:
                    self._on_error(str(e))
                time.sleep(0.1)

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
        """估算内存占用"""
        perf = self._config.performance
        total = (
            perf.kws_model_memory_mb
            + perf.asr_model_memory_mb
            + perf.nlu_model_memory_mb
            + perf.audio_buffer_memory_mb
        )
        logger.info("内存估算: KWS=%dMB + ASR=%dMB + NLU=%dMB + Buffer=%dMB = %dMB",
                     perf.kws_model_memory_mb,
                     perf.asr_model_memory_mb,
                     perf.nlu_model_memory_mb,
                     perf.audio_buffer_memory_mb,
                     total)
        logger.info("性能目标: KWS<%dms, ASR<%dms, NLU<%dms, 端到端<%dms",
                     perf.kws_latency_ms,
                     perf.asr_latency_ms,
                     perf.nlu_latency_ms,
                     perf.total_latency_ms)
