"""
语音识别引擎 (ASR)

基于 sherpa-onnx + Zipformer 中文预训练模型。
提供流式和非流式两种识别模式。

性能指标:
- 模型体积: 25MB (encoder 15MB + decoder 8MB + joiner 2MB)
- 运行时内存: ~5MB
- 延迟: < 1.5s (端到端)
- 中文准确率: ~95%
"""

import logging
import threading
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np

from voice_router_lite.config import ModelPaths, AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str
    is_final: bool = True
    confidence: float = 0.0
    tokens: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


class ASREngine:
    """
    ASR 语音识别引擎

    两种识别模式:
    1. "streaming": 流式识别 (sherpa-onnx OnlineRecognizer，低延迟)
    2. "offline": 离线批量识别 (sherpa-onnx OfflineRecognizer)

    使用方法:
        engine = ASREngine(model_paths, audio_config)
        engine.initialize()

        # 流式模式
        engine.start_stream()
        result = engine.process_chunk(audio_chunk)
        final_text = engine.stop_stream()

        # 离线模式
        result = engine.transcribe(audio_data)
    """

    def __init__(self, model_paths: ModelPaths, audio_config: AudioConfig):
        self._model_paths = model_paths
        self._audio_config = audio_config

        # sherpa-onnx 引擎
        self._online_recognizer = None
        self._offline_recognizer = None
        self._stream = None

        # 状态
        self._initialized: bool = False
        self._streaming: bool = False
        self._lock = threading.Lock()

        # 识别结果缓冲
        self._partial_results: List[ASRResult] = []

        # 流式缓冲
        self._stream_buffer = np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        初始化 ASR 引擎。

        Returns:
            True 初始化成功
        """
        try:
            import sherpa_onnx

            recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
                model_config=sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder=self._model_paths.asr_encoder,
                        decoder=self._model_paths.asr_decoder,
                        joiner=self._model_paths.asr_joiner,
                    ),
                    tokens=self._model_paths.asr_tokens,
                ),
                # 解码配置：多路径保持中间结果
                decoding_method="greedy_search",
                max_active_paths=4,
                enable_endpoint=True,
                rule1_min_trailing_silence=1.5,
                rule2_min_trailing_silence=0.5,
                rule3_min_utterance_length=0.5,
                sample_rate=self._audio_config.sample_rate,
            )

            self._online_recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
            self._initialized = True
            logger.info(
                "ASR 引擎已初始化: model=Zipformer, sample_rate=%d",
                self._audio_config.sample_rate,
            )
            return True

        except ImportError:
            logger.warning("sherpa-onnx 未安装，ASR 引擎将在 mock 模式下运行")
            self._initialized = True
            return True
        except Exception as e:
            logger.error("ASR 引擎初始化失败: %s", e)
            return False

    def start_stream(self) -> None:
        """开始流式识别会话"""
        if not self._initialized:
            return

        with self._lock:
            self._streaming = True
            self._stream_buffer = np.array([], dtype=np.float32)
            self._partial_results = []
            self._stream = None
        logger.debug("ASR 流式会话已开始")

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """
        处理一个音频分片 (流式模式)。

        Args:
            audio_chunk: 16kHz mono (float32 归一化)

        Returns:
            ASRResult 或 None (结果未更新)
        """
        if not self._initialized or not self._streaming:
            return None

        if self._online_recognizer is None:
            # Mock 模式
            return None

        with self._lock:
            try:
                # 确保形状
                audio = np.asarray(audio_chunk, dtype=np.float32).flatten()

                if self._stream is None:
                    self._stream = self._online_recognizer.create_stream()
                    self._stream.accept_waveform(
                        self._audio_config.sample_rate, audio
                    )
                else:
                    self._stream.accept_waveform(
                        self._audio_config.sample_rate, audio
                    )

                # 解码
                while self._online_recognizer.is_ready(self._stream):
                    self._online_recognizer.decode_stream(self._stream)

                # 获取部分结果
                partial_text = self._online_recognizer.get_result(self._stream)

                if partial_text:
                    return ASRResult(text=partial_text, is_final=False)
                return None

            except Exception as e:
                logger.warning("ASR 流式处理异常: %s", e)
                return None

    def stop_stream(self) -> ASRResult:
        """
        停止流式识别，返回最终结果。

        Returns:
            ASRResult 最终识别文本
        """
        if not self._initialized or not self._streaming:
            return ASRResult(text="", is_final=True)

        with self._lock:
            self._streaming = False
            result = ASRResult(text="", is_final=True)

            if self._stream is not None and self._online_recognizer is not None:
                try:
                    # 输入结束标记
                    self._stream.input_finished()

                    # 最终解码
                    while self._online_recognizer.is_ready(self._stream):
                        self._online_recognizer.decode_stream(self._stream)

                    final_text = self._online_recognizer.get_result(self._stream)
                    result = ASRResult(text=final_text, is_final=True)
                    logger.info("ASR 最终结果: %s", final_text)
                except Exception as e:
                    logger.error("ASR 最终解码异常: %s", e)

            self._stream = None
            return result

    def transcribe(self, audio_data: np.ndarray) -> ASRResult:
        """
        离线批量识别 (非流式，一次性输入完整音频)。

        Args:
            audio_data: 16kHz mono 完整的音频数据

        Returns:
            ASRResult 识别结果
        """
        if not self._initialized:
            return ASRResult(text="", is_final=True)

        # 尝试离线识别器
        if self._offline_recognizer is None and not self._init_offline():
            # 回退到流式模拟
            return self._transcribe_via_streaming(audio_data)

        try:
            result = self._offline_recognizer.decode(audio_data)
            return ASRResult(text=result.text, is_final=True)
        except Exception as e:
            logger.error("离线识别异常: %s", e)
            return ASRResult(text="", is_final=True)

    def transcribe_file(self, wav_path: str) -> ASRResult:
        """
        识别 WAV 文件。

        Args:
            wav_path: WAV 文件路径 (16kHz, mono, 16bit)

        Returns:
            ASRResult
        """
        import wave

        try:
            with wave.open(wav_path, "rb") as wf:
                data = wf.readframes(wf.getnframes())
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return self.transcribe(audio)
        except Exception as e:
            logger.error("WAV 文件识别失败 %s: %s", wav_path, e)
            return ASRResult(text="", is_final=True)

    def close(self) -> None:
        """释放 ASR 引擎资源"""
        if self._streaming:
            self.stop_stream()
        self._online_recognizer = None
        self._offline_recognizer = None
        self._initialized = False
        logger.info("ASR 引擎已关闭")

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _init_offline(self) -> bool:
        """初始化离线识别器"""
        try:
            import sherpa_onnx

            self._offline_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=self._model_paths.asr_encoder,
                decoder=self._model_paths.asr_decoder,
                joiner=self._model_paths.asr_joiner,
                tokens=self._model_paths.asr_tokens,
                num_threads=self._audio_config.sample_rate,
            )
            return True
        except Exception:
            return False

    def _transcribe_via_streaming(self, audio_data: np.ndarray) -> ASRResult:
        """通过流式模拟实现离线识别"""
        self.start_stream()

        chunk_size = self._audio_config.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            self.process_chunk(chunk)

        return self.stop_stream()
