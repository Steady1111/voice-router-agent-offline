"""
路由器内存优化：KWS / ASR 互斥加载 + ASR 子进程隔离。

待机仅驻留 KWS + NLU；识别时卸载 KWS → 加载/拉起 ASR → 完成后反向恢复。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from voice_router_lite.config import PipelineConfig
from voice_router_lite.kws.engine import KWSEngine

if TYPE_CHECKING:
    from voice_router_lite.asr.engine import ASREngine

logger = logging.getLogger(__name__)


class ModelScheduler:
    """管理重模型的分时加载策略（128MB 路由器量产用）。"""

    def __init__(
        self,
        config: PipelineConfig,
        kws: Optional[KWSEngine],
        asr: Optional["ASREngine"],
    ) -> None:
        self._config = config
        self._kws = kws
        self._asr = asr

    @property
    def kws(self) -> Optional[KWSEngine]:
        return self._kws

    def unload_kws(self) -> None:
        if self._kws is None:
            return
        logger.info("ModelScheduler: 卸载 KWS 以释放内存")
        try:
            self._kws.close()
        except Exception as exc:
            logger.warning("KWS 卸载异常: %s", exc)
        self._kws = None

    def reload_kws(self) -> None:
        if not self._config.enable_kws or self._config.wake_word_threshold < 0:
            return
        if self._kws is not None:
            return
        logger.info("ModelScheduler: 重新加载 KWS")
        self._kws = KWSEngine(
            self._config.models,
            self._config.audio,
            wake_word_threshold=self._config.wake_word_threshold,
        )
        self._kws.initialize()

    def before_asr(self) -> None:
        if self._config.model_serial_exclusive:
            self.unload_kws()
        if not self._config.asr_subprocess and self._asr is not None:
            self._asr.ensure_loaded()

    def transcribe(self, audio_int16: np.ndarray) -> str:
        self.before_asr()
        try:
            if self._config.asr_subprocess:
                from voice_router_lite.router.asr_subprocess import transcribe_in_subprocess

                return transcribe_in_subprocess(audio_int16, self._config)
            if self._asr is None:
                return ""
            result = self._asr.transcribe(audio_int16)
            return result.text
        finally:
            self.after_asr()

    def after_asr(self) -> None:
        if (
            not self._config.asr_subprocess
            and self._asr is not None
            and self._config.asr_unload_after_use
        ):
            self._asr.unload()
        if self._config.model_serial_exclusive:
            self.reload_kws()
