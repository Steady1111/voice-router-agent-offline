"""
指令理解引擎 (NLU Engine)

封装意图分类和槽位提取模型。
提供统一的 NLU 推理接口。

使用方法:
    engine = NLUEngine(model_paths)
    engine.initialize()
    result = engine.understand("打开客厅的风扇")
    print(result.intent, result.slots)
"""

import logging
from typing import Dict, List

from voice_router_lite.config import ModelPaths
from voice_router_lite.nlu.model import CNNLSTMNLU, NLUResult

logger = logging.getLogger(__name__)


class NLUEngine:
    """
    NLU 指令理解引擎

    封装模型加载、推理、后处理逻辑。

    使用方法:
        engine = NLUEngine(model_paths)
        engine.initialize()
        result = engine.understand(text)
    """

    def __init__(self, model_paths: ModelPaths,
                 confidence_threshold: float = 0.5):
        self._model_paths = model_paths
        self._confidence_threshold = confidence_threshold
        self._model = CNNLSTMNLU()
        self._initialized: bool = False

        # 指令后处理规则
        self._post_processors = [
            self._normalize_device_name,
            self._infer_missing_slots,
        ]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def initialize(self, prefer_int8: bool = False) -> bool:
        """
        初始化 NLU 引擎。

        ONNX 模型不可用时自动回退到规则引擎模式。

        Args:
            prefer_int8: 路由器部署时优先 INT8 量化模型

        Returns:
            True 初始化成功
        """
        model_path = self._model_paths.resolve_nlu_model(prefer_int8)
        try:
            ok = self._model.load_onnx(
                model_path=model_path,
                vocab_path=self._model_paths.nlu_vocab,
                intent_labels_path=self._model_paths.nlu_intent_labels,
                slot_labels_path=self._model_paths.nlu_slot_labels,
            )
            # 即使 ONNX 加载失败，规则引擎也能工作
            self._initialized = True
            logger.info("NLU 引擎已初始化: mode=%s model=%s", self._model._mode, model_path)
            if not ok:
                logger.info("NLU: ONNX 模型不可用，使用规则引擎兜底")
            return True
        except Exception as e:
            logger.error("NLU 引擎初始化异常: %s", e)
            # 规则引擎不需要模型文件即可工作
            self._model._load_rules(self._model_paths.nlu_intent_labels)
            self._initialized = True
            return True

    def understand(self, text: str) -> NLUResult:
        """
        理解用户指令。

        Args:
            text: ASR 识别的文本

        Returns:
            NLUResult 包含意图、槽位和置信度
        """
        if not self._initialized:
            return NLUResult(text=text)

        # 1. 模型推理
        result = self._model.predict(text)

        # 2. 低置信度回退规则引擎
        if (
            self._model._mode == "onnx"
            and result.confidence < self._confidence_threshold
        ):
            rule_result = self._model._predict_rules(text)
            if rule_result.is_valid and (
                not result.is_valid or rule_result.confidence >= result.confidence
            ):
                logger.info(
                    "NLU 低置信度 %.2f < %.2f，回退规则引擎",
                    result.confidence,
                    self._confidence_threshold,
                )
                result = rule_result

        # 3. 后处理
        for processor in self._post_processors:
            result = processor(result)

        logger.debug(
            "NLU: text='%s' → intent=%s, confidence=%.2f, slots=%s",
            text, result.intent, result.confidence, result.slots,
        )

        return result

    def understand_batch(self, texts: List[str]) -> List[NLUResult]:
        """批量理解"""
        return [self.understand(t) for t in texts]

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_device_name(result: NLUResult) -> NLUResult:
        """标准化设备名称"""
        device_type = result.slots.get("device_type", "").lower()

        aliases = {
            "风扇": "fan", "风机": "fan",
            "灯": "light", "灯光": "light", "照明": "light",
            "led": "led", "LED": "led",
            "空调": "ac", "冷气": "ac",
            "插座": "relay", "开关": "relay",
        }

        if device_type in aliases:
            result.slots["device_type"] = aliases[device_type]

        return result

    @staticmethod
    def _infer_missing_slots(result: NLUResult) -> NLUResult:
        """根据意图补全缺失槽位"""
        intent = result.intent
        slots = result.slots

        # 开关设备的默认状态 (V2: device_control，兼容 V1: set_device_state)
        if intent in ("device_control", "set_device_state"):
            if "state" not in slots:
                if any(w in result.text for w in ["开", "启动"]):
                    slots["state"] = "on"
                elif any(w in result.text for w in ["关", "停止", "熄"]):
                    slots["state"] = "off"
            # 标准化中文状态值 → 英文
            elif slots.get("state") in ("打开", "开启", "开"):
                slots["state"] = "on"
            elif slots.get("state") in ("关闭", "关掉", "关", "熄"):
                slots["state"] = "off"

        # 调节类指令 (V2: device_adjust，兼容 V1: adjust_fan_speed)
        if intent in ("device_adjust", "adjust_fan_speed"):
            if "speed_level" not in slots and "direction" in slots:
                direction = slots["direction"]
                # 中文方向 → up/down
                if any(w in direction for w in ["大", "高", "热", "快", "亮", "调大", "加大"]):
                    direction = "up"
                elif any(w in direction for w in ["小", "低", "冷", "慢", "暗", "调小", "减小"]):
                    direction = "down"
                slots["speed_delta"] = "1" if direction == "up" else "-1"

        return result
