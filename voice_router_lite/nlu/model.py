"""
指令理解模型定义 (NLU Model)

CNN+LSTM 架构用于意图分类和槽位提取。
支持从 ONNX 模型文件加载进行推理。

性能指标:
- 模型体积: 2MB
- 推理内存: 3MB
- 推理延迟: < 100ms
- 准确率: > 90% (10类意图)
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class NLUResult:
    """NLU 指令理解结果"""
    text: str                                # 原始输入文本
    intent: str = "unknown"                  # 意图名称
    intent_id: int = 0                       # 意图类别 ID
    confidence: float = 0.0                  # 置信度
    slots: Dict[str, str] = field(default_factory=dict)  # 槽位: {key: value}
    is_valid: bool = False                   # 是否有效指令

    def to_command(self) -> Dict:
        """转换为设备控制指令"""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "slots": self.slots,
        }


# ---------------------------------------------------------------------------
# 模型定义 (纯 numpy 推理)
# ---------------------------------------------------------------------------
class CNNLSTMNLU:
    """
    CNN + LSTM 指令理解模型 (numpy 推理版本)

    支持两种推理模式:
    1. "onnx": 从 ONNX 模型文件加载 (推荐，部署用)
    2. "numpy": 从 .npy 权重文件加载 (备选)

    使用方法:
        model = CNNLSTMNLU()
        model.load_onnx("models/nlu/intent_model.onnx")
        result = model.predict("打开客厅的风扇")
    """

    def __init__(self):
        # 资源配置
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.intent_labels: Dict[int, str] = {}
        self.slot_labels: Dict[int, str] = {}
        self.max_seq_len: int = 50
        self.embedding_dim: int = 128
        self.vocab_size: int = 5000

        # ONNX 会话
        self._onnx_session = None
        self._mode: str = "rule"  # "onnx" | "numpy" | "rule"

    # ------------------------------------------------------------------
    # 公开 API: 加载
    # ------------------------------------------------------------------

    def load_onnx(self, model_path: str,
                  vocab_path: str,
                  intent_labels_path: str,
                  slot_labels_path: Optional[str] = None) -> bool:
        """
        从 ONNX 文件加载模型。

        Args:
            model_path: ONNX 模型文件路径
            vocab_path: 词汇表 JSON ({"token": id})
            intent_labels_path: 意图标签 JSON ({"0": "set_device_state", ...})
            slot_labels_path: 槽位标签 JSON (可选)
        """
        try:
            import onnxruntime as ort

            # 加载词汇表
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)

            # 加载意图标签
            with open(intent_labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.intent_labels = {int(k): v for k, v in raw.items()}

            # 加载槽位标签
            if slot_labels_path and os.path.exists(slot_labels_path):
                with open(slot_labels_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.slot_labels = {int(k): v for k, v in raw.items()}

            # 创建 ONNX 推理会话
            self._onnx_session = ort.InferenceSession(model_path)
            self._mode = "onnx"

            logger.info(
                "NLU ONNX 模型已加载: intents=%d, vocab_size=%d",
                len(self.intent_labels), self.vocab_size,
            )
            return True

        except ImportError:
            logger.warning("onnxruntime 未安装，使用规则引擎模式")
            self._mode = "rule"
            return self._load_rules(intent_labels_path)
        except Exception as e:
            logger.error("NLU 模型加载失败: %s", e)
            self._mode = "rule"
            self._load_rules(intent_labels_path)
            return False

    def load_numpy(self, weights_path: str,
                   vocab_path: str,
                   intent_labels_path: str) -> bool:
        """从 numpy 权重文件加载 (备选)"""
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

            with open(intent_labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.intent_labels = {int(k): v for k, v in raw.items()}

            self._weights = np.load(weights_path, allow_pickle=True).item()
            self._mode = "numpy"
            logger.info("NLU numpy 模型已加载")
            return True
        except Exception as e:
            logger.error("NLU numpy 加载失败: %s", e)
            return False

    # ------------------------------------------------------------------
    # 公开 API: 推理
    # ------------------------------------------------------------------

    def predict(self, text: str) -> NLUResult:
        """
        对文本进行意图分类和槽位提取。

        Args:
            text: ASR 识别文本

        Returns:
            NLUResult 包含意图、槽位和置信度
        """
        text = text.strip()
        if not text:
            return NLUResult(text=text, intent="unknown",
                             confidence=0.0, is_valid=False)

        if self._mode == "onnx":
            return self._predict_onnx(text)
        elif self._mode == "numpy":
            return self._predict_numpy(text)
        else:
            return self._predict_rules(text)

    def predict_batch(self, texts: List[str]) -> List[NLUResult]:
        """批量推理"""
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    # 内部: ONNX 推理
    # ------------------------------------------------------------------

    def _predict_onnx(self, text: str) -> NLUResult:
        try:
            input_ids = self._tokenize(text)
            input_ids = np.array([input_ids], dtype=np.int64)

            outputs = self._onnx_session.run(
                None,
                {"input_ids": input_ids}
            )

            intent_logits = outputs[0]  # [1, num_intents]
            slot_logits = outputs[1] if len(outputs) > 1 else None  # [1, seq_len, num_slots]

            # 意图分类
            intent_id = int(np.argmax(intent_logits, axis=1)[0])
            intent_probs = self._softmax(intent_logits[0])
            confidence = float(intent_probs[intent_id])
            intent = self.intent_labels.get(intent_id, "unknown")

            # 槽位提取
            slots = {}
            if slot_logits is not None and self.slot_labels:
                slot_ids = np.argmax(slot_logits[0], axis=1)
                slots = self._decode_slots(text, slot_ids)

            return NLUResult(
                text=text,
                intent=intent,
                intent_id=intent_id,
                confidence=confidence,
                slots=slots,
                is_valid=intent != "unknown" and confidence > 0.5,
            )

        except Exception as e:
            logger.warning("ONNX 推理异常: %s，回退到规则引擎", e)
            return self._predict_rules(text)

    # ------------------------------------------------------------------
    # 内部: Numpy 推理
    # ------------------------------------------------------------------

    def _predict_numpy(self, text: str) -> NLUResult:
        # numpy 版本的前向传播（简易实现）
        # 生产环境推荐使用 ONNX
        try:
            input_ids = self._tokenize(text)
            x = np.array([input_ids])

            # Embedding
            embed = self._weights.get("embedding", np.eye(self.vocab_size)[:self.embedding_dim])
            x_emb = embed[x]

            # 简单的全连接分类 (如果没有完整的 LSTM 权重)
            if "fc_weight" in self._weights:
                x_flat = x_emb.mean(axis=1)  # mean pooling
                logits = x_flat @ self._weights["fc_weight"] + self._weights.get("fc_bias", 0)
                intent_id = int(np.argmax(logits))
                confidence = float(np.max(self._softmax(logits)))
            else:
                # 回退到规则
                return self._predict_rules(text)

            return NLUResult(
                text=text,
                intent=self.intent_labels.get(intent_id, "unknown"),
                intent_id=intent_id,
                confidence=confidence,
                slots=self._extract_rules_slots(text),
                is_valid=confidence > 0.5,
            )
        except Exception:
            return self._predict_rules(text)

    # ------------------------------------------------------------------
    # 内部: 规则引擎 (兜底)
    # ------------------------------------------------------------------

    def _load_rules(self, intent_labels_path: str) -> bool:
        """加载意图标签到规则引擎"""
        try:
            with open(intent_labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.intent_labels = {int(k): v for k, v in raw.items()}
        except Exception:
            self.intent_labels = {
                0: "set_device_state",
                1: "adjust_fan_speed",
                2: "set_light_brightness",
                3: "query_device_status",
                4: "timer_setting",
                5: "scene_mode",
                6: "router_reboot",
                7: "router_wifi_restart",
                8: "control_ac",
                9: "help",
            }
        return True

    # 意图关键词映射
    INTENT_KEYWORDS = {
        "set_device_state": ["打开", "关闭", "开", "关", "启动", "停止"],
        "adjust_fan_speed": ["调大", "调小", "风度", "风速", "大一点", "小一点"],
        "set_light_brightness": ["亮度", "亮一点", "暗一点", "调亮", "调暗"],
        "query_device_status": ["多少", "怎么样", "状态", "开着吗", "关了"],
        "timer_setting": ["定时", "分钟后", "小时后", "延迟"],
        "scene_mode": ["睡眠", "工作", "离开", "回家", "模式"],
        "router_reboot": ["重启路由", "重启网络", "网络卡"],
        "router_wifi_restart": ["重启WIFI", "WIFI断了", "重启wifi"],
        "control_ac": ["空调", "温度", "制冷", "制热", "调温"],
        "help": ["帮助", "能做什么", "功能", "怎么用"],
    }

    # 槽位关键词
    SLOT_KEYWORDS = {
        "fan": ["风扇"],
        "light": ["灯", "灯光", "照明"],
        "led": ["LED", "led"],
        "relay": ["插座", "开关", "继电器"],
        "ac": ["空调"],
        "on": ["开", "打开", "开启", "启动"],
        "off": ["关", "关闭", "停止", "熄"],
    }

    def _predict_rules(self, text: str) -> NLUResult:
        """规则引擎兜底推理"""
        text_lower = text.lower()

        # 1. 意图匹配
        best_intent = "unknown"
        best_score = 0

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_intent = intent

        # 2. 槽位提取
        slots = self._extract_rules_slots(text)

        confidence = min(best_score / 3.0, 1.0) if best_score > 0 else 0.0

        return NLUResult(
            text=text,
            intent=best_intent,
            intent_id=0,
            confidence=confidence,
            slots=slots,
            is_valid=best_intent != "unknown" and best_score > 0,
        )

    def _extract_rules_slots(self, text: str) -> Dict[str, str]:
        """基于规则的槽位提取"""
        slots = {}
        text_lower = text.lower()

        # 设备类型
        for dev, kws in {"fan": ["风扇"], "light": ["灯"], "led": ["led"],
                         "relay": ["插座", "继电器"], "ac": ["空调"]}.items():
            if any(kw.lower() in text_lower for kw in kws):
                slots["device_type"] = dev
                break

        # 状态
        if any(w in text for w in ["打开", "开启", "启动"]):
            slots["state"] = "on"
        elif any(w in text for w in ["关闭", "停止", "熄"]):
            slots["state"] = "off"

        # 调节方向（暗/小/低优先于 亮/大/高 判断）
        if any(w in text for w in ["暗一点", "暗些", "暗", "小", "低", "冷", "慢"]):
            slots["direction"] = "down"
        elif any(w in text for w in ["亮一点", "亮些", "亮", "大", "高", "热", "快"]):
            slots["direction"] = "up"

        # 场景模式
        modes = ["睡眠", "工作", "离开", "回家"]
        for mode in modes:
            if mode in text:
                slots["mode_name"] = mode
                break

        return slots

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[int]:
        """文本转 token IDs (字符级)"""
        tokens = []
        for ch in text[:self.max_seq_len]:
            # 字符级编码
            if ch in self.vocab:
                tokens.append(self.vocab[ch])
            else:
                tokens.append(self.vocab.get("<UNK>", 0))

        # 填充到 max_seq_len
        pad_id = self.vocab.get("<PAD>", 0)
        while len(tokens) < self.max_seq_len:
            tokens.append(pad_id)

        return tokens[:self.max_seq_len]

    def _decode_slots(self, text: str, slot_ids: np.ndarray) -> Dict[str, str]:
        """从 slot IDs 解码槽位"""
        slots = {}
        if not self.slot_labels:
            return slots

        current_slot = None
        current_tokens = []

        for i, sid in enumerate(slot_ids[:len(text)]):
            label = self.slot_labels.get(int(sid), "O")

            if label.startswith("B-"):
                if current_slot:
                    slots[current_slot] = "".join(current_tokens)
                current_slot = label[2:]  # 去掉 B- 前缀
                current_tokens = [text[i]]
            elif label.startswith("I-") and current_slot == label[2:]:
                current_tokens.append(text[i])
            else:
                if current_slot:
                    slots[current_slot] = "".join(current_tokens)
                    current_slot = None
                    current_tokens = []

        if current_slot:
            slots[current_slot] = "".join(current_tokens)

        return slots

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_stable = x - np.max(x)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x)
