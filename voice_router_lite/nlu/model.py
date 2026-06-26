"""
指令理解模型定义 (NLU Model)

CNN+LSTM 架构用于意图分类和槽位提取。
支持从 ONNX 模型文件加载进行推理。

意图重心：路由器运维管理（10 个意图，~65%）+ 智能家居设备控制（6 个意图，~35%）

性能指标:
- 模型体积: ~10MB (fp32) / ~2-3MB (量化目标)
- 推理内存: 3-5MB
- 推理延迟: < 100ms
- 意图数: 16（路由器10 + 智能家居6）
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
            token_ids = self._tokenize(text)
            input_ids = np.array([token_ids], dtype=np.int64)
            # attention_mask: 1 for real tokens, 0 for padding
            attention_mask = np.array(
                [[1 if t != self.vocab.get("<PAD>", 0) else 0 for t in token_ids]],
                dtype=np.int64,
            )

            # 动态检测 ONNX 模型输入名称
            onnx_inputs = [inp.name for inp in self._onnx_session.get_inputs()]
            feed_dict = {"input_ids": input_ids}
            if "attention_mask" in onnx_inputs:
                feed_dict["attention_mask"] = attention_mask

            outputs = self._onnx_session.run(None, feed_dict)

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
                0: "router_reboot", 1: "router_wifi_restart", 2: "router_wifi_config",
                3: "router_network_query", 4: "router_led_control", 5: "router_device_manage",
                6: "router_network_diag", 7: "router_system", 8: "router_qos", 9: "router_security",
                10: "device_control", 11: "device_adjust", 12: "device_query",
                13: "timer_setting", 14: "scene_mode", 15: "help",
            }
        return True

    # 意图关键词映射（使用长度加权评分，更长的匹配更精确）
    INTENT_KEYWORDS = {
        # 🔵 路由器核心
        "router_reboot": ["重启路由", "重启网络", "网络卡", "网卡了", "路由器重启", "死机了",
                          "网络重启", "重启一下网络", "重新启动路由", "重启一下", "没响应了"],
        "router_wifi_restart": ["重启WIFI", "WIFI断了", "重启wifi", "WIFI重启", "Wi-Fi重启",
                                "无线重启", "WiFi连不上", "无线断了", "WiFi重开", "重新打开WiFi",
                                "关闭WiFi", "打开WiFi"],
        "router_wifi_config": ["改WiFi密码", "WiFi密码", "WiFi改个名字", "WiFi改名字",
                               "改个密码", "修改密码", "WiFi改名", "无线密码",
                               "访客WiFi", "访客网络", "打开访客", "关闭访客",
                               "WiFi名字", "设置WiFi", "WiFi设置", "5G频段", "2.4G频段",
                               "WiFi信道", "WiFi模式", "SSID改"],
        "router_network_query": ["IP地址", "IP是多少", "谁连了", "连了几个", "网速",
                                 "我的IP", "路由器IP", "连接设备", "设备列表",
                                 "网速怎么样", "运行多久", "温度多少", "内存剩多少",
                                 "CPU负载", "联网设备", "宽带速度"],
        "router_led_control": ["路由器灯", "路由器的灯", "路由器LED", "路由器的LED",
                               "指示灯", "LED灯", "熄灭", "点亮"],
        "router_device_manage": ["踢掉", "拉黑", "屏蔽", "禁止上网", "取消拉黑",
                                 "解除屏蔽", "恢复上网", "允许联网", "黑名单", "屏蔽列表"],
        "router_network_diag": ["测速", "测一下网速", "网络测速", "延迟多少", "Ping一下",
                                "延迟怎么样", "丢包率", "检查丢包", "DNS正常", "DNS解析",
                                "网络诊断", "全面诊断", "一键诊断"],
        "router_system": ["固件更新", "有新固件", "固件升级", "更新系统", "备份配置",
                          "备份设置", "导出配置", "恢复出厂", "恢复默认设置", "重置路由器",
                          "系统日志", "定时重启", "凌晨重启"],
        "router_qos": ["限速", "限制网速", "限制带宽", "优先", "提速", "更高的优先级",
                       "QoS", "带宽管理", "流量控制", "50兆", "100兆", "10兆", "20兆"],
        "router_security": ["防火墙", "安全防护", "打开VPN", "关闭VPN", "VPN服务",
                            "家长控制", "上网时间管理", "儿童模式", "青少年模式",
                            "防蹭网", "局域网安全", "安全日志", "攻击记录"],
        # 🏠 智能家居扩展
        "device_control": ["打开", "关闭", "启动", "停止", "关掉", "开一下", "关一下"],
        "device_adjust": ["调大", "调小", "风速", "大一点", "小一点", "亮度", "亮一点",
                          "暗一点", "调亮", "调暗", "温度调高", "温度调低"],
        "device_query": ["开着吗", "关了吗", "状态怎么样", "的状态", "查看一下"],
        "timer_setting": ["定时", "分钟后", "小时后", "延迟", "分钟关闭", "分钟打开"],
        "scene_mode": ["睡眠", "工作", "离开", "回家", "节能", "模式"],
        "help": ["帮助", "能做什么", "功能", "怎么用", "功能列表"],
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

        # 1. 意图匹配（使用关键词长度加权评分）
        best_intent = "unknown"
        best_score = 0

        for intent, keywords in self.INTENT_KEYWORDS.items():
            # 按关键词长度加权：更长的匹配更精确
            score = sum(len(kw) for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_intent = intent

        # 2. 槽位提取
        slots = self._extract_rules_slots(text)

        confidence = min(best_score / 10.0, 1.0) if best_score > 0 else 0.0

        return NLUResult(
            text=text,
            intent=best_intent,
            intent_id=0,
            confidence=confidence,
            slots=slots,
            is_valid=best_intent != "unknown" and best_score > 0,
        )

    def _extract_rules_slots(self, text: str) -> Dict[str, str]:
        """基于规则的槽位提取（ONNX 模型推理的兜底）"""
        slots = {}
        text_lower = text.lower()
        import re

        # 设备类型
        for dev, kws in {"fan": ["风扇"], "light": ["灯"], "led": ["led"],
                         "relay": ["插座", "继电器"], "ac": ["空调"]}.items():
            if any(kw.lower() in text_lower for kw in kws):
                slots["device_type"] = dev
                break

        # 状态
        if any(w in text for w in ["打开", "开启", "启动", "启用"]):
            slots["state"] = "on"
        elif any(w in text for w in ["关闭", "关掉", "关了", "停止", "熄", "禁用"]):
            slots["state"] = "off"

        # 调节方向（暗/小/低优先于 亮/大/高 判断）
        if any(w in text for w in ["暗一点", "暗些", "暗", "小", "低", "冷", "慢"]):
            slots["direction"] = "down"
        elif any(w in text for w in ["亮一点", "亮些", "亮", "大", "高", "热", "快"]):
            slots["direction"] = "up"

        # 定时时长: "[数字][分钟/小时/秒]后"
        duration_match = re.search(
            r'(\d+)\s*(分钟|小时|秒|minute|hour|second|min|h|s)', text
        )
        if duration_match:
            slots["timer_duration"] = duration_match.group(0).replace(" ", "")

        # 场景模式
        modes = ["睡眠", "工作", "离开", "回家", "节能"]
        for mode in modes:
            if mode in text:
                slots["mode_name"] = mode
                break

        # 🔵 路由器查询类型
        if any(w in text for w in ["IP地址", "IP是多少", "IP", "地址"]):
            slots["query_type"] = "ip"
        elif any(w in text for w in ["谁连了", "连了几个", "连接设备", "设备列表", "多少设备",
                                      "联网设备", "连接的设备"]):
            slots["query_type"] = "devices"
        elif any(w in text for w in ["网速", "测速", "速度", "宽带"]):
            slots["query_type"] = "speed"
        elif any(w in text for w in ["运行多久", "温度", "内存", "CPU负载"]):
            slots["query_type"] = "system_status"

        # 🔵 WiFi 配置类型
        if any(w in text for w in ["密码", "password"]):
            slots["config_type"] = "password"
        elif any(w in text for w in ["名字", "名称", "SSID"]):
            slots["config_type"] = "ssid"
        elif any(w in text for w in ["访客", "guest"]):
            slots["config_type"] = "guest"
        elif any(w in text for w in ["5G", "5G频段"]):
            slots["config_type"] = "5g_band"
        elif any(w in text for w in ["2.4G", "2.4G频段"]):
            slots["config_type"] = "2.4g_band"
        elif any(w in text for w in ["信道", "channel"]):
            slots["config_type"] = "channel"
        elif any(w in text for w in ["模式"]):
            slots["config_type"] = "mode"

        # 🔵 设备管理动作
        if any(w in text for w in ["踢掉", "踢"]):
            slots["manage_action"] = "kick"
        elif any(w in text for w in ["拉黑", "屏蔽", "禁止上网"]):
            slots["manage_action"] = "block"
        elif any(w in text for w in ["取消拉黑", "解除屏蔽", "恢复上网", "允许联网"]):
            slots["manage_action"] = "unblock"
        elif any(w in text for w in ["限速"]):
            slots["manage_action"] = "limit_speed"
        elif any(w in text for w in ["查看", "列出", "显示"]) and any(w in text for w in ["黑名单", "屏蔽列表", "已拉黑"]):
            slots["manage_action"] = "list_blocked"

        # 🔵 网络诊断类型
        if any(w in text for w in ["诊断", "全面诊断", "一键诊断"]):
            slots["diag_type"] = "full_diag"
        elif any(w in text for w in ["测速"]):
            slots["diag_type"] = "speed_test"
        elif any(w in text for w in ["延迟", "Ping", "ping"]):
            slots["diag_type"] = "latency"
        elif any(w in text for w in ["丢包"]):
            slots["diag_type"] = "packet_loss"
        elif any(w in text for w in ["DNS", "dns"]):
            slots["diag_type"] = "dns"

        # 🔵 系统运维动作
        if any(w in text for w in ["固件更新", "有新固件", "固件升级", "更新系统"]):
            slots["sys_action"] = "firmware_update"
        elif any(w in text for w in ["备份配置", "备份设置", "导出配置", "保存配置"]):
            slots["sys_action"] = "backup"
        elif any(w in text for w in ["恢复出厂", "恢复默认设置", "重置路由器", "恢复出厂设置"]):
            slots["sys_action"] = "factory_reset"
        elif any(w in text for w in ["系统日志", "异常"]):
            slots["sys_action"] = "view_logs"
        elif any(w in text for w in ["定时重启", "凌晨重启"]):
            slots["sys_action"] = "scheduled_reboot"

        # 🔵 QoS 动作
        if any(w in text for w in ["优先", "提速"]):
            slots["qos_action"] = "prioritize"
        elif any(w in text for w in ["限速", "限制"]):
            slots["qos_action"] = "limit"

        # 🔵 带宽值提取
        bw_match = re.search(r'(\d+)\s*[兆M]', text)
        if bw_match:
            slots["bandwidth_value"] = bw_match.group(0).replace(" ", "")

        # 🔵 安全策略类型
        if any(w in text for w in ["防火墙", "安全防护"]):
            slots["security_type"] = "firewall"
        elif any(w in text for w in ["VPN", "vpn"]):
            slots["security_type"] = "vpn"
        elif any(w in text for w in ["家长控制", "上网时间管理", "儿童模式", "青少年模式"]):
            slots["security_type"] = "parental_control"
        elif any(w in text for w in ["防蹭网", "局域网安全"]):
            slots["security_type"] = "lan_security"
        elif any(w in text for w in ["安全日志", "攻击记录"]):
            slots["security_type"] = "security_logs"

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
