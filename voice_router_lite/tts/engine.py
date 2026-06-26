"""
语音合成引擎 (TTS)

基于预录制音频片段的轻量级 TTS。
零运行时内存，完全离线，适合资源受限设备。

语音库设计:
- 按功能分类的预录制音频片段
- 支持拼接合成（"已打开" + "风扇"）
- WAV 格式存储，16kHz mono 16bit

使用方法:
    engine = TTSEngine(model_paths, audio_config)
    engine.initialize()
    engine.speak("fan_on")           # 播放预定义回复
    engine.speak_sentence("已打开风扇")  # 拼接合成
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np

from voice_router_lite.config import ModelPaths, AudioConfig
from voice_router_lite.audio.player import AudioPlayer

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    离线 TTS 引擎

    音频片段分类:
    - system:   系统提示 ("正在处理...", "无法识别", "你好")
    - feedback: 指令反馈 ("已打开", "已关闭", "已调到")
    - devices:  设备名称 ("风扇", "灯光", "LED", "空调")
    - numbers:  数字 ("一", "二", ... "十")
    - modes:    场景模式 ("睡眠模式", "工作模式")
    - tones:    提示音 (蜂鸣)

    使用方法:
        tts = TTSEngine(model_paths, audio_config)
        tts.initialize()
        tts.speak_sentence("已打开客厅的风扇")
    """

    def __init__(self, model_paths: ModelPaths, audio_config: AudioConfig):
        self._model_paths = model_paths
        self._audio_config = audio_config
        self._player = AudioPlayer(audio_config)

        # 音频片段缓存 {name: numpy_array}
        self._clips: Dict[str, np.ndarray] = {}
        # 片段索引 {category: {name: filepath}}
        self._index: Dict[str, Dict[str, str]] = {}
        self._initialized: bool = False

        # 默认提示音（纯代码生成，不需要文件）
        self._builtin_tones: Dict[str, Callable] = {
            "wake_beep": lambda: self._generate_tone(880, 100),
            "ok_beep": lambda: self._generate_tone(440, 80),
            "error_beep": lambda: self._generate_tone(220, 200),
            "processing_beep": lambda: self._generate_tone_double(660, 50, 880, 50),
        }

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def initialize(self, *, open_player: bool = True) -> bool:
        """
        初始化 TTS 引擎。

        Args:
            open_player: False 时不打开本机扬声器（Web 服务仅需合成音频）

        Returns:
            True 初始化成功
        """
        if open_player:
            self._player.open()

        # 加载音频索引
        index_path = self._model_paths.tts_index
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                logger.info("TTS 音频索引已加载: %d 个分类", len(self._index))
            except Exception as e:
                logger.warning("TTS 索引加载失败: %s", e)

        # 预加载常用片段
        self._preload_essential_clips()

        self._initialized = True
        logger.info("TTS 引擎已初始化: clips=%d", len(self._clips))
        return True

    def speak(self, clip_name: str, blocking: bool = True) -> None:
        """
        播放指定音频片段。

        Args:
            clip_name: 片段名称
            blocking: True 阻塞播放
        """
        audio = self._get_clip(clip_name)
        if audio is not None:
            self._player.play(audio, blocking=blocking)
        elif clip_name in self._builtin_tones:
            tone = self._builtin_tones[clip_name]()
            if tone is not None:
                self._player.play(tone, blocking=blocking)
        else:
            logger.debug("TTS 片段不存在: %s", clip_name)

    def speak_sentence(self, text: str, blocking: bool = True) -> None:
        """
        拼接播放完整语句。

        拼接策略:
        - "已打开风扇" → "已打开" + "风扇"
        - "无法识别" → "无法识别"
        - 未预定义的文本 → 播放错误音

        Args:
            text: 要播放的文本
            blocking: True 阻塞
        """
        clips = self._synthesize(text)

        if not clips:
            logger.debug("TTS 无法合成: %s", text)
            self.speak("error_beep", blocking=False)
            return

        for clip_name in clips:
            self.speak(clip_name, blocking=True)

    def speak_feedback(self, intent: str, slots: Dict[str, str],
                       blocking: bool = True) -> None:
        """
        根据 NLU 结果播放反馈语音。

        Args:
            intent: 意图名称
            slots: 槽位字典
            blocking: True 阻塞
        """
        text = self._build_feedback_text(intent, slots)
        if text:
            self.speak_sentence(text, blocking=blocking)

    def add_clip(self, name: str, audio: np.ndarray) -> None:
        """
        动态添加音频片段。

        Args:
            name: 片段名称
            audio: int16 音频数据
        """
        self._clips[name] = np.asarray(audio, dtype=np.int16)

    def beep_ok(self) -> None:
        """播放确认音"""
        self.speak("ok_beep", blocking=False)

    def beep_error(self) -> None:
        """播放错误音"""
        self.speak("error_beep", blocking=False)

    def beep_wake(self) -> None:
        """播放唤醒音"""
        self.speak("wake_beep", blocking=False)

    @property
    def sample_rate(self) -> int:
        return self._audio_config.sample_rate

    def render_tone(self, name: str) -> Optional[np.ndarray]:
        """合成提示音（不播放）。"""
        if name in self._builtin_tones:
            return self._builtin_tones[name]()
        clip = self._get_clip(name)
        return clip.copy() if clip is not None else None

    def render_sentence(self, text: str) -> Optional[np.ndarray]:
        """拼接合成整句音频（不播放）。"""
        clip_names = self._synthesize(text)
        if not clip_names:
            return None
        gap = np.zeros(int(self.sample_rate * 0.08), dtype=np.int16)
        parts: list[np.ndarray] = []
        for name in clip_names:
            audio = self._get_clip(name)
            if audio is None and name in self._builtin_tones:
                audio = self._builtin_tones[name]()
            if audio is None:
                continue
            if parts:
                parts.append(gap)
            parts.append(np.asarray(audio, dtype=np.int16))
        if not parts:
            return None
        return np.concatenate(parts)

    def close(self) -> None:
        """释放资源"""
        self._player.close()
        self._clips.clear()
        self._initialized = False
        logger.info("TTS 引擎已关闭")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_clip(self, name: str) -> Optional[np.ndarray]:
        """获取音频片段数据"""
        # 先查缓存
        if name in self._clips:
            return self._clips[name]

        # 查索引
        for category, clips in self._index.items():
            if name in clips:
                filepath = clips[name]
                audio = self._load_wav(filepath)
                if audio is not None:
                    self._clips[name] = audio
                    return audio

        # 尝试直接加载
        filepath = os.path.join(self._model_paths.tts_clips_dir, f"{name}.wav")
        if os.path.exists(filepath):
            audio = self._load_wav(filepath)
            if audio is not None:
                self._clips[name] = audio
                return audio

        return None

    def _preload_essential_clips(self) -> None:
        """预加载常用片段"""
        essential = ["ok_beep", "error_beep"]

        # 如果能加载到预录制片段就预加载
        for name in ["已打开", "已关闭", "无法识别"]:
            filepath = os.path.join(self._model_paths.tts_clips_dir, f"{name}.wav")
            if os.path.exists(filepath):
                essential.append(name)

        for name in essential:
            if name not in self._builtin_tones:
                clip = self._get_clip(name)
                if clip is not None:
                    self._clips[name] = clip

    def _synthesize(self, text: str) -> List[str]:
        """
        文本拼接合成。

        返回可播放的片段名称列表。
        """
        # 尝试精确匹配
        if self._get_clip(text) is not None:
            return [text]

        clips = []

        # 拼接规则
        phrases = [
            ("已打开", ["已打开"]),
            ("已关闭", ["已关闭"]),
            ("已调到", ["已调到"]),
            ("正在处理", ["正在处理"]),
            ("无法识别", ["无法识别"]),
            ("你好", ["你好"]),
            ("再见", ["再见"]),
            # 拼接模式
            ("已打开风扇", ["已打开", "风扇"]),
            ("已关闭风扇", ["已关闭", "风扇"]),
            ("已打开灯", ["已打开", "灯光"]),
            ("已关闭灯", ["已关闭", "灯光"]),
        ]

        for phrase, parts in phrases:
            if text == phrase or phrase in text:
                for part in parts:
                    if self._get_clip(part) is not None:
                        clips.append(part)
                if clips:
                    return clips

        # 无法合成
        return []

    @staticmethod
    def _build_feedback_text(intent: str, slots: Dict[str, str]) -> str:
        """根据意图生成中文反馈文本"""
        device_name_map = {
            "fan": "风扇", "light": "灯光", "led": "LED",
            "relay": "插座", "ac": "空调",
        }
        device = device_name_map.get(slots.get("device_type", ""), "设备")

        feedback_map = {
            "device_control": f"已{slots.get('state', '切换')}{device}",
            "device_adjust": "已调整风扇速度",
            "device_query": f"{device}状态查询",
            "timer_setting": "定时已设置",
            "scene_mode": f"已切换到{slots.get('mode_name', '')}模式",
            "set_device_state": f"已{slots.get('state', '切换')}{device}",
            "adjust_fan_speed": "已调整风扇速度",
            "set_light_brightness": "已调整灯光亮度",
            "query_device_status": f"{device}状态查询",
            "router_reboot": "正在重启路由器",
            "router_wifi_restart": "正在重启WiFi",
            "control_ac": "已调整空调设置",
            "help": "我可以控制风扇、灯光、LED等设备",
        }

        return feedback_map.get(intent, "")

    @staticmethod
    def _load_wav(filepath: str) -> Optional[np.ndarray]:
        """加载 WAV 文件"""
        import wave
        try:
            with wave.open(filepath, "rb") as wf:
                data = wf.readframes(wf.getnframes())
            return np.frombuffer(data, dtype=np.int16).copy()
        except Exception as e:
            logger.debug("WAV 加载失败 %s: %s", filepath, e)
            return None

    @staticmethod
    def _generate_tone(frequency: float, duration_ms: int) -> np.ndarray:
        """生成纯音"""
        sample_rate = 16000
        num_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        tone = (np.sin(2 * np.pi * frequency * t) * 8192).astype(np.int16)  # 50% 音量
        # 快速渐入渐出
        fade_len = min(50, num_samples // 4)
        if fade_len > 0:
            fade = np.linspace(0, 1, fade_len)
            tone[:fade_len] = (tone[:fade_len] * fade).astype(np.int16)
            tone[-fade_len:] = (tone[-fade_len:] * fade[::-1]).astype(np.int16)
        return tone

    @staticmethod
    def _generate_tone_double(f1: float, d1: int,
                               f2: float, d2: int) -> np.ndarray:
        """生成双音"""
        t1 = TTSEngine._generate_tone(f1, d1)
        t2 = TTSEngine._generate_tone(f2, d2)
        gap = np.zeros(1600, dtype=np.int16)  # 100ms 间隔
        return np.concatenate([t1, gap, t2])


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """int16 PCM mono → WAV bytes (for browser playback)."""
    import io
    import wave

    pcm = np.asarray(audio, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
