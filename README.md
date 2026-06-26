# Voice Router Lite

> 离线语音控制引擎 | 为瘦设备（128MB MIPS 路由器）设计

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 概述

Voice Router Lite 是一个**完全离线**的语音控制引擎，专为资源受限设备设计。所有语音处理逻辑在本地设备端完成，无任何网络依赖。

### 核心功能

| 功能 | 描述 | 模型 | 延迟 |
|------|------|------|------|
| **语音唤醒 (KWS)** | 自定义唤醒词检测 | sherpa-onnx KWS (3.3MB) | < 200ms |
| **语音识别 (ASR)** | 中文流式识别 | sherpa-onnx Zipformer zh 14M (25MB) | < 1.5s |
| **指令理解 (NLU)** | 意图分类 + 槽位提取 | CNN+LSTM (11MB fp32) | < 100ms |
| **语音合成 (TTS)** | 预录制音频拼接 | 无运行时内存 | < 300ms |
| **降噪处理** | 频谱门降噪 + 回声消除 | 无模型 | < 5ms/帧 |
| **设备控制** | GPIO/Relay/Fan/LED | - | - |

### 性能指标

- **峰值引擎内存**: ~48MB（KWS 8 + ASR 25 + NLU 11 + 缓冲 5）
- **待机内存**: ~76MB（含系统 ~60MB）
- **模型磁盘体积**: ~41MB（ASR 25 + KWS 5 + NLU 11）
- **端到端延迟**: < 2.0s
- **离线运行**: 零网络依赖（预研阶段 ESP32 经 WiFi 传 PCM）

## 架构

```
┌──────────────────────────────────────────────────────┐
│                 VoiceRouterPipeline                  │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ 音频采集 │ → │ 降噪处理 │ → │  VAD     │        │
│  │(I2S/USB) │   │(频谱门)  │   │(WebRTC)  │        │
│  └──────────┘   └──────────┘   └─────┬────┘        │
│                                      ↓               │
│  ┌──────────┐   ┌──────────┐   ┌─────┴────┐        │
│  │ 设备控制 │ ← │  NLU     │ ← │   ASR    │        │
│  │(GPIO/WS) │   │(CNN+LSTM)│   │(sherpa)  │        │
│  └──────────┘   └──────────┘   └──────────┘        │
│        ↓                               ↑             │
│  ┌──────────┐                          │             │
│  │   TTS    │                          │             │
│  │(预录制)  │                          │             │
│  └──────────┘                          │             │
└──────────────────────────────────────────────────────┘
        ↓                    ↓                  ↑
   [I2S DAC]           [继电器/LED/风扇]    [麦克风]
```

## 快速开始

### 安装

```bash
# 最小安装
pip install -e .

# 完整安装（含音频和 ASR）
pip install -e ".[all]"
```

### 使用示例

```python
from voice_router_lite import VoiceRouterPipeline

# 创建管道
pipeline = VoiceRouterPipeline()
pipeline.initialize()

# 文本指令（不依赖音频硬件）
result = pipeline.process_utterance(text="打开风扇")
print(f"意图: {result['intent']}, 槽位: {result['slots']}")

# 音频指令（需要麦克风）
# pipeline.start_listening()
# time.sleep(10)
# pipeline.stop_listening()

pipeline.close()
```

更多示例见 `examples/demo.py`。

## 模块 API

### VoiceRouterPipeline

```python
pipeline = VoiceRouterPipeline(config=None)

# 生命周期
pipeline.initialize()                    # 初始化所有组件
pipeline.start()                          # 启动音频采集
pipeline.stop()                           # 停止
pipeline.close()                          # 释放资源

# 指令处理
result = pipeline.process_utterance(
    audio_data=None,  # int16 numpy array
    text=None,        # 直接提供文本（跳过 ASR）
)

# 持续监听
pipeline.start_listening()                # 后台线程持续监听
pipeline.stop_listening()

# 回调注册
pipeline.on_wake(lambda result: print(f"唤醒: {result.keyword}"))
pipeline.on_result(lambda data: print(f"结果: {data}"))
pipeline.on_error(lambda msg: print(f"错误: {msg}"))
```

### 独立模块

```python
from voice_router_lite.audio import AudioCapture, AudioPreprocessor
from voice_router_lite.asr import ASREngine
from voice_router_lite.nlu import NLUEngine
from voice_router_lite.kws import KWSEngine
from voice_router_lite.tts import TTSEngine
from voice_router_lite.device import DeviceManager, FanDriver, LEDDriver

# 音频采集
cap = AudioCapture(config.audio)
cap.open()
chunk = cap.read()  # numpy array shape=(480,), int16

# 降噪
preproc = AudioPreprocessor(config.audio)
clean = preproc.process(chunk)

# ASR 识别
asr = ASREngine(config.models, config.audio)
asr.initialize()
result = asr.transcribe(audio_data)

# NLU 理解
nlu = NLUEngine(config.models)
nlu.initialize()
result = nlu.understand("打开风扇")

# 设备控制
mgr = DeviceManager()
mgr.register_device("风扇", FanDriver())
mgr.execute_command("set_device_state", {"device_type": "fan", "state": "on"})
```

## 项目结构

```
voice-router-agent-offline/
├── voice_router_lite/          # 核心引擎
│   ├── __init__.py             # 公开 API
│   ├── config.py               # 配置管理
│   ├── pipeline.py             # 管道编排器
│   ├── audio/                  # 音频处理
│   │   ├── capture.py          # 音频采集 (I2S/USB)
│   │   ├── player.py           # 音频播放 (I2S DAC)
│   │   ├── vad.py              # 语音活动检测
│   │   └── denoise.py          # 降噪/回声消除/AGC
│   ├── kws/                    # 唤醒词检测
│   │   └── engine.py           # sherpa-onnx KWS 封装
│   ├── asr/                    # 语音识别
│   │   └── engine.py           # sherpa-onnx ASR 封装
│   ├── nlu/                    # 指令理解
│   │   ├── engine.py           # NLU 引擎封装
│   │   └── model.py            # CNN+LSTM 模型定义
│   ├── tts/                    # 语音合成
│   │   └── engine.py           # 预录制音频拼接 TTS
│   └── device/                 # 设备控制
│       └── manager.py          # 设备管理器 + 驱动
├── tests/                      # 测试
│   └── test_pipeline.py
├── examples/                   # 示例
│   └── demo.py
├── models/                     # 模型文件 (gitignored)
├── audio_clips/                # TTS 音频片段 (gitignored)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 运行测试

```bash
pytest tests/ -v
```

## 技术选型

| 组件 | 方案 | 模型大小 | 选型理由 |
|------|------|---------|---------|
| ASR | sherpa-onnx Zipformer zh 14M | 25MB | 128MB 路由器可承载 |
| NLU | CNN+LSTM | 11MB (fp32) | 准确率与体积平衡 |
| KWS | sherpa-onnx KWS | 3.3MB | 支持自定义唤醒词微调 |
| TTS | 预录制拼接 | 0MB | 零运行时内存 |
| 降噪 | 频谱门 | 0MB | CPU 轻量，延迟 < 5ms |

## 部署到路由器

```bash
# 1. 下载/训练模型（含 NLU INT8）
bash download_models.sh
python3 -m voice_router_lite.nlu.train_nlu

# 2. 打包（见 deploy/openwrt/README.md）
tar czf voice_router.tar.gz voice_router_lite/ models/ audio_clips/ deploy/

# 3. 路由器上运行守护进程（ALSA 直连，无 WebSocket）
python3 -m voice_router_lite router --config deploy/openwrt/voice-router.json
```

预研阶段仍用 `python3 -m voice_router_lite web` + ESP32。

## License

MIT
