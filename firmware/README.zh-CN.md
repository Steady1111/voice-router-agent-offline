# ESP32-S3 固件

> 中文 · [English](README.md)

「按住说话」/ 持续采音边缘节点。把 16kHz / 16 位 / 单声道 PCM 通过 WebSocket
发往中枢服务，并执行风扇等指令。

接线、烧录与 FAQ 见 **[docs/04-开发与硬件手册.md](../docs/04-开发与硬件手册.md)**。

## 硬件

- ESP32-S3-DevKitC-1（任意带 WiFi 的型号）
- INMP441 I2S MEMS 麦克风
- 一个自复位按钮（或直接用开发板上的 BOOT 键）
- 可选：一个 0.5W 喇叭 + I2S DAC 用于播放 TTS（本固件未接线——中枢仍在浏览器里播放 TTS）

默认引脚映射：

| 信号 | ESP32-S3 引脚 |
|------|--------------|
| I2S BCLK | GPIO 4 |
| I2S LRCLK | GPIO 5 |
| I2S DIN | GPIO 6 |
| 按钮 | GPIO 0（BOOT） |
| 状态灯 | GPIO 48（板载 RGB） |

在 `include/config.h`（从 `config.h.example` 复制而来）中修改它们。

## 构建

```bash
pip install platformio
cp include/config.h.example include/config.h
# 编辑 config.h，填入你的 WiFi 与中枢的 host:port（config.h 不入库）
pio run -t upload
pio device monitor
```

串口调试脚本见 `scripts/dev/`。

## 通信协议

与中枢的 `/ws/audio` 端点一致：

```
客户端 → 服务端：
  {"event": "start", "sample_rate": 16000, "format": "pcm_s16le"}
  <二进制 PCM>
  {"event": "stop"}
服务端 → 客户端：
  {"event": "transcript", "text": "..."}
  {"event": "reply", "text": "...", "actions": [...]}
  ...
```

本固件忽略服务端回传的 `tts_audio` 数据——目前播放发生在浏览器看板里。等你接上喇叭后，
把这些二进制帧接入 I2S DAC 的播放流即可。
