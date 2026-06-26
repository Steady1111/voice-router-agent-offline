# 从 voice-router-agent 迁移可复用模块方案

> 目的：评估从同级目录 `../voice-router-agent` 迁移哪些内容到当前瘦设备离线项目 `voice-router-agent-offline`。
>
> 当前项目定位是 128MB MIPS 路由器/瘦设备离线语音控制方案，因此迁移必须避开 Whisper、LLM、Piper、云服务和重依赖链路，只保留前端看板、ESP32 采音/执行端、轻量 API 壳和文档资产。

## 1. 迁移原则

1. 保留当前项目的核心链路：

   ```text
   音频采集 -> VAD/按键触发 -> 小 ASR -> 规则/小模型 NLU -> 设备命令 -> 预录制反馈
   ```

2. 不引入大模型链路：

   ```text
   Whisper -> Qwen/Ollama/LLM function calling -> Piper/Edge TTS
   ```

3. 迁移内容以“可观察、可调试、可连接 ESP32”为主：
   - Web 看板
   - 文本调试入口
   - 设备状态展示
   - ESP32 采音和风扇执行固件
   - 轻量 FastAPI 静态服务和 API 壳

4. 后端逻辑统一接入当前项目已有模块：
   - `voice_router_lite.pipeline.VoiceRouterPipeline`
   - `voice_router_lite.nlu`
   - `voice_router_lite.device`
   - `voice_router_lite.audio`
   - `voice_router_lite.tts`

## 2. 源项目可迁移内容总览

源项目路径：

```text
/Users/duanduan/CodingFiles/voice-router-agent
```

目标项目路径：

```text
/Users/duanduan/CodingFiles/voice-router-agent-offline
```

### 2.1 推荐直接迁移

| 源路径 | 建议目标路径 | 说明 |
|---|---|---|
| `server/web/static/index.html` | `web/static/index.html` | Web 看板页面，可直接作为第一版 UI |
| `server/web/static/style.css` | `web/static/style.css` | 看板样式，可直接迁移 |
| `server/web/static/app.js` | `web/static/app.js` | 前端交互可迁移，但需要改少量接口语义 |
| `firmware/platformio.ini` | `firmware/platformio.ini` | ESP32-S3 PlatformIO 配置 |
| `firmware/include/config.h.example` | `firmware/include/config.h.example` | WiFi、WebSocket、引脚配置模板 |
| `firmware/src/main.cpp` | `firmware/src/main.cpp` | ESP32 采音、风扇控制、显示逻辑 |
| `firmware/README.zh-CN.md` | `firmware/README.zh-CN.md` | 固件说明文档 |
| `docs/hardware-shopping-list.zh-CN.md` | `docs/hardware-shopping-list.zh-CN.md` | 硬件采购清单 |
| `docs/testing-plan.zh-CN.md` | `docs/testing-plan.zh-CN.md` | 测试计划 |

### 2.2 可迁移但必须瘦身改造

| 源路径 | 建议目标路径 | 改造点 |
|---|---|---|
| `server/web/routes.py` | `web/routes.py` 或 `voice_router_lite/web/routes.py` | 去掉 `orchestrator`、LLM、重 TTS 依赖，改接 `VoiceRouterPipeline` |
| `server/main.py` | `server.py` 或 `voice_router_lite/web/app.py` | 只保留 FastAPI、静态页面挂载、轻量 API 初始化 |
| `server/adapters/device_mock.py` | 不建议原样搬，可参考 | 设备状态格式可参考，但应和当前 `DeviceManager` 打通 |
| `server/transport/ws_audio.py` | `voice_router_lite/web/ws_audio.py` | WebSocket 协议可参考，处理链改成 `ASREngine + NLUEngine + DeviceManager + TTSEngine` |
| `server/config.py` | 不建议原样搬，可参考 | 原配置包含 LLM/云服务字段，当前项目已有 `voice_router_lite/config.py` |

### 2.3 不建议迁移

以下文件属于大模型/重服务端路线，不适合当前瘦设备方案：

| 源路径 | 原因 |
|---|---|
| `server/adapters/asr_whisper.py` | Whisper 模型过大，不适合 128MB 路由器 |
| `server/adapters/llm_ollama.py` | Ollama/Qwen 需要 GB 级内存 |
| `server/adapters/llm_xfyun.py` | 云端 LLM，不符合断网自愈和离线目标 |
| `server/adapters/tts_piper.py` | Piper 运行时内存偏重 |
| `server/adapters/tts_edge.py` | 依赖联网 |
| `server/adapters/asr_aliyun.py` | 依赖云 ASR |
| `server/intent/orchestrator.py` | 基于 LLM tool call，不适合当前项目 |
| `server/intent/tools.py` | 与 LLM function calling 强绑定 |
| `pyproject.toml` | 依赖包含 `openai`、`edge-tts`、`httpx`、`paramiko`、`sounddevice` 等，不应整份替换 |
| `.env.example` | 原变量围绕 Whisper/LLM/TTS/云服务设计，不适合直接使用 |

## 3. 前端页面迁移方案

### 3.1 页面能力

源项目 `server/web/static/` 的看板已有这些能力：

- 顶部网络状态
- 唤醒状态
- 对话记录
- 按住说话按钮
- 浏览器麦克风采集 PCM
- 文本调试输入框
- 设备状态列表
- TTS 音频播放

这些能力对当前 offline 项目仍然有价值。

### 3.2 需要适配的接口

源项目前端目前调用：

```text
GET  /api/devices
POST /api/text
GET  /api/network
WS   /ws/audio
```

当前项目建议保留同名接口，降低前端改造量。

建议接口行为如下：

#### GET /api/devices

返回当前 `DeviceManager` 中的设备状态，格式兼容前端：

```json
[
  {
    "id": "fan",
    "name": "风扇",
    "type": "fan",
    "state": "off",
    "level": 0
  },
  {
    "id": "led",
    "name": "LED",
    "type": "led",
    "state": "off",
    "level": 0
  }
]
```

#### POST /api/text

请求：

```json
{
  "text": "打开风扇"
}
```

处理：

```python
pipeline.process_utterance(text=text)
```

返回：

```json
{
  "transcript": "打开风扇",
  "reply": "已打开风扇",
  "intent": "set_device_state",
  "slots": {
    "device_type": "fan",
    "state": "on"
  },
  "success": true
}
```

#### GET /api/network

瘦设备第一版可以先返回本机/路由器状态快照，不强制搬原项目 `NetworkMonitor`：

```json
{
  "online": true,
  "consecutive_failures": 0,
  "last_target_ok": "local",
  "in_recovery": false
}
```

后续如果要恢复断网自愈，再单独实现轻量 ping 监控。

#### WS /ws/audio

前端已支持浏览器麦克风采集并发送 PCM：

```text
{"event":"start","sample_rate":16000,"format":"pcm_s16le"}
<binary pcm chunks>
{"event":"stop"}
```

目标项目服务端收到后应：

1. 拼接 PCM。
2. 转成 `numpy.int16`。
3. 调用当前项目 `ASREngine.transcribe()`。
4. 调用 `pipeline.process_utterance(text=asr_text)` 或拆开调用 `NLUEngine + DeviceManager`。
5. 返回 transcript/reply/done。

如果暂时没有 ASR 模型，可以先只实现 `/api/text`，`/ws/audio` 标记为后续阶段。

### 3.3 前端需要小改的点

`app.js` 当前假定服务端会返回 MP3 TTS 二进制。当前项目的 TTS 是预录制 WAV/蜂鸣，不一定适合通过浏览器播放。

建议第一版：

- 保留 TTS 播放逻辑，但允许服务端不返回 `tts_audio`。
- 文本调试和设备列表先跑通。
- 设备图标映射补充当前项目设备：

```js
const DEVICE_ICONS = {
  fan: "🌀",
  led: "💡",
  relay: "🔌",
  light: "💡",
  ac: "❄️",
};
```

页面标题建议从 `voice-router-agent` 改为：

```text
voice-router-lite
```

或：

```text
离线语音路由器
```

## 4. 后端轻量 Web 服务方案

当前 offline 项目没有 Web 服务。建议新增一层非常薄的 Web 包，而不是搬原项目完整 `server/`。

建议目录：

```text
voice-router-agent-offline/
├── web/
│   ├── __init__.py
│   ├── app.py
│   ├── routes.py
│   ├── ws_audio.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
```

也可以放到包内：

```text
voice_router_lite/
├── web/
│   ├── __init__.py
│   ├── app.py
│   ├── routes.py
│   ├── ws_audio.py
│   └── static/
```

推荐放到 `voice_router_lite/web/`，方便随 Python 包一起安装。

### 4.1 依赖建议

只新增：

```toml
web = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
]
```

不要引入：

```text
openai
edge-tts
httpx
paramiko
sounddevice
pydantic-settings
```

除非后续明确需要。

### 4.2 app.py 职责

只做：

- 创建 `FastAPI`
- 初始化 `PipelineConfig`
- 初始化 `VoiceRouterPipeline`
- 挂载 routes
- 挂载 static

不要做：

- LLM 初始化
- Whisper 初始化
- Piper 初始化
- 云服务适配器初始化

### 4.3 routes.py 职责

只提供：

```text
GET  /health
GET  /api/devices
POST /api/text
GET  /api/network
WS   /ws/audio
```

其中 `/ws/audio` 可以在第一阶段先返回“暂未启用 ASR”错误，等 ASR 模型和音频路径确认后再接。

## 5. ESP32 固件迁移方案

源项目固件路径：

```text
../voice-router-agent/firmware
```

建议整体迁移到：

```text
firmware/
```

### 5.1 可保留能力

固件里的这些能力和大模型无关，适合保留：

- ESP32-S3 WiFi 连接
- WebSocket 长连接
- INMP441 I2S 采集 16kHz PCM
- 向服务端发送 `start` / PCM bytes / `stop`
- 接收 `start_record` / `stop_record`
- 接收风扇控制 JSON：

  ```json
  {
    "event": "fan",
    "action": "on",
    "speed": 178,
    "speed_level": 3
  }
  ```

- TB6612 风扇 PWM 控制
- OLED/数码管状态显示

### 5.2 需要改造的点

1. WebSocket 服务地址改为当前 offline 项目的 Web 服务地址。
2. 固件 README 里删除 Whisper/Qwen/Piper 描述。
3. 如果当前 offline 方案改成“按键触发录音”，固件需要加一个按键输入，或保留服务端下发 `start_record`。
4. 如果路由器最终直接接 USB/I2S 麦克风，ESP32 固件可以只作为预研/硬件验证模块，不进入最终部署。

## 6. 文档迁移方案

建议搬：

```text
docs/hardware-shopping-list.zh-CN.md
docs/testing-plan.zh-CN.md
firmware/README.zh-CN.md
```

可参考但不一定搬：

```text
docs/architecture.zh-CN.md
docs/开发日志.md
docs/plan_风扇排查.md
```

迁移后需要统一术语：

| 原项目术语 | 当前项目建议 |
|---|---|
| voice-router-agent | voice-router-lite / voice-router-agent-offline |
| Whisper | sherpa-onnx Zipformer small |
| Qwen/Ollama/LLM | NLU 规则/小模型 |
| Piper/Edge TTS | 预录制音频拼接 |
| LLM tool call | 固定意图 + 槽位 + DeviceManager |

## 7. 分阶段实施建议

### 阶段 A：只搬前端看板和文本调试

目标：不接音频、不接 ESP32，先能在浏览器输入“打开风扇”并看到状态变化。

任务：

1. 新增 `voice_router_lite/web/static/`。
2. 复制 `index.html`、`style.css`、`app.js`。
3. 新增轻量 FastAPI app。
4. 实现：

   ```text
   GET  /health
   GET  /api/devices
   POST /api/text
   GET  /api/network
   ```

5. `POST /api/text` 调用当前 `VoiceRouterPipeline.process_utterance(text=...)`。

验收：

- 浏览器打开首页。
- 输入“打开风扇”。
- 对话框显示回复。
- 设备状态从关变开。

### 阶段 B：接入浏览器麦克风音频

目标：页面按住说话，浏览器发送 PCM，服务端用当前 ASR/NLU 处理。

任务：

1. 新增 `ws_audio.py`。
2. 实现 `/ws/audio`。
3. PCM bytes 转 `numpy.int16`。
4. 调用 `ASREngine.transcribe()`。
5. 发送：

   ```json
   {"event": "transcript", "text": "..."}
   {"event": "reply", "text": "..."}
   {"event": "done"}
   ```

验收：

- 浏览器按住说话能显示识别文本。
- 指令能更新设备状态。

### 阶段 C：迁移 ESP32 固件

目标：ESP32 采音和风扇控制跑通。

任务：

1. 复制 `firmware/`。
2. 修改 `config.h.example` 和 README。
3. 服务端支持 ESP32 WebSocket 会话。
4. 设备状态变化时广播风扇指令。

验收：

- ESP32 能连上 offline 服务。
- 服务端能收到 PCM。
- “打开风扇”后 ESP32 收到 fan 指令。
- 风扇 PWM 输出变化。

### 阶段 D：瘦设备最终裁剪

目标：为路由器部署裁剪不必要内容。

任务：

1. 确认是否最终还需要 Web 看板。
2. 确认是否需要 ESP32，还是路由器本机接麦克风/继电器。
3. 关闭 KWS 或改按键触发。
4. 保留小 ASR、NLU、DeviceManager、预录制反馈。
5. 清理开发期 Web/调试依赖。

验收：

- 内存预算仍符合 `README.md` 的瘦设备目标。
- 不出现 Whisper、Qwen、Piper、Edge TTS、云服务依赖。

## 8. 风险与注意事项

1. 原前端不是独立前端工程，没有 React/Vue/Vite，只有静态 HTML/CSS/JS。
2. 原 `server/main.py` 和 `server/web/routes.py` 依赖大模型编排器，不能整搬。
3. 原 `app.js` 可以保留 UI，但服务端消息格式要兼容。
4. 固件依赖具体硬件引脚，迁移后必须确认：
   - INMP441 引脚
   - TB6612 引脚
   - OLED/数码管引脚
   - ESP32-S3 开发板型号
5. `pyproject.toml` 不要从源项目覆盖当前项目，否则会引入大量不适合瘦设备的依赖。
6. 如果最终部署到 128MB MIPS 路由器，FastAPI/uvicorn 是否保留要重新评估；Web 看板更适合开发调试阶段。

## 9. 建议最终迁移清单

第一批建议执行：

```text
../voice-router-agent/server/web/static/index.html
../voice-router-agent/server/web/static/style.css
../voice-router-agent/server/web/static/app.js
```

第二批建议执行：

```text
新增 voice_router_lite/web/app.py
新增 voice_router_lite/web/routes.py
新增 voice_router_lite/web/ws_audio.py
```

第三批建议执行：

```text
../voice-router-agent/firmware/platformio.ini
../voice-router-agent/firmware/include/config.h.example
../voice-router-agent/firmware/src/main.cpp
../voice-router-agent/firmware/README.zh-CN.md
```

文档按需执行：

```text
../voice-router-agent/docs/hardware-shopping-list.zh-CN.md
../voice-router-agent/docs/testing-plan.zh-CN.md
../voice-router-agent/docs/architecture.zh-CN.md
```

明确不执行：

```text
../voice-router-agent/server/adapters/asr_whisper.py
../voice-router-agent/server/adapters/llm_ollama.py
../voice-router-agent/server/adapters/llm_xfyun.py
../voice-router-agent/server/adapters/tts_piper.py
../voice-router-agent/server/adapters/tts_edge.py
../voice-router-agent/server/intent/orchestrator.py
../voice-router-agent/server/intent/tools.py
../voice-router-agent/pyproject.toml
../voice-router-agent/.env.example
```

## 10. 推荐决策

建议先执行阶段 A。

原因：

- 改动最小。
- 不碰 ASR 模型和硬件。
- 能快速验证当前 offline 项目的 NLU、设备状态、文本调试体验。
- 后续是否继续接浏览器麦克风和 ESP32，可以根据阶段 A 的效果再决定。

