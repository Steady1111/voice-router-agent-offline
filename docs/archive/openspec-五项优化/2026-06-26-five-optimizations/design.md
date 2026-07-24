<!--
  BASELINE: LOCKED v1.0 | 2026-06-26-five-optimizations | 2026-06-26
  实现唯一依据之一。变更须同步 tasks.md 并更新 BASELINE.md 修订历史。
-->

# Design — 五项优化

> **状态：LOCKED（基线 v1.0）**  
> 与 [tasks.md](./tasks.md) 共同构成后续实现的**唯一技术依据**。  
> 冲突时以本文为准（架构/边界）；任务范围以 tasks 为准。

## 1. 系统架构（优化后）

```
┌─────────────────────────────────────────────────────────────┐
│  路由器 / Mac 预研机                                          │
│                                                             │
│  ┌──────────────────┐    ┌─────────────────────────────┐  │
│  │ voice-router-web │    │ voice-routerd (daemon)      │  │
│  │ FastAPI :28080   │    │ KWS → ASR子进程 → NLU → TTS │  │
│  │ 用户视图+debug   │◄──►│ ubus / 设备控制              │  │
│  └────────┬─────────┘    └───────────┬─────────────────┘  │
│           │ WS 观察者                   │ WS 控制            │
└───────────┼─────────────────────────────┼───────────────────┘
            │                             │
            ▼                             ▼
     浏览器 / 手机 App              ESP32-S3
                                   INMP441 / 风扇 / RGB / 数码管
```

**分进程理由（Q2）：** Web 只读状态 + 对话展示；daemon 承担实时语音。进程间通过本地 HTTP/共享状态文件或轻量 IPC 同步风扇/路由状态。

---

## 2. 优化 1：语音识别

### 2.1 预研链路（不变协议）

ESP32 PCM → WebSocket → Mac `ws_audio.py` → ASR → `handle_text(voice_wake=True)` → ESP32 `fan` 事件

**增强：**

- `tools/collect_asr_samples.py`：落盘 WAV + 元数据
- 采集窗 A/B 配置表（环境变量）
- `command` 字段展示纠正后指令（已实现，文档化）
- 误识别表持续从 `asr_debug/` 补充

### 2.2 风扇兜底状态机（仅 voice_wake）

```
fan_is_on=false + 短句含风/扇 + 非路由词 → 打开风扇
fan_is_on=true  + 短句含风/扇 + 非路由词 + 无「开」→ 关闭风扇
命中 _FAN_*_MISHEARD → 按表
含路由关键词 → 仅走 NLU，不兜底
网页文本输入 → 仅 NLU，不兜底
```

### 2.3 唤醒双线研发（Q6）

代码已支持「先 KWS，未命中再能量 fallback」（`ws_audio.py`）。文档从「预研二选一」改为 **A/B 并行**：

```
┌──────────────────────────────────────────────┐
│ A 线（现设备，ESP32 烂麦）                      │
│ ENERGY_WAKE=1 → 能量超阈即唤醒 → ASR/NLU       │
│ 用途：演示通风扇、自动采 asr_debug 指令样本      │
│ 缺点：易误触，无真实唤醒词语义                    │
└──────────────────────────────────────────────┘
              并行，同一套服务，env 切换验收
┌──────────────────────────────────────────────┐
│ B 线（KWS「小T小T」，量产目标）                   │
│ 每帧跑 sherpa KWS；命中为真唤醒                   │
│ 验收环境：Mac 好麦 / 路由器 USB-I2S（非 ESP32 麦） │
│ 工作：调 keywords 阈值、录 kws_samples 做命中率表  │
│ 不重训 KWS 大模型（v1 仅调参 + 评估）             │
└──────────────────────────────────────────────┘
```

| | A 线 | B 线 |
|--|------|------|
| 开关 | `VOICE_ROUTER_ENERGY_WAKE=1` | 始终跑 KWS；测 B 时可 `ENERGY_WAKE=0` |
| 硬件 | ESP32 INMP441 | Mac / USB-I2S |
| 量产 | **关闭** | **仅此项** |

Web 顶栏/调试页可展示本次唤醒来源：`kws` vs `energy_wake`（待实现）。

### 2.4 量产链路

ALSA → `voice-routerd` → **仅 B 线 KWS** → VAD 结束 → ASR 子进程 → NLU

- `confidence < 0.5` 且 `router_*`：**拒绝执行**
- ASR 空：**不猜测**，播报「没听清」
- **关闭** `VOICE_ROUTER_ENERGY_WAKE`

### 2.5 录音三条线（Q5）

| 目录/方式 | 用途 | 训练？ | v1 |
|-----------|------|--------|-----|
| `audio_clips/` | TTS 播放「已打开风扇」 | 否 | 可选延后 |
| `asr_debug/` | `SAVE_ASR_WAV=1` 自动采指令 | 间接 → NLU 文本增广 | **优先** |
| `kws_samples/` | 人工录「小T小T」正/负样本 | 否 → KWS 评估/调阈 | B 线并行 |

**用户参与方式：** 不必坐着录 30 句 TTS；正常对着设备说开/关扇，开采集开关即可。KWS 样本仅在 B 线验收时按需补录（建议好麦）。

### 2.6 NLU 再训

从 `asr_debug/` 导出 ≥50 条 ESP32 误识别样本，加入 `train_nlu.py` 同音变体，重导 ONNX + INT8。**不重训 Zipformer ASR。**

---

## 3. 优化 2：设备状态面板

### 3.1 UI 结构

- **默认**：顶栏三状态点 + 对话区 + 右侧「路由器」「风扇」两卡
- **`?debug=1`**：展开性能监控 7 卡 + 可选 Mock 设备

### 3.2 新增 API

见 [spec.md](./spec.md) § API

### 3.3 数据流

- ESP32 `device_state` → `ESP32Bridge` → WebSocket hub → 浏览器
- `last_command` 由服务端在 `handle_text` 成功后写入共享状态

---

## 4. 优化 3：语音播报

### 4.1 v1 策略（Q1 修订）

**预研（现在）：** 网页 `speechSynthesis` 女声 + 蜂鸣；**不要求**录 `audio_clips`。

**量产（后续）：** 本人录制 `audio_clips/` 女声 WAV → I2S 喇叭；清单见 [spec.md](./spec.md) § TTS 素材。

### 4.2 播放路径

| 环境 | v1 播放 | 量产播放 |
|------|---------|----------|
| daemon | 蜂鸣 / 无 | `TTSEngine` → ALSA/I2S |
| web | speechSynthesis **女声** | 有 WAV 则播 WAV，否则女声 |
| ESP32 | 灯+数码管 | 同左 |

### 4.3 规则

- 播报 `reply`，不播报 raw ASR
- 有完整句 WAV 时不播蜂鸣
- 网页女声：排除男声系统音（见 explore 文档 §3.4）

---

## 5. 优化 4：前端布局

### 5.1 改动文件

- `voice_router_lite/web/static/index.html`
- `voice_router_lite/web/static/style.css`
- `voice_router_lite/web/static/app.js`

### 5.2 移除/合并

- 默认隐藏 Mock LED/继电器
- 唤醒大卡合并进顶栏状态点
- 监控默认折叠

### 5.3 响应式

`< 900px`：单栏 + 底部设备抽屉

---

## 6. 优化 5：内存

### 6.1 量产配置（`deploy/openwrt/voice-router.json`）

保持：`model_serial_exclusive`, `asr_subprocess`, `prefer_int8_nlu`, cgroups

### 6.2 Web 进程瘦身

- Web 进程 **不加载 ASR**（ESP32 模式已有）
- Web 进程 **不 open ALSA**（TTS 仅 daemon）

### 6.3 INT8 启用门控

`intent_model.int8.onnx` 存在且 `tests/test_nlu.py` 意图准确率 ≥99% → `prefer_int8_nlu=true`

### 6.4 度量

daemon 各阶段打 RSS 日志；`/api/monitor` 展示（debug）

---

## 7. 进程拆分设计（Q2）

| 进程 | 入口 | 职责 |
|------|------|------|
| `voice-router-web` | `python3 -m voice_router_lite web` | FastAPI、静态页、WS 观察者、文本调试 |
| `voice-routerd` | `python3 -m voice_router_lite router` | KWS、ASR 子进程、NLU、TTS、ubus |

**状态同步（v1 简单方案）：**

- 共享 JSON 状态文件 `/tmp/voice-router-state.json`（fan、last_command、daemon_alive）
- 或 Web 只读 HTTP 调 daemon 本地端口（后续）；Phase 1 先用文件 + 现有 `DeviceManager` 在 web 进程内对 ESP32 的控制

**预研阶段说明：** Mac 上仍可单进程 `web` 联调 ESP32；分进程在 OpenWrt `procd` 配置中落地（两个 init 脚本）。

---

## 8. ESP32 角色（Q3）

预研与近期展示：**ESP32 继续作为风扇执行端 + 采音端**。

量产远期可切换 USB-I2S 麦 + GPIO 风扇，但不在本变更 scope 内移除 ESP32 支持。
