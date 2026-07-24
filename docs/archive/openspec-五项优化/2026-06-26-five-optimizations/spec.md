# Spec — 五项优化

> **验收参考**（非实现依据）。任务范围与完成标准以 [tasks.md](./tasks.md) 为准；API/FR 与 design 冲突时以 design 为准。

## 1. 功能需求

### FR-1 语音识别（预研）

| ID | 需求 |
|----|------|
| FR-1.1 | 唤醒后采集指令音频，最长 8s，静音 2s 结束 |
| FR-1.2 | ASR 结果经 repair 后进入 NLU；响应含 `asr_text`、`command`、`corrected` |
| FR-1.3 | `voice_wake=True` 时允许风扇状态机兜底（边界见 design.md §2.2） |
| FR-1.4 | `voice_wake=False`（网页输入）禁止风扇状态机兜底 |
| FR-1.5 | 设置 `VOICE_ROUTER_SAVE_ASR_WAV=1` 时落盘 WAV + JSON 元数据（**优先于 TTS 录音**） |
| FR-1.6 | 唤醒双线：A 线能量 fallback 可开；B 线 KWS 每帧运行；可区分唤醒来源 |

### FR-1b 唤醒双线（Q6）

| ID | 需求 |
|----|------|
| FR-1b.1 | 预研默认 `ENERGY_WAKE=1`，KWS 未命中时允许能量唤醒 |
| FR-1b.2 | B 线验收时可用 `ENERGY_WAKE=0` 单独测 KWS |
| FR-1b.3 | 量产 router 模式仅 KWS，关闭能量唤醒 |
| FR-1b.4 | `kws_samples/` 存唤醒词正/负样本，供命中率统计（v1 不重训模型） |

### FR-2 语音识别（量产）

| ID | 需求 |
|----|------|
| FR-2.1 | 仅 KWS「小T小T」唤醒，关闭能量唤醒 |
| FR-2.2 | NLU `router_*` 且 confidence < 0.5 → 不执行 |
| FR-2.3 | ASR 空文本 → 不猜测，播报「没听清，请再说一次」 |
| FR-2.4 | ASR 在子进程中运行，识别结束 3s 内子进程退出 |

### FR-3 设备状态

| ID | 需求 |
|----|------|
| FR-3.1 | 用户视图仅展示「路由器」「风扇」两块状态卡 |
| FR-3.2 | `?debug=1` 显示性能监控 7 卡 |
| FR-3.3 | ESP32 断线时风扇卡显示「离线」，不展示过时 RPM |
| FR-3.4 | `GET /api/devices` 默认仅返回 fan + router |

### FR-4 语音播报

| ID | 需求 |
|----|------|
| FR-4.1 | 执行成功/失败/没听清均有语音或蜂鸣反馈 |
| FR-4.2 | 播报文本为 `reply`，与 `command` 展示分离 |
| FR-4.3 | **v1**：网页 speechSynthesis 女声或蜂鸣即可 |
| FR-4.4 | **量产可选**：`audio_clips` 本人女声 WAV（不训练模型） |
| FR-4.5 | 有完整句 WAV 时不播 error_beep |
| FR-4.6 | 网页 speechSynthesis 使用女声（无素材时兜底） |

### FR-5 前端

| ID | 需求 |
|----|------|
| FR-5.1 | 对话区展示纠正后 `command`；可选展示 `raw_asr` 小字 |
| FR-5.2 | 顶栏显示路由器/风扇/唤醒三状态点 |
| FR-5.3 | 宽度 <900px 时单栏布局 |
| FR-5.4 | 无登录页（Q4） |

### FR-6 内存

| ID | 需求 |
|----|------|
| FR-6.1 | 待机 RSS ≤ 75 MB |
| FR-6.2 | 唤醒峰值 RSS ≤ 88 MB |
| FR-6.3 | ASR 退出后 3s RSS ≤ 76 MB |
| FR-6.4 | 量产启用 `prefer_int8_nlu`（INT8 通过精度门控） |

---

## 2. API 规格

### GET `/api/router-status`

**响应 200：**

```json
{
  "wan_up": true,
  "lan_clients": 3,
  "voice_daemon": "running",
  "ram_used_mb": 72,
  "ram_total_mb": 128,
  "cpu_pct": 12.5
}
```

| 字段 | 类型 | 预研 | 量产 |
|------|------|------|------|
| `wan_up` | bool | mock true | ubus |
| `lan_clients` | int \| null | 0 或 mock | DHCP 计数 |
| `voice_daemon` | string | `web` / `running` / `stopped` | `running` / `stopped` |
| `ram_used_mb` | float | monitor | /proc |
| `ram_total_mb` | int | 128 | 实际 |
| `cpu_pct` | float \| null | monitor | 可选 |

### GET `/api/esp32-status`（扩展）

在现有字段上增加：

```json
{
  "connected": true,
  "sessions": [{"host": "192.168.1.50"}],
  "fan": {"on": true, "level": 2},
  "last_ack_at": "2026-06-26T10:00:00Z",
  "last_command": "关闭风扇"
}
```

### WebSocket 消息（浏览器 `?role=browser`）

| type | 字段 | 说明 |
|------|------|------|
| `transcript` | `text`, `raw_asr`, `corrected`, `command` | NLU 后推送 |
| `reply` | `text` | TTS 播报句 |
| `wake` | `keyword` | 唤醒通知 |
| `device_state` | `fan_on`, `level` | ESP32 ACK |

---

## 3. TTS 素材清单（量产可选，v1 可跳过）

> `audio_clips` **仅用于播放**，不参与模型训练。v1 预研用网页女声 TTS 即可。

### feedback/

| key | 文本 |
|-----|------|
| wake | 我在 |
| unclear | 没听清，请再说一次 |
| unknown | 无法理解指令 |
| ok | 好的 |

### devices/

| key | 文本 |
|-----|------|
| fan_on | 已打开风扇 |
| fan_off | 已关闭风扇 |
| fan_level | 风扇已调到{level}档 |

### router/（高频，可 Phase 2 补录）

| key | 文本 |
|-----|------|
| wifi_on | 已打开 WiFi |
| wifi_off | 已关闭 WiFi |
| guest_on | 已打开访客网络 |
| guest_off | 已关闭访客网络 |

**录制规范：** 16kHz、mono、16-bit PCM WAV；同一人；无背景音乐；单句 <3s。

---

## 3b. 指令采集（`asr_debug/`，优先）

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `VOICE_ROUTER_SAVE_ASR_WAV` | 关 | 演示时开 `1` |
| `VOICE_ROUTER_ASR_DEBUG_DIR` | `./asr_debug/` | 输出目录 |

元数据字段：`timestamp`, `raw_asr`, `repaired`, `command`, `intent`, `success`, `wake_source`（`kws`/`energy_wake`）, `peak`, `duration_sec`

用途：NLU 误听文本增广 + 规则表更新。**不重训 ASR。**

---

## 3c. KWS 样本（`kws_samples/`，B 线）

| 子目录 | 内容 |
|--------|------|
| `positive/` | 「小T小T」正样本（好麦录制） |
| `negative/` | 环境噪声、相似发音、误触 |

用途：命中率/误唤醒率统计；调 `keywords.txt` 阈值。**v1 不重训 sherpa KWS 模型。**

---

## 4. 非功能需求

| ID | 需求 |
|----|------|
| NFR-1 | 局域网 HTTP，无 TLS/登录（v1） |
| NFR-2 | ASR 样本仅本地存储，默认不上传 |
| NFR-3 | Web 与 daemon 可独立启停（Q2） |
| NFR-4 | 固件 WebSocket 协议向后兼容 |

---

## 5. 验收标准

### AC-1 风扇语音（ESP32 预研）

| 用例 | 通过线 |
|------|--------|
| 「打开风扇」×10 | ≥8 成功 |
| 「关闭风扇」×10 | ≥8 成功 |
| 风扇运转中关扇 ×5 | ≥3 成功 |
| 无关胡话 ×10 | 0 次误执行 router 危险操作 |

### AC-2 网页

| 用例 | 通过线 |
|------|--------|
| 默认页 | 仅两设备卡，无 LED/relay |
| `?debug=1` | 监控 7 卡可见 |
| 手机同 WiFi | 可访问、布局不崩 |

### AC-3 语音反馈（v1 放宽）

| 用例 | 通过线 |
|------|--------|
| 开扇成功 | 听到「已打开风扇」类反馈（**女声 TTS 或蜂鸣均可**） |
| 没听清 | 有反馈，无男声默认音 |
| 量产加测 | 有 `audio_clips` 时播 WAV（Phase 2 可选） |

### AC-3b 唤醒双线

| 用例 | 通过线 |
|------|--------|
| A 线 ESP32 + ENERGY_WAKE=1 | 能唤醒并进入指令采集 |
| B 线 Mac 麦 + ENERGY_WAKE=0 | 「小T小T」×20，≥14 唤醒 |
| B 线负样本 ×10 | ≤2 误唤醒 |
| 量产 router | 能量唤醒关闭，仅 KWS |

### AC-4 内存（QEMU MIPS 或实机）

| 阶段 | 通过线 |
|------|--------|
| 待机 | ≤75 MB |
| 唤醒峰值 | ≤88 MB |
| ASR 后 | ≤76 MB |

### AC-5 进程

| 用例 | 通过线 |
|------|--------|
| 停 daemon 留 web | 网页可打开，router-status 显示 daemon stopped |
| 启 daemon | voice_daemon → running |

---

## 6. 测试命令

```bash
# 单元测试
python3 -m pytest tests/ -q

# 预研服务
VOICE_ROUTER_ENERGY_WAKE=1 python3 -m voice_router_lite web

# ASR 指令采集（演示时开着）
VOICE_ROUTER_SAVE_ASR_WAV=1 VOICE_ROUTER_ASR_DEBUG_DIR=./asr_debug \
  VOICE_ROUTER_ENERGY_WAKE=1 python3 -m voice_router_lite web

# KWS B 线单独测（好麦，关能量唤醒）
VOICE_ROUTER_ENERGY_WAKE=0 python3 -m voice_router_lite web
```
