<!--
  BASELINE: LOCKED v1.0 | 2026-06-26-five-optimizations | 2026-06-26
  实现唯一依据之一。变更须同步 design.md 并更新 BASELINE.md 修订历史。
-->

# Tasks — 五项优化

> **状态：LOCKED（基线 v1.0）**  
> 与 [design.md](./design.md) 共同构成后续实现的**唯一执行依据**。  
> 每条任务须有 ID（如 `1A.3`）；完成以「完成标准」列为准。

> 执行顺序：Phase 1 → Phase 2 → Phase 3  
> **2026-06-26 修订：** 1A 指令采集优先；1B TTS 降为可选；新增 1E KWS B 线。

---

## Phase 1（可并行，约 1 周）

### 1A 指令样本采集（**优先**）

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 1A.1 | 新增采集 CLI | `tools/collect_asr_samples.py` | 读取 `asr_debug/` 生成 CSV 报告 |
| 1A.2 | 文档化环境变量 | `spec.md`（已有）、`离线语音方案-路由器部署.md` 补一句 | `SAVE_ASR_WAV` 用法可查 |
| 1A.3 | 采集窗配置 | `voice_router_lite/web/ws_audio.py` | `VOICE_ROUTER_SKIP_SEC` 等可调 |
| 1A.4 | 演示时自动采样本 | `asr_debug/*.wav` | **用着采** ≥20 条开/关扇，带 JSON 元数据 |
| 1A.5 | 记录唤醒来源 | `ws_audio.py` 落盘元数据 | `wake_source`: `kws` / `energy_wake` |

**验收：** 采集工具输出成功率表；≥20 条样本。**不需要坐着录 TTS。**

---

### 1B TTS 女声 WAV（**可选延后**，量产前）

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 1B.1 | 录制 feedback 4 句 | `audio_clips/feedback/*.wav` | 可选 |
| 1B.2 | 录制 devices 3 句 | `audio_clips/devices/*.wav` | 可选 |
| 1B.3 | 更新索引 | `audio_clips/index.json` | 有录音时更新 |
| 1B.4 | 验证播放 | `voice_router_lite/tts/engine.py` | 有 WAV 时可播 |

**v1 跳过条件：** 网页女声 TTS + 蜂鸣已满足 AC-3。

---

### 1C INT8 NLU

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 1C.1 | 导出 INT8 | `tools/export_nlu_int8.py`（已有则跑） | `models/intent_model.int8.onnx` 存在 |
| 1C.2 | 精度回归 | `tests/test_nlu.py` | 意图准确率 ≥99% |
| 1C.3 | 路由器配置 | `deploy/openwrt/voice-router.json` | `prefer_int8_nlu: true` |

**验收：** pytest 绿；加载 INT8 无回退告警。

---

### 1D router-status API

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 1D.1 | 新增路由 | `voice_router_lite/web/app.py` | `GET /api/router-status` 200 |
| 1D.2 | 实现逻辑 | `voice_router_lite/web/router_status.py`（新） | 预研 mock + monitor 数据 |
| 1D.3 | 扩展 esp32-status | `voice_router_lite/web/app.py` 或 bridge | `last_command`, `last_ack_at` |
| 1D.4 | 收缩 devices | `voice_router_lite/devices/manager.py` | 默认仅 fan+router |

**验收：** curl 两个 API 字段符合 spec.md。

---

### 1E KWS B 线（与 A 线并行）

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 1E.1 | KWS 命中率脚本 | `tools/eval_kws_samples.py`（新） | 对 `kws_samples/` 输出命中率 |
| 1E.2 | 样本目录约定 | `kws_samples/positive/`, `negative/` | README 说明录制规范 |
| 1E.3 | 调 keywords 阈值 | `models/kws/keywords.txt` | Mac 麦 ≥14/20 命中 |
| 1E.4 | UI 显示唤醒来源 | `app.js` + WS 事件 | debug 可见 kws / energy_wake |
| 1E.5 | B 线验收文档 | `docs/openspec/.../spec.md` AC-3b | 测法写清 |

**说明：** A 线继续 `ENERGY_WAKE=1` 演示；B 线用好麦 `ENERGY_WAKE=0` 单独测。**不重训 KWS 模型。**

---

## Phase 2（依赖 Phase 1，约 1 周）

### 2A NLU 再训

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 2A.1 | 从 asr_debug 导误识别 | `tools/build_misheard_training.py`（新） | 生成训练增广 JSON |
| 2A.2 | 重训 + 导出 | `voice_router_lite/nlu/train_nlu.py` | 新 ONNX |
| 2A.3 | 回归测试 | `tests/test_nlu.py` | 含新样本用例 |

**依赖：** 1A 样本 ≥50 条。

---

### 2B 前端用户视图 + debug

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 2B.1 | 布局改版 | `index.html`, `style.css` | 双栏 + 顶栏状态点 |
| 2B.2 | debug 开关 | `app.js` | URL `?debug=1` 控制监控 |
| 2B.3 | 路由器/风扇卡 | `app.js` | 轮询 router-status + WS |
| 2B.4 | 隐藏 Mock 设备 | `app.js` | 默认列表无 led/relay |
| 2B.5 | 女声 TTS 兜底 | `app.js` | `pickFemaleVoice()` 排除男声 |
| 2B.6 | 响应式 | `style.css` | <900px 单栏 |

**验收：** AC-2 网页用例。

---

### 2C 路由器 DAC 播放（量产）

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 2C.1 | daemon TTS 路径 | `voice_router_lite/router/daemon.py` | 执行后 `speak_sentence(reply)` |
| 2C.2 | Web 不 open ALSA | `voice_router_lite/web/app.py` | `tts.initialize(open_player=False)` |
| 2C.3 | procd 双服务 | `deploy/openwrt/voice-router-web.init`, `voice-routerd.init` | 分进程启停 |

**依赖：** 2C.1 无 WAV 时蜂鸣亦可；**1B 非阻塞**。

---

## Phase 3（上板，约 1 周）

### 3A 量产语音策略（仅 B 线 KWS）

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 3A.1 | 关闭能量唤醒 | OpenWrt env / json | router 模式 `ENERGY_WAKE=0` |
| 3A.2 | 置信度拒绝 | `voice_router_lite/web/service.py` | router_* <0.5 拒绝 |
| 3A.3 | 空 ASR 不猜测 | `voice_router_lite/router/daemon.py` | 量产路径无状态机 |
| 3A.4 | KWS B 线实机验收 | USB-I2S 麦 | AC-3b 通过 |

---

### 3B 内存与稳定性

| # | 任务 | 文件 | 完成标准 |
|---|------|------|----------|
| 3B.1 | RSS 阶段日志 | `voice_router_lite/router/daemon.py` | 启动/KWS/ASR/退出 四条 |
| 3B.2 | QEMU 72h | `docs/` 或 CI 脚本 | 泄漏 <5MB |
| 3B.3 | 更新部署 doc | `离线语音方案-路由器部署.md` | 内存实测数据 |

**验收：** AC-4。

---

### 3C 全量验收

| # | 任务 | 完成标准 |
|---|------|----------|
| 3C.1 | 跑 AC-1～AC-5 + AC-3b | 记录到 `docs/模型训练日志.md` 或验收表 |
| 3C.2 | OpenSpec archive | 移入 `docs/openspec/archive/` 并标完成日期 |

---

## 不在本 tasks 范围

- 固件引脚变更
- WebSocket 协议破坏性修改
- 外网 HTTPS / 登录
- 移除 ESP32 风扇执行（Q3 保留）
- **重训 Zipformer ASR / sherpa KWS 大模型**

---

## 建议首批执行（/opsx:apply 入口）

1. **1A** 开着 `SAVE_ASR_WAV` 演示通风扇（模型数据，零额外录音）
2. **1D + 2B** 状态面板 + 前端（展示向）
3. **1E** KWS B 线（Mac 好麦，与 A 线并行）
4. **2A** NLU 再训（等 1A 样本够）
5. **1B** TTS 录音（**量产前再做**）
6. **1C + 3B** 内存（量产前）
