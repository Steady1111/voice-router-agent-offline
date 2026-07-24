# OpenSpec：五项优化（2026-06-26）

> 变更 ID：`2026-06-26-five-optimizations`  
> 状态：**基线已锁定 — apply 进行中（2026-06-26）**  
> 工作流：`propose` ✅ → `explore` ✅ → **baseline** ✅ → `apply` 🔄 → `archive` ⬜

## 基线（实现唯一依据）

| 文档 | 角色 | 状态 |
|------|------|------|
| **[design.md](./design.md)** | 架构、边界、技术方案 | **LOCKED** |
| **[tasks.md](./tasks.md)** | 任务、文件路径、完成标准 | **LOCKED** |
| [BASELINE.md](./BASELINE.md) | 锁定记录与治理规则 | — |

其余文档（proposal / spec / explore）仅作背景与验收参考；**冲突以 design + tasks 为准**。

## 文档索引

| 文档 | 说明 | 实现依据 |
|------|------|----------|
| [proposal.md](./proposal.md) | 为什么要做、做什么、不做什么 | 参考 |
| [design.md](./design.md) | 技术方案、架构、边界、已拍板决策 | **唯一** |
| [spec.md](./spec.md) | 需求规格、API、验收标准 | 验收参考 |
| [tasks.md](./tasks.md) | 可执行任务清单（Phase 1–3） | **唯一** |

边界细节另见：[../../superpowers/specs/2026-06-26-five-optimizations-explore.md](../../superpowers/specs/2026-06-26-five-optimizations-explore.md)（参考，已随基线同步）

**录音 / 采集实操：** [../../录音采集指南.md](../../录音采集指南.md)

## 已拍板决策（2026-06-26）

| # | 决策 |
|---|------|
| Q1 | TTS 预录女声 WAV：**v1 可选延后**；预研用 speechSynthesis 女声 + 蜂鸣 |
| Q2 | **Web 与 voice-routerd 分进程** |
| Q3 | **ESP32 继续作风扇执行端**（预研展示 + 近期量产演示） |
| Q4 | 网页 **无登录**（局域网信任域） |
| Q5 | **指令采集优先**：演示时开 `SAVE_ASR_WAV`，用着采，服务 NLU/规则（不重训 ASR） |
| Q6 | **唤醒双线研发**：A 线能量唤醒（现设备）与 B 线 KWS 并行，量产仅 B 线 |

## 录音三条线（勿混淆）

| 线 | 用途 | v1 优先级 |
|----|------|-----------|
| `audio_clips/` | TTS 播放听感，**不训练模型** | 低，可延后 |
| `asr_debug/` | 指令 WAV + 误识别标注 → NLU/规则 | **高** |
| `kws_samples/` | 唤醒词正/负样本 → KWS 调参评估 | 中，B 线并行 |

## 评审结论（自审）

| 维度 | 结论 |
|------|------|
| 需求理解 | ✅ 五项与产品约束一致；录音用途已拆分 |
| 设计合理性 | ✅ 唤醒双线并行；TTS 与模型数据解耦 |
| 任务可执行 | ✅ 1A 指令采集优先；1B TTS 降为可选 |
| 风险 | ⚠️ ASR 准确率依赖 `asr_debug` 样本量；KWS B 线需好麦验收 |

**实现入口：** [design.md](./design.md) + [tasks.md](./tasks.md)（基线 v1.0，2026-06-26 锁定）。变更须先改基线再写码。
