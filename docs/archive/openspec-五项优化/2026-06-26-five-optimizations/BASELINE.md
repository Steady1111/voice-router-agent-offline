# 基线锁定记录

| 字段 | 值 |
|------|-----|
| **基线 ID** | `2026-06-26-five-optimizations` |
| **锁定日期** | 2026-06-26 |
| **状态** | `LOCKED` — 已进入 `/opsx:apply` 实现阶段 |
| **实现唯一依据** | [design.md](./design.md)、[tasks.md](./tasks.md) |

## 治理规则

1. **写代码前**：任务必须能在 `tasks.md` 中找到对应条目（如 `1A.3`）；架构/边界以 `design.md` 为准。
2. **冲突处理**：与 `proposal.md`、`spec.md`、`explore.md` 不一致时，**以 design + tasks 为准**。
3. **变更流程**：需求或方案变动 → 先改 `design.md` / `tasks.md` 并注明修订日期 → 再改代码。禁止只改代码不更新基线。
4. **验收**：完成标准以 `tasks.md` 各条「完成标准」+ `spec.md` 验收章节（AC-*）对照；spec 不单独新增未在 tasks 登记的工作项。
5. **归档**：全部 tasks 完成且 AC 通过后，变更移入 `docs/openspec/archive/`。

## 已锁定决策摘要（Q1–Q6）

| # | 决策 |
|---|------|
| Q1 | TTS 预录 WAV v1 可延后；预研网页女声 + 蜂鸣 |
| Q2 | Web 与 voice-routerd 分进程 |
| Q3 | ESP32 继续作风扇执行端 |
| Q4 | 网页无登录 |
| Q5 | 指令采集 `asr_debug` 优先（用着采，服务 NLU） |
| Q6 | 唤醒双线：A 能量 + B KWS 并行；量产仅 B |

## 文档层级

```
实现层（唯一依据）
  ├── design.md    架构、边界、技术方案
  └── tasks.md     可执行任务、文件路径、完成标准

参考层（不单独驱动实现）
  ├── proposal.md  背景与范围
  ├── spec.md      API / FR / 验收清单
  └── explore.md   边界细节展开
```

## 建议首批任务（来自 tasks.md）

1. **1A** — 指令样本采集（`SAVE_ASR_WAV`）
2. **1D + 2B** — router-status API + 前端展示
3. **1E** — KWS B 线
4. **2A** — NLU 再训（样本 ≥50 后）
5. **1B** — TTS 录音（量产前可选）
6. **1C + 3B** — INT8 / 内存

## 修订历史

| 日期 | 修订 | 说明 |
|------|------|------|
| 2026-06-26 | v1.0 | 初版基线锁定；含 Q5/Q6、TTS 延后、唤醒双线 |
