# KWS 唤醒词样本（B 线）

用于 `tools/eval_kws_samples.py` 评估「小T小T」命中率。**不用于重训 sherpa 模型。**

## 目录

| 目录 | 内容 |
|------|------|
| `positive/` | 正样本：清晰说出「小T小T」 |
| `negative/` | 负样本：环境噪声、相似发音、日常对话 |

## 录制规范

- 16 kHz、mono、16-bit PCM WAV
- 建议 **Mac / USB 好麦** 录制（不要用 ESP32 INMP441 做 B 线验收）
- 正样本每条 1–2 秒，可说 1–2 遍唤醒词
- 负样本 2–5 秒环境音或易误触语句

## 评估

```bash
VOICE_ROUTER_ENERGY_WAKE=0 python3 -m voice_router_lite web   # 另终端
python3 tools/eval_kws_samples.py --root kws_samples
```

通过线见 [docs/06-优化基线.md](../docs/06-优化基线.md) AC-3b：正样本 ≥14/20，负样本误唤醒 ≤2/10。  
录制与评估说明见 [docs/05-模型与数据采集.md](../docs/05-模型与数据采集.md)。
