# scripts/dev — 预研调试脚本

临时/本机串口与 KWS 排查用，**非运行时依赖**。多数硬编码了开发机串口路径，使用前请按本机修改。

| 脚本 | 用途 |
|------|------|
| `flash_esp32.sh` / `flash_and_monitor.py` | 烧录并监视 |
| `monitor_esp32.py` / `serial_monitor.py` / `_m.py` / `_monitor.py` | 串口监视 |
| `scan_only.py` | 仅看 WiFi 扫描日志 |
| `debug_kws_model.py` / `diagnose_kws.py` | KWS 排查 |
| `test_kws_live.py` / `test_kws_record.py` | KWS 现场/录音试跑 |
| `verify_preresearch.sh` | 预研环境快速检查 |

正式采集与评估请用 `tools/`。
