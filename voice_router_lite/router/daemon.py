"""
路由器量产守护进程：ALSA 直连 + 分时模型调度 + ubus 设备控制。

无 FastAPI / WebSocket，专为 OpenWrt procd 托管设计。
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from voice_router_lite.config import PipelineConfig, router_default_config
from voice_router_lite.pipeline import VoiceRouterPipeline

logger = logging.getLogger(__name__)


def run_router_daemon(config: PipelineConfig | None = None) -> int:
    """启动路由器语音守护进程（阻塞直到收到 SIGTERM/SIGINT）。"""
    cfg = config or router_default_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 50)
    logger.info("Voice Router 守护进程启动 (路由器量产模式)")
    logger.info(
        "策略: serial_exclusive=%s asr_subprocess=%s prefer_int8_nlu=%s",
        cfg.model_serial_exclusive,
        cfg.asr_subprocess,
        cfg.prefer_int8_nlu,
    )
    logger.info("=" * 50)

    pipeline = VoiceRouterPipeline(cfg)
    pipeline.initialize()
    pipeline.start_listening()

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        logger.info("收到信号 %s，准备退出...", signum)
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop:
            time.sleep(0.5)
    finally:
        pipeline.stop_listening()
        pipeline.close()
        logger.info("Voice Router 守护进程已退出")

    return 0


def main(argv: list[str] | None = None) -> int:
    config_path = None
    if argv:
        for i, arg in enumerate(argv):
            if arg == "--config" and i + 1 < len(argv):
                config_path = argv[i + 1]
    if config_path:
        cfg = PipelineConfig.from_json(config_path)
    else:
        cfg = router_default_config()
    return run_router_daemon(cfg)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
