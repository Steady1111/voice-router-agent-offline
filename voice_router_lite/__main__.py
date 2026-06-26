"""Package entrypoint: web console, dev pipeline, or router daemon."""

from __future__ import annotations

import argparse
import logging
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Voice Router Lite")
    parser.add_argument(
        "command",
        nargs="?",
        default="web",
        choices=["web", "pipeline", "router", "prepare-models"],
        help="web=预研看板, pipeline=开发管道, router=路由器守护进程, prepare-models=检查模型",
    )
    parser.add_argument("--text", help="pipeline/router 模式下直接执行文本指令")
    parser.add_argument("--config", help="JSON 配置文件路径")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.command == "web":
        from voice_router_lite.web.app import main as web_main

        web_main()
        return 0

    if args.command == "prepare-models":
        from voice_router_lite.tools.prepare_models import main as prepare_main

        return prepare_main([])

    if args.command == "router":
        from voice_router_lite.router.daemon import main as router_main

        router_argv = []
        if args.config:
            router_argv.extend(["--config", args.config])
        return router_main(router_argv)

    from voice_router_lite import VoiceRouterPipeline, PipelineConfig

    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    pipeline = VoiceRouterPipeline(config)
    pipeline.initialize()

    try:
        if args.text:
            result = pipeline.process_utterance(text=args.text)
            print(result)
            return 0 if result.get("success") else 1

        pipeline.start_listening()
        print("持续监听中，Ctrl+C 退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止监听")
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
