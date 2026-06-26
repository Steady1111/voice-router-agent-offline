"""OpenWrt / 路由器量产部署模块（ALSA 直连，无 WebSocket）。"""

from voice_router_lite.router.daemon import run_router_daemon
from voice_router_lite.config import router_default_config

__all__ = ["run_router_daemon", "router_default_config"]
