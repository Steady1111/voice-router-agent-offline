"""OpenWrt UBus integration for router management commands."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

_UBUS_CLIENT: Optional["UBusClient"] = None


class UBusClient:
    """Thin wrapper around the `ubus` CLI shipped with OpenWrt."""

    def __init__(self) -> None:
        self._available = shutil.which("ubus") is not None
        if not self._available:
            logger.info("ubus CLI 不可用，路由器指令将使用模拟模式")

    @property
    def available(self) -> bool:
        return self._available

    def call(self, object_name: str, method: str, message: Optional[dict] = None) -> Any:
        if not self._available:
            return None

        cmd = ["ubus", "call", object_name, method]
        if message is not None:
            cmd.append(json.dumps(message, ensure_ascii=False))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    "ubus 调用失败: %s %s -> %s",
                    object_name,
                    method,
                    result.stderr.strip() or result.stdout.strip(),
                )
                return None
            output = result.stdout.strip()
            if not output:
                return {}
            return json.loads(output)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            logger.warning("ubus 调用异常 (%s %s): %s", object_name, method, exc)
            return None

    def reboot(self) -> bool:
        result = self.call("system", "reboot")
        return result is not None

    def get_system_info(self) -> dict[str, Any]:
        info = self.call("system", "info") or {}
        board = self.call("system", "board") or {}
        return {"system": info, "board": board}

    def get_network_devices(self) -> list[dict[str, Any]]:
        dump = self.call("network.device", "status") or {}
        devices = []
        if isinstance(dump, dict):
            for name, data in dump.items():
                if isinstance(data, dict):
                    devices.append({"name": name, **data})
        return devices

    def wireless_restart(self) -> bool:
        # OpenWrt 常见接口：重启无线
        result = self.call("network.wireless", "up")
        return result is not None


def get_ubus_client() -> UBusClient:
    global _UBUS_CLIENT
    if _UBUS_CLIENT is None:
        _UBUS_CLIENT = UBusClient()
    return _UBUS_CLIENT
