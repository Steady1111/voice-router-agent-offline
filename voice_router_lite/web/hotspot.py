"""Mac 预研环境：读取个人热点下联设备数。"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_HOTSPOT_IFACES = ("bridge100", "bridge101", "ap1")


def mac_hotspot_client_count() -> Optional[int]:
    """统计 Mac 所连热点上的下联设备数（不含本机与网关）。"""
    if platform.system() != "Darwin":
        return None

    hotspot = _detect_constrained_hotspot()
    if hotspot:
        iface, self_ip = hotspot
        count = _count_arp_clients(iface, exclude_ips={self_ip, _gateway_ip(self_ip)})
        if count is not None:
            return count

    dhcp = _count_dhcp_leases()
    if dhcp is not None:
        return dhcp

    for iface in _list_bridge_ifaces():
        count = _count_arp_clients(iface)
        if count is not None and count > 0:
            return count

    return None


def _detect_constrained_hotspot() -> Optional[tuple[str, str]]:
    """macOS 连 iPhone 热点时接口带 constrained 标记（常见 en0）。"""
    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    blocks: list[tuple[str, str]] = []
    current: Optional[str] = None
    lines: list[str] = []

    for line in result.stdout.splitlines():
        if line and not line.startswith(("\t", " ")) and ":" in line:
            if current is not None:
                blocks.append((current, "\n".join(lines)))
            iface, _, rest = line.partition(":")
            current = iface
            lines = [rest.strip()] if rest.strip() else []
        elif current is not None:
            lines.append(line)

    if current is not None:
        blocks.append((current, "\n".join(lines)))

    for iface, block in blocks:
        if "constrained" not in block:
            continue
        match = re.search(r"^\s*inet (\d+\.\d+\.\d+\.\d+)", block, re.MULTILINE)
        if match:
            return iface, match.group(1)
    return None


def _gateway_ip(self_ip: str) -> str:
    """热点网关多为 x.x.x.1。"""
    parts = self_ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.1"
    return ""


def _list_bridge_ifaces() -> list[str]:
    found: list[str] = []
    try:
        result = subprocess.run(
            ["ifconfig", "-l"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            for name in result.stdout.split():
                if name in _HOTSPOT_IFACES or name.startswith("bridge"):
                    found.append(name)
    except (OSError, subprocess.TimeoutExpired):
        pass
    for iface in _HOTSPOT_IFACES:
        if iface not in found:
            found.append(iface)
    return found


def _count_dhcp_leases(path: str = "/var/db/dhcpd_leases") -> Optional[int]:
    try:
        from pathlib import Path

        leases = Path(path)
        if not leases.is_file():
            return None
        text = leases.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.debug("无法读取 DHCP 租约 %s: %s", path, exc)
        return None

    ips = re.findall(r"^ip_address=(\S+)", text, flags=re.MULTILINE)
    return len(ips) if ips else 0


def _count_arp_clients(
    iface: str,
    *,
    exclude_ips: Optional[set[str]] = None,
) -> Optional[int]:
    exclude_ips = exclude_ips or set()
    try:
        result = subprocess.run(
            ["arp", "-an", "-i", iface],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    count = 0
    for line in result.stdout.splitlines():
        lower = line.lower()
        if " at " not in line or "incomplete" in lower:
            continue
        match = re.search(r"\((\d+\.\d+\.\d+\.\d+)\)", line)
        if not match:
            continue
        ip = match.group(1)
        if ip in exclude_ips:
            continue
        count += 1
    return count
