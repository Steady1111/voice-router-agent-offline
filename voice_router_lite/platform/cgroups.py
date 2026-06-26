"""Optional cgroup v2 resource limits for voice service on Linux/OpenWrt."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CGROUP_ROOT = Path("/sys/fs/cgroup")


def apply_resource_limits(
    cpu_quota_pct: int = 30,
    memory_mb: int = 80,
    cgroup_name: str = "voice-router-lite",
) -> bool:
    """Apply CPU/memory limits to the current process when cgroup v2 is available."""
    if os.name != "posix" or not CGROUP_ROOT.is_dir():
        return False

    try:
        group = CGROUP_ROOT / cgroup_name
        group.mkdir(parents=True, exist_ok=True)

        period_us = 100_000
        quota_us = int(period_us * max(1, min(cpu_quota_pct, 100)) / 100)
        (group / "cpu.max").write_text(f"{quota_us} {period_us}\n")
        (group / "memory.max").write_text(f"{memory_mb * 1024 * 1024}\n")

        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            cgroup_path = handle.read().strip().split(":", 2)[-1]
        if cgroup_path.startswith("/"):
            cgroup_path = cgroup_path[1:]
        (group / "cgroup.procs").write_text(f"{os.getpid()}\n")
        logger.info(
            "已应用 cgroup 限制: cpu=%d%%, memory=%dMB (group=%s)",
            cpu_quota_pct,
            memory_mb,
            cgroup_name,
        )
        return True
    except OSError as exc:
        logger.debug("cgroup 限制未生效 (可能无 root 权限): %s", exc)
        return False
