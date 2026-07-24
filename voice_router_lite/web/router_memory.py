"""路由器内存实时监控：量产状态机模拟 + 真机 RSS / meminfo 采样。"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from voice_router_lite.config import (
    EngineMemoryEstimate,
    PipelineConfig,
    estimate_engine_memory,
    router_default_config,
)

logger = logging.getLogger(__name__)

PHASE_STANDBY = "standby"
PHASE_ASR = "asr_active"

MEMORY_SNAPSHOT_PATH = Path(
    os.getenv("VOICE_ROUTER_MEMORY_SNAPSHOT", "/tmp/voice-router-memory.json")
)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _detect_router_device() -> bool:
    try:
        with open("/proc/meminfo") as f:
            total_kb = int(f.readline().split()[1])
            return total_kb < 512 * 1024
    except OSError:
        return False


def read_linux_meminfo_mb() -> dict[str, float]:
    """读取 Linux 全机内存（OpenWrt 路由器）。"""
    try:
        with open("/proc/meminfo") as f:
            raw: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    raw[parts[0].rstrip(":")] = int(parts[1])
        total = raw.get("MemTotal", 0) / 1024.0
        avail = raw.get("MemAvailable", raw.get("MemFree", 0)) / 1024.0
        used = max(0.0, total - avail)
        return {"total_mb": total, "used_mb": used, "available_mb": avail}
    except OSError:
        return {}


def process_tree_rss_mb(pid: int | None = None) -> float:
    """当前进程及子进程 RSS（MB），用于 daemon 或 ASR worker。"""
    if not HAS_PSUTIL:
        return 0.0
    try:
        proc = psutil.Process(pid or os.getpid())
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / (1024 * 1024)
    except Exception:
        return 0.0


def write_daemon_memory_snapshot(
    *,
    rss_mb: float,
    phase: str,
    estimate: EngineMemoryEstimate,
) -> None:
    """voice-routerd 写入快照，供同机 Web 控制台读取。"""
    try:
        meminfo = read_linux_meminfo_mb()
        payload = {
            "ts": time.time(),
            "phase": phase,
            "rss_mb": round(rss_mb, 1),
            "memory": estimate.to_dict(),
            "meminfo": {k: round(v, 1) for k, v in meminfo.items()},
        }
        MEMORY_SNAPSHOT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("写入内存快照失败: %s", exc)


def read_daemon_memory_snapshot(max_age_sec: float = 8.0) -> dict | None:
    try:
        if not MEMORY_SNAPSHOT_PATH.is_file():
            return None
        data = json.loads(MEMORY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if time.time() - float(data.get("ts", 0)) > max_age_sec:
            return None
        return data
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


class RouterMemoryTracker:
    """跟踪 OpenWrt 量产内存形态，并在采样周期内刷新监控值。"""

    def __init__(self) -> None:
        self._cfg = router_default_config()
        self._prefer_int8 = True
        self._phase = PHASE_STANDBY
        self._asr_until = 0.0
        self._mode = self._resolve_mode()
        self._last_estimate: EngineMemoryEstimate | None = None

    @staticmethod
    def _resolve_mode() -> str:
        env = os.getenv("VOICE_ROUTER_MEMORY_MODE", "").strip().lower()
        if env in {"live_rss", "router_sim", "daemon_snapshot"}:
            return env
        if _detect_router_device():
            return "live_rss"
        return "router_sim"

    def configure(self, cfg: PipelineConfig, *, prefer_int8: bool) -> None:
        self._prefer_int8 = prefer_int8
        if cfg.deployment_mode == "router":
            self._cfg = cfg
        else:
            sim = router_default_config()
            sim.prefer_int8_nlu = prefer_int8
            self._cfg = sim

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def phase(self) -> str:
        return self._phase

    def enter_asr(self, hold_sec: float = 10.0) -> None:
        self._phase = PHASE_ASR
        self._asr_until = time.time() + hold_sec

    def exit_asr(self) -> None:
        self._phase = PHASE_STANDBY
        self._asr_until = 0.0

    def _maybe_revert_phase(self) -> None:
        if self._phase == PHASE_ASR and self._asr_until > 0 and time.time() > self._asr_until:
            self._phase = PHASE_STANDBY
            self._asr_until = 0.0

    def _estimate_router_phase(self) -> EngineMemoryEstimate:
        est = estimate_engine_memory(
            self._cfg,
            profile="router",
            prefer_int8_nlu=self._prefer_int8,
            phase=self._phase,
        )
        est.mode = self._mode
        est.phase = self._phase
        est.scope = "router_sim"
        est.note = (
            "Mac 预研：模拟 OpenWrt voice-routerd（待机 KWS+NLU ↔ 识别 NLU+ASR）"
        )
        return est

    def _estimate_live_rss(self) -> EngineMemoryEstimate:
        est = self._estimate_router_phase()
        rss = process_tree_rss_mb()
        meminfo = read_linux_meminfo_mb()
        est.mode = "live_rss"
        est.phase = self._phase
        est.rss_mb = rss
        if meminfo:
            est.device_current_mb = meminfo.get("used_mb", est.device_current_mb)
            est.device_total_mb = meminfo.get("total_mb", float(self._cfg.performance.device_total_memory_mb))
        else:
            est.device_current_mb = float(self._cfg.performance.system_reserved_memory_mb) + est.engine_current_mb
            est.device_total_mb = float(self._cfg.performance.device_total_memory_mb)
        est.scope = "live_rss"
        est.note = "OpenWrt 真机：全机 used 来自 /proc/meminfo，进程 RSS 见 rss_mb"
        return est

    def _estimate_from_daemon_snapshot(self, snap: dict) -> EngineMemoryEstimate:
        est = self._estimate_router_phase()
        est.mode = "daemon_snapshot"
        est.phase = str(snap.get("phase", PHASE_STANDBY))
        est.rss_mb = float(snap.get("rss_mb") or 0.0)
        mem = snap.get("memory") or {}
        meminfo = snap.get("meminfo") or {}
        if meminfo.get("used_mb") is not None:
            est.device_current_mb = float(meminfo["used_mb"])
        elif mem.get("device_current_mb") is not None:
            est.device_current_mb = float(mem["device_current_mb"])
        else:
            est.device_current_mb = float(mem.get("device_standby_mb") or est.device_standby_mb)
        est.device_total_mb = float(
            meminfo.get("total_mb") or self._cfg.performance.device_total_memory_mb
        )
        est.scope = "daemon_snapshot"
        est.note = "来自 voice-routerd 快照（/tmp/voice-router-memory.json）"
        return est

    def tick(self) -> EngineMemoryEstimate:
        self._maybe_revert_phase()

        if self._mode == "daemon_snapshot" or (
            self._mode == "router_sim" and read_daemon_memory_snapshot() is not None
        ):
            snap = read_daemon_memory_snapshot()
            if snap is not None:
                self._last_estimate = self._estimate_from_daemon_snapshot(snap)
                return self._last_estimate

        if self._mode == "live_rss":
            self._last_estimate = self._estimate_live_rss()
            return self._last_estimate

        self._last_estimate = self._estimate_router_phase()
        return self._last_estimate

    @property
    def last_estimate(self) -> EngineMemoryEstimate | None:
        return self._last_estimate
