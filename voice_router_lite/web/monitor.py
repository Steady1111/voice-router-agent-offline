"""Performance monitor for real-time metrics collection.

Tracks: peak RAM, memory fragmentation, CPU load, latency, recognition accuracy,
MTBF, and thermal profile. Designed for the web console's monitoring panel.

重要说明：
- CPU 监控：基于实际推理耗时估算，不依赖 psutil（开发机 CPU 不代表路由器 CPU）
- 内存监控：仅监控离线引擎核心模块（KWS/ASR/NLU/音频缓冲）的内存占用，
  不包含 Python 解释器、系统库等与路由器部署无关的开销。
"""

from __future__ import annotations

import os
import platform
import time
import threading
from collections import deque
from dataclasses import dataclass, field

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class MetricHistory:
    """Rolling time-series buffer for a single metric."""
    values: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))

    def add(self, value: float, timestamp: float | None = None) -> None:
        self.values.append((timestamp or time.time(), value))

    def latest(self) -> float | None:
        return self.values[-1][1] if self.values else None

    def to_list(self) -> list[dict]:
        return [{"t": t, "v": round(v, 2)} for t, v in self.values]


class PerformanceMonitor:
    """Singleton performance monitor collecting system and application metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._error_count = 0
        self._request_count = 0
        self._total_latency = 0.0
        self._latency_samples: deque[float] = deque(maxlen=1000)
        self._accuracy_samples: deque[float] = deque(maxlen=1000)

        # Metric histories for time-series charts
        self.ram_history = MetricHistory()
        self.frag_history = MetricHistory()
        self.cpu_history = MetricHistory()
        self.latency_history = MetricHistory()
        self.accuracy_history = MetricHistory()
        self.error_history = MetricHistory()
        self.thermal_history = MetricHistory()

        # 音频电平监控（ESP32 KWS 链路写入）
        self._audio_energy: float = 0.0
        self._audio_state: str = "idle"  # idle / listening / recording / wake

        # ESP32 外接温度传感器（DS18B20 等经 WebSocket telemetry 上报）
        self._esp32_temp_celsius: float | None = None
        self._esp32_temp_at: float = 0.0

        # ------------------------------------------------------------------
        # 路由器引擎核心内存估算（仅离线引擎模块，不含 Python 运行时等开销）
        # ------------------------------------------------------------------
        self._engine_ram_mb: float = 0.0      # 当前估算内存 (MB)
        self._peak_engine_ram_mb: float = 0.0  # 峰值内存 (MB)
        self._engine_ram_baseline_mb: float = 0.0  # 基线内存 (MB)

        # ------------------------------------------------------------------
        # CPU 推理耗时累积（用于估算路由器 CPU 占用率）
        # ------------------------------------------------------------------
        self._inference_time_ms: float = 0.0   # 本采样窗口内的推理总耗时 (ms)
        self._inference_count: int = 0         # 本窗口内的推理次数
        self._sample_interval_s: float = 2.0   # 采样窗口 (秒)

        # Background sampling
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self, interval: float = 2.0) -> None:
        """Start background metric collection."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    #  Background sampling
    # ------------------------------------------------------------------

    def _collect_loop(self, interval: float) -> None:
        self._sample_interval_s = interval
        while self._running:
            self._sample()
            time.sleep(interval)

    def _sample(self) -> None:
        with self._lock:
            now = time.time()

            # ---- 内存：引擎核心模块估算（不是 psutil 进程 RSS） ----
            ram_mb = self._engine_ram_mb
            if ram_mb > self._peak_engine_ram_mb:
                self._peak_engine_ram_mb = ram_mb
            self.ram_history.add(ram_mb, now)

            # 碎片率：基于引擎基线内存的增长比例
            if self._engine_ram_baseline_mb > 0:
                frag = (ram_mb - self._engine_ram_baseline_mb) / self._engine_ram_baseline_mb * 100
                frag = max(0, frag)
            else:
                frag = 0
            self.frag_history.add(frag, now)

            # ---- CPU：基于实际推理耗时估算 ----
            # 公式：CPU% = (推理总耗时 / 采样窗口时长) * 100
            # 这反映的是"如果路由器单核跑这些推理，占用多少 CPU"
            cpu = 0.0
            if self._sample_interval_s > 0:
                cpu = (self._inference_time_ms / (self._sample_interval_s * 1000)) * 100
                cpu = min(cpu, 100.0)  # 上限 100%
            self.cpu_history.add(cpu, now)

            # 重置本窗口的推理累积
            self._inference_time_ms = 0.0
            self._inference_count = 0

            self._sample_thermal(now)

    def _sample_thermal(self, now: float) -> None:
        """Try Linux thermal zones, then hwmon fallback."""
        temp: float | None = None

        for zone in ["/sys/class/thermal/thermal_zone0/temp",
                      "/sys/class/thermal/thermal_zone1/temp"]:
            try:
                with open(zone) as f:
                    temp = int(f.read().strip()) / 1000.0
                    break
            except (OSError, ValueError):
                continue

        if temp is None and os.path.exists("/sys/class/hwmon"):
            for root, _dirs, files in os.walk("/sys/class/hwmon"):
                for fname in files:
                    if fname.endswith("_input") and "temp" in fname:
                        try:
                            with open(os.path.join(root, fname)) as f:
                                temp = int(f.read().strip()) / 1000.0
                                break
                        except Exception:
                            continue
                if temp is not None:
                    break

        if temp is not None:
            self.thermal_history.add(temp, now)

    # ------------------------------------------------------------------
    #  Public recording methods (called from routes / service)
    # ------------------------------------------------------------------

    def record_request(self, latency_ms: float, success: bool = True) -> None:
        """Record a completed NLU request."""
        with self._lock:
            self._request_count += 1
            self._total_latency += latency_ms
            self._latency_samples.append(latency_ms)
            self.latency_history.add(latency_ms, time.time())
            if not success:
                self._error_count += 1

    def record_accuracy(self, confidence: float) -> None:
        """Record an NLU confidence score."""
        with self._lock:
            self._accuracy_samples.append(confidence)
            self.accuracy_history.add(confidence, time.time())

    def record_error(self) -> None:
        """Record an error event."""
        with self._lock:
            self._error_count += 1
            self.error_history.add(self._error_count, time.time())

    def update_audio_energy(self, energy: float, state: str = "listening") -> None:
        """Update ESP32 audio energy level (called from ws_audio KWS loop)."""
        with self._lock:
            self._audio_energy = energy
            self._audio_state = state

    def update_esp32_temperature(self, temp_c: float) -> None:
        """Update temperature from ESP32 external sensor telemetry."""
        with self._lock:
            self._esp32_temp_celsius = float(temp_c)
            self._esp32_temp_at = time.time()
            self.thermal_history.add(self._esp32_temp_celsius, self._esp32_temp_at)

    # ------------------------------------------------------------------
    #  路由器引擎资源追踪（CPU + 内存，仅离线引擎核心模块）
    # ------------------------------------------------------------------

    def update_engine_memory(self, ram_mb: float) -> None:
        """更新离线引擎核心模块的内存估算值 (MB)。

        由 Pipeline 初始化完成后调用，传入各模块实际估算的内存总量。
        仅包括：KWS模型 + NLU模型 + 音频缓冲区。
        不包含 Python 解释器、系统库等与路由器部署无关的开销。
        """
        with self._lock:
            self._engine_ram_mb = ram_mb
            if self._engine_ram_baseline_mb == 0:
                self._engine_ram_baseline_mb = ram_mb

    def record_inference_time(self, stage: str, elapsed_ms: float) -> None:
        """记录一次推理耗时（KWS / NLU），用于估算路由器 CPU 占用。

        stage: 推理阶段标识，如 "kws", "nlu"
        elapsed_ms: 该阶段推理耗时（毫秒）
        """
        with self._lock:
            self._inference_time_ms += elapsed_ms
            self._inference_count += 1

    # ------------------------------------------------------------------
    #  Snapshot API
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a single-point-in-time summary of all metrics.

        所有资源指标均基于离线引擎核心模块估算：
        - 内存：仅 KWS/ASR/NLU 模型 + 音频缓冲区
        - CPU：基于实际推理耗时 / 采样窗口计算
        """
        with self._lock:
            uptime = time.time() - self._start_time

            current_ram_mb = self._engine_ram_mb
            peak_ram_mb = self._peak_engine_ram_mb

            # 目标设备：128MB 路由器
            TARGET_RAM_MB = 128
            ram_usage_pct = round(current_ram_mb / TARGET_RAM_MB * 100, 1) if current_ram_mb else None

            # Latency stats
            p50, p95, p99 = 0.0, 0.0, 0.0
            avg_latency = 0.0
            if self._latency_samples:
                sorted_lat = sorted(self._latency_samples)
                n = len(sorted_lat)
                p50 = sorted_lat[int(n * 0.50)]
                p95 = sorted_lat[int(n * 0.95)]
                p99 = sorted_lat[int(n * 0.99)]
                avg_latency = self._total_latency / self._request_count if self._request_count else 0.0

            avg_accuracy = 0.0
            high_conf_ratio = 0.0
            if self._accuracy_samples:
                avg_accuracy = sum(self._accuracy_samples) / len(self._accuracy_samples)
                high_conf_count = sum(1 for s in self._accuracy_samples if s > 0.8)
                high_conf_ratio = high_conf_count / len(self._accuracy_samples) * 100

            mtbf = uptime / max(self._error_count, 1)

            # 判断是否在目标路由器上运行
            is_router = self._detect_router()

            return {
                "uptime_seconds": round(uptime, 1),
                "uptime_formatted": self._format_uptime(uptime),
                # 内存 — 以路由器 128MB 为基准，仅引擎核心模块
                "target_ram_mb": TARGET_RAM_MB,
                "current_ram_mb": round(current_ram_mb, 1),
                "ram_usage_pct": ram_usage_pct,
                "peak_ram_mb": round(peak_ram_mb, 1),
                "fragmentation_pct": round(self.frag_history.latest() or 0, 1),
                # CPU — 基于实际推理耗时估算
                "cpu_percent": round(self.cpu_history.latest() or 0, 1),
                # 环境标识
                "host_info": {
                    "system": platform.system(),
                    "node": platform.node(),
                    "machine": platform.machine(),
                    "is_router": is_router,
                    "note": "CPU/内存均基于离线引擎核心模块估算，非开发机实际指标",
                },
                "latency": {
                    "avg_ms": round(avg_latency, 1),
                    "p50_ms": round(p50, 1),
                    "p95_ms": round(p95, 1),
                    "p99_ms": round(p99, 1),
                },
                "accuracy": {
                    "avg_pct": round(avg_accuracy * 100, 1) if self._accuracy_samples else None,
                    "high_conf_pct": round(high_conf_ratio, 1),
                    "sample_count": len(self._accuracy_samples),
                },
                "mtbf_seconds": round(mtbf, 1),
                "mtbf_formatted": self._format_uptime(mtbf),
                "thermal_celsius": self._latest_thermal_celsius(),
                "thermal_source": self._thermal_source(),
                "error_count": self._error_count,
                "request_count": self._request_count,
                "has_psutil": HAS_PSUTIL,
                "sample_timestamp": time.time(),
                # 音频电平监控
                "audio": {
                    "energy": round(self._audio_energy, 5),
                    "state": self._audio_state,
                    "level_pct": round(min(self._audio_energy * 100, 100), 1),
                },
            }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_router() -> bool:
        """Heuristic: detect if running on an embedded router (OpenWRT / low RAM)."""
        try:
            with open("/proc/meminfo") as f:
                total_kb = int(f.readline().split()[1])
                return total_kb < 512 * 1024  # < 512MB 视为嵌入式设备
        except Exception:
            return False

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _latest_thermal_celsius(self) -> float | None:
        if self._esp32_temp_celsius is not None and (time.time() - self._esp32_temp_at) < 120:
            return round(self._esp32_temp_celsius, 1)
        host = self.thermal_history.latest()
        return round(host, 1) if host is not None else None

    def _thermal_source(self) -> str:
        if self._esp32_temp_celsius is not None and (time.time() - self._esp32_temp_at) < 120:
            return "esp32"
        if self.thermal_history.latest() is not None:
            return "host"
        return "none"


# ------------------------------------------------------------------
#  Singleton
# ------------------------------------------------------------------

_monitor: PerformanceMonitor | None = None


def get_monitor() -> PerformanceMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
        _monitor.start()
    return _monitor


def shutdown_monitor() -> None:
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None
