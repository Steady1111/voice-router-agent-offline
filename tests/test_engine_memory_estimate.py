"""路由器内存状态机与实时监控。"""

import os

import pytest

from voice_router_lite.config import (
    PipelineConfig,
    estimate_engine_memory,
    router_default_config,
)
from voice_router_lite.web.monitor import PerformanceMonitor
from voice_router_lite.web.router_memory import (
    PHASE_ASR,
    PHASE_STANDBY,
    RouterMemoryTracker,
)


def test_router_standby_vs_asr_phase():
    cfg = router_default_config()
    standby = estimate_engine_memory(cfg, profile="router", phase=PHASE_STANDBY)
    active = estimate_engine_memory(cfg, profile="router", phase=PHASE_ASR)
    assert standby.engine_current_mb == 17.0
    assert active.engine_current_mb == 34.0
    assert standby.device_current_mb == 77.0
    assert active.device_current_mb == 94.0
    assert active.kws_mb == 0.0
    assert active.asr_mb == 25.0


def test_tracker_phase_switch():
    tracker = RouterMemoryTracker()
    tracker.configure(router_default_config(), prefer_int8=True)
    tracker._mode = "router_sim"
    est_idle = tracker.tick()
    assert est_idle.device_current_mb == 77.0
    tracker.enter_asr(hold_sec=30)
    est_asr = tracker.tick()
    assert est_asr.device_current_mb == 94.0
    tracker.exit_asr()
    assert tracker.tick().device_current_mb == 77.0


def test_monitor_sample_updates_history():
    mon = PerformanceMonitor()
    mon._running = False
    mon.configure_router_memory(router_default_config(), prefer_int8=True)
    mon.enter_router_asr()
    mon._sample()
    assert mon._engine_ram_mb == 94.0
    mon.exit_router_asr()
    mon._sample()
    assert mon._engine_ram_mb == 77.0


def test_resolve_nlu_int8_when_file_exists():
    cfg = PipelineConfig()
    from voice_router_lite.config import resolve_nlu_runtime_mb
    mb, variant = resolve_nlu_runtime_mb(cfg, prefer_int8_nlu=True)
    if os.path.exists(cfg.models.nlu_intent_model_int8):
        assert variant == "int8"
        assert mb == 4.0
