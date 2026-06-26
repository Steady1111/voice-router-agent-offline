"""路由器量产架构测试"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_router_lite.config import router_default_config, PipelineConfig, ModelPaths
from voice_router_lite.router.asr_subprocess import resolve_asr_worker_cmd
from voice_router_lite.router.model_scheduler import ModelScheduler


class TestRouterConfig:
    def test_router_defaults(self):
        cfg = router_default_config()
        assert cfg.deployment_mode == "router"
        assert cfg.model_serial_exclusive is True
        assert cfg.asr_subprocess is True
        assert cfg.prefer_int8_nlu is True
        assert cfg.asr_num_threads == 1
        assert cfg.use_openwrt_ubus is True
        assert cfg.audio.buffer_duration_sec == 2.0

    def test_resolve_nlu_int8(self, tmp_path):
        models = ModelPaths(base_dir=str(tmp_path))
        fp32 = os.path.join(tmp_path, "intent_model.onnx")
        int8 = os.path.join(tmp_path, "intent_model.int8.onnx")
        open(fp32, "w").close()
        open(int8, "w").close()
        models.nlu_intent_model = fp32
        models.nlu_intent_model_int8 = int8
        assert models.resolve_nlu_model(False) == fp32
        assert models.resolve_nlu_model(True) == int8


class TestAsrSubprocess:
    def test_resolve_python_fallback(self, monkeypatch):
        monkeypatch.delenv("VOICE_ROUTER_ASR_WORKER", raising=False)
        monkeypatch.setattr(
            "voice_router_lite.router.asr_subprocess._repo_native_worker",
            lambda: None,
        )
        cfg = router_default_config()
        cmd = resolve_asr_worker_cmd(cfg)
        assert cmd[0] == sys.executable
        assert "voice_router_lite.router.asr_worker" in cmd

    def test_resolve_native_env(self, tmp_path, monkeypatch):
        worker = tmp_path / "voice-router-asr-worker"
        worker.write_text("#!/bin/sh\necho ok\n")
        worker.chmod(0o755)
        monkeypatch.setenv("VOICE_ROUTER_ASR_WORKER", str(worker))
        cfg = router_default_config()
        cmd = resolve_asr_worker_cmd(cfg)
        assert cmd[0] == str(worker)
        assert "--models-dir" in cmd
        assert cfg.models.base_dir in cmd


class TestModelScheduler:
    def test_serial_exclusive_unloads_kws(self):
        cfg = router_default_config()

        class FakeKws:
            closed = False

            def close(self):
                self.closed = True

        class FakeAsr:
            loaded = False
            unloaded = False

            def ensure_loaded(self):
                self.loaded = True

            def transcribe(self, audio):
                return type("R", (), {"text": "打开风扇"})()

            def unload(self):
                self.unloaded = True

        kws = FakeKws()
        asr = FakeAsr()
        sched = ModelScheduler(cfg, kws, asr)  # type: ignore

        audio = np.zeros(1600, dtype=np.int16)
        cfg.asr_subprocess = False
        text = sched.transcribe(audio)

        assert kws.closed is True
        assert asr.loaded is True
        assert asr.unloaded is True
        assert text == "打开风扇"

    def test_pipeline_enables_scheduler(self):
        from voice_router_lite.pipeline import VoiceRouterPipeline

        cfg = PipelineConfig()
        cfg.deployment_mode = "router"
        cfg.model_serial_exclusive = True
        pipeline = VoiceRouterPipeline(cfg)
        pipeline.initialize()
        assert pipeline._scheduler is not None
        pipeline.close()
