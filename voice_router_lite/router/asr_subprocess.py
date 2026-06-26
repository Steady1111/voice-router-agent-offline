"""在独立子进程中运行 ASR，进程退出后由内核回收全部模型内存。"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from voice_router_lite.config import PipelineConfig

logger = logging.getLogger(__name__)


def _repo_native_worker() -> Path | None:
    """开发树内 native/bin/voice-router-asr-worker（若已编译）。"""
    root = Path(__file__).resolve().parents[2]
    candidate = root / "native" / "bin" / "voice-router-asr-worker"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def resolve_asr_worker_cmd(config: PipelineConfig) -> list[str]:
    """优先使用 C++ 原生 worker，否则回退 Python asr_worker 模块。"""
    env_path = os.getenv("VOICE_ROUTER_ASR_WORKER")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    native = _repo_native_worker()
    if native is not None:
        candidates.append(native)

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            logger.debug("ASR worker: native %s", path)
            return [
                str(path),
                "--models-dir",
                config.models.base_dir,
                "--sample-rate",
                str(config.audio.sample_rate),
                "--threads",
                str(config.asr_num_threads),
            ]

    logger.debug("ASR worker: Python fallback")
    return [
        sys.executable,
        "-m",
        "voice_router_lite.router.asr_worker",
        "--sample-rate",
        str(config.audio.sample_rate),
        "--threads",
        str(config.asr_num_threads),
    ]


def _sherpa_native_lib_dir() -> str | None:
    """pip sherpa_onnx 自带的动态库目录（macOS/Linux 运行时需要）。"""
    try:
        import importlib.util

        spec = importlib.util.find_spec("sherpa_onnx")
        if not spec or not spec.origin:
            return None
        lib_dir = Path(spec.origin).parent / "lib"
        if lib_dir.is_dir():
            return str(lib_dir)
    except Exception:
        return None
    return None


def _subprocess_env(models_dir: str) -> dict[str, str]:
    env = {**dict(os.environ), "VOICE_ROUTER_MODELS_DIR": models_dir}
    lib_dir = _sherpa_native_lib_dir()
    if not lib_dir:
        return env
    if sys.platform == "darwin":
        prev = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = lib_dir + (f":{prev}" if prev else "")
    elif sys.platform.startswith("linux"):
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir + (f":{prev}" if prev else "")
    return env


def transcribe_in_subprocess(audio_int16: np.ndarray, config: PipelineConfig) -> str:
    """将 PCM 送入 asr_worker 子进程，返回识别文本。"""
    if audio_int16.dtype != np.int16:
        audio_int16 = audio_int16.astype(np.int16)

    cmd = resolve_asr_worker_cmd(config)
    env = _subprocess_env(config.models.base_dir)

    try:
        proc = subprocess.run(
            cmd,
            input=audio_int16.tobytes(),
            capture_output=True,
            timeout=30,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("ASR 子进程超时")
        return ""

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.error("ASR 子进程失败 (code=%s): %s", proc.returncode, err)
        return ""

    text = proc.stdout.decode("utf-8", errors="replace").strip()
    logger.info("ASR 子进程识别: '%s'", text)
    return text
