#!/usr/bin/env python3
"""Evaluate KWS hit rate on kws_samples/ WAV files (task 1E.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from voice_router_lite.audio.capture import load_wav
from voice_router_lite.config import DEFAULT_CONFIG
from voice_router_lite.kws.engine import KWSEngine


def _eval_dir(engine: KWSEngine, directory: Path, expect_hit: bool) -> tuple[int, int]:
    if not directory.is_dir():
        return 0, 0
    hits = 0
    total = 0
    for wav in sorted(directory.glob("*.wav")):
        total += 1
        engine.reset()
        audio = load_wav(str(wav))
        audio_f = audio.astype(np.float32) / 32768.0
        # Chunk 0.5s windows (KWS expects streaming chunks)
        chunk = int(0.5 * 16000)
        detected = False
        for i in range(0, len(audio_f), chunk):
            part = audio_f[i : i + chunk]
            if len(part) < 1600:
                continue
            if engine.detect(part).detected:
                detected = True
                break
        if detected == expect_hit:
            hits += 1
        status = "OK" if detected == expect_hit else "MISS"
        print(f"  [{status}] {wav.name} detected={detected}")
    return hits, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="kws_samples", help="Sample root")
    args = parser.parse_args()
    root = Path(args.root)

    engine = KWSEngine(
        DEFAULT_CONFIG.models,
        DEFAULT_CONFIG.audio,
        wake_word_threshold=DEFAULT_CONFIG.wake_word_threshold,
    )
    if not engine.initialize():
        print("KWS init failed", file=sys.stderr)
        return 1

    print("Positive (expect wake):")
    pos_h, pos_t = _eval_dir(engine, root / "positive", expect_hit=True)
    print("Negative (expect no wake):")
    neg_h, neg_t = _eval_dir(engine, root / "negative", expect_hit=False)

    print("---")
    if pos_t:
        print(f"Positive: {pos_h}/{pos_t} ({100 * pos_h / pos_t:.0f}%)")
    else:
        print("Positive: no samples (add WAV to kws_samples/positive/)")
    if neg_t:
        print(f"Negative: {neg_h}/{neg_t} false-wake allowed ≤2 → {neg_t - neg_h} false wakes")
    else:
        print("Negative: no samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
