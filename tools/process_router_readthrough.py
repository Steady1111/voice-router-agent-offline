#!/usr/bin/env python3
"""Split ordered router command read-through audio, ASR each clip, export calibration CSV."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_router_lite.asr import ASREngine
from voice_router_lite.config import DEFAULT_CONFIG
from voice_router_lite.web.service import WebCommandService


def load_template(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("expected_text"):
                rows.append(row)
    return rows


def convert_to_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", "16000", "-ac", "1", "-f", "wav", str(dst),
        ],
        check=True,
        capture_output=True,
    )


def load_pcm16(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16)


def split_by_silence(
    audio: np.ndarray,
    *,
    frame_ms: int = 20,
    energy_threshold: float = 0.012,
    min_speech_ms: int = 350,
    min_silence_ms: int = 180,
    pad_ms: int = 120,
) -> list[np.ndarray]:
    sr = 16000
    frame = int(sr * frame_ms / 1000)
    min_speech = int(min_speech_ms / frame_ms)
    min_silence = int(min_silence_ms / frame_ms)
    pad = int(pad_ms / 1000 * sr)

    segments: list[tuple[int, int]] = []
    in_speech = False
    start = 0
    silence_run = 0
    speech_run = 0

    for i in range(0, len(audio) - frame, frame):
        chunk = audio[i : i + frame].astype(np.float32) / 32768.0
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        if energy >= energy_threshold:
            if not in_speech:
                start = max(0, i - pad)
                in_speech = True
            speech_run += 1
            silence_run = 0
        elif in_speech:
            silence_run += 1
            if silence_run >= min_silence and speech_run >= min_speech:
                end = min(len(audio), i + pad)
                segments.append((start, end))
                in_speech = False
                speech_run = 0
                silence_run = 0

    if in_speech and speech_run >= min_speech:
        segments.append((start, len(audio)))

    return [audio[s:e].copy() for s, e in segments if e - s > sr // 5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--template", default=str(ROOT / "asr_debug/router_voice_readthrough.template.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "asr_debug/router_readthrough"))
    parser.add_argument("--out-csv", default=str(ROOT / "asr_debug/router_voice_readthrough.csv"))
    args = parser.parse_args()

    template = load_template(Path(args.template))
    out_dir = Path(args.out_dir)
    wav_path = out_dir / "full.wav"
    convert_to_wav(args.audio, wav_path)
    audio = load_pcm16(wav_path)
    clips = split_by_silence(audio)

    print(f"Audio: {len(audio)/16000:.1f}s, segments={len(clips)}, expected={len(template)}")

    asr = ASREngine(DEFAULT_CONFIG.models, DEFAULT_CONFIG.audio, num_threads=2)
    if not asr.initialize():
        print("ASR init failed", file=sys.stderr)
        return 1

    svc = WebCommandService()
    svc.initialize()

    rows_out: list[dict[str, str]] = []
    pairs: list[dict] = []

    for i, expected in enumerate(template):
        seq = expected["seq"]
        if i >= len(clips):
            raw = ""
            note = "missing_segment"
        else:
            clip = clips[i]
            clip_path = out_dir / f"cmd_{int(seq):02d}.wav"
            import wave

            with wave.open(str(clip_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(clip.tobytes())

            result = asr.transcribe(clip)
            raw = result.text.strip()
            note = ""

        handled = svc.handle_text(raw) if raw else {"success": False, "intent": "unknown", "transcript": ""}
        repaired = handled.get("transcript") or handled.get("command") or raw
        ok = (
            handled.get("success")
            and handled.get("intent") == expected["intent"]
            and (repaired == expected["expected_text"] or raw == expected["expected_text"])
        )
        row = {
            **expected,
            "raw_asr": raw,
            "repaired_text": repaired,
            "nlu_intent": handled.get("intent", ""),
            "success": "yes" if ok else "no",
            "notes": note,
        }
        rows_out.append(row)
        if raw and raw != expected["expected_text"]:
            pairs.append({
                "asr_text": raw,
                "target_text": expected["expected_text"],
                "intent": expected["intent"],
                "seq": seq,
            })
        print(f"{seq:2s} exp={expected['expected_text']!r} raw={raw!r} intent={handled.get('intent')} ok={ok}")

    out_csv = Path(args.out_csv)
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    pairs_path = out_dir / "misheard_pairs.json"
    pairs_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_csv} ({len(rows_out)} rows, {sum(1 for r in rows_out if r['success']=='yes')} ok)")
    print(f"Misheard pairs: {len(pairs)} → {pairs_path}")
    if len(clips) != len(template):
        print(f"WARNING: segment count {len(clips)} != template {len(template)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
