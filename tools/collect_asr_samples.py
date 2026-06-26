#!/usr/bin/env python3
"""Summarize asr_debug/ samples into CSV report (task 1A.1)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR debug sample report")
    parser.add_argument(
        "--dir",
        default="./asr_debug",
        help="Directory with *.json sidecars",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional CSV output path",
    )
    args = parser.parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        print(f"No samples yet (create {root} via VOICE_ROUTER_SAVE_ASR_WAV=1)")
        return 0

    rows: list[dict] = []
    for path in sorted(root.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta["_file"] = path.name
        rows.append(meta)

    total = len(rows)
    success = sum(1 for r in rows if r.get("success"))
    fan_cmds = sum(
        1 for r in rows
        if "风扇" in (r.get("command") or r.get("repaired") or "")
    )
    wake_kws = sum(1 for r in rows if r.get("wake_source") == "kws")
    wake_energy = sum(1 for r in rows if r.get("wake_source") == "energy_wake")

    print(f"Samples: {total}")
    if total:
        print(f"  Success rate: {success}/{total} ({100 * success / total:.0f}%)")
        print(f"  Fan-related: {fan_cmds}")
        print(f"  Wake kws: {wake_kws}  energy: {wake_energy}")
    else:
        print("  (empty — run with VOICE_ROUTER_SAVE_ASR_WAV=1)")

    if args.out and rows:
        fields = [
            "timestamp", "raw_asr", "repaired", "command", "intent",
            "success", "wake_source", "peak", "duration_sec", "_file",
        ]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
