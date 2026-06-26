#!/usr/bin/env python3
"""Build NLU training augmentations from asr_debug/ misheard text (task 2A.1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./asr_debug")
    parser.add_argument("--out", default="./asr_debug/misheard_training.json")
    args = parser.parse_args()

    root = Path(args.dir)
    entries: list[dict] = []
    for path in sorted(root.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        raw = (meta.get("raw_asr") or "").strip()
        command = (meta.get("command") or meta.get("repaired") or "").strip()
        intent = meta.get("intent") or ""
        if not raw or not command or raw == command:
            continue
        if not meta.get("success") and not intent:
            continue
        entries.append({
            "asr_text": raw,
            "target_text": command,
            "intent": intent,
            "source": path.name,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} misheard pairs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
