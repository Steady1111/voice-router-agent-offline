#!/usr/bin/env python3
"""Apply router read-through ASR alignment and export calibration artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_router_lite.web.service import WebCommandService

# Full-file ASR from 路由器47条录音.m4a (ordered read-through)
FULL_TRANSCRIPT = (
    "重启路由器重启路由崇喜网络网络卡了重启一下落又期袭击了重启一下重启歪饭打开歪歪关闭外发也连不上重启一下"
    "麻烦密码多少改外犯密码外犯改革名字打开房客网络关闭访客网络切换到五级频段指示多少十一连的外犯连了几个设备"
    "网速怎么样略有期运行多久了内存上多少皮油附载多少关闭录系灯打开洛器灯提掉这个设备拉黑这部手机取消拉黑这个设备"
    "查看黑名单测一下网速聘一下延迟多少网络诊断电台是正常吗检查顾健更新配置恢复出厂设置查看系统日制设定定时重启"
    "打开课玩死关闭科外这台电脑的网速给游戏机优先打开防火墙关闭防火墙打开奖场控制打开微偏方寸网"
)

# Hand-aligned raw ASR chunks (seq -> raw)
RAW_BY_SEQ = {
    1: "重启路由器",
    2: "重启路由",
    3: "崇喜网络",
    4: "网络卡了重启一下",
    5: "落又期袭击了重启一下",
    6: "重启歪饭",
    7: "打开歪歪",
    8: "关闭外发",
    9: "也连不上重启一下",
    10: "麻烦密码多少",
    11: "改外犯密码",
    12: "外犯改革名字",
    13: "打开房客网络",
    14: "关闭访客网络",
    15: "切换到五级频段",
    16: "指示多少",
    17: "十一连的外犯",
    18: "连了几个设备",
    19: "网速怎么样",
    20: "略有期运行多久了",
    21: "内存上多少",
    22: "皮油附载多少",
    23: "关闭录系灯",
    24: "打开洛器灯",
    25: "提掉这个设备",
    26: "拉黑这部手机",
    27: "取消拉黑这个设备",
    28: "查看黑名单",
    29: "测一下网速",
    30: "聘一下",
    31: "延迟多少",
    32: "网络诊断",
    33: "电台是正常吗",
    34: "检查顾健更新",
    35: "配置",
    36: "恢复出厂设置",
    37: "查看系统日制",
    38: "设定定时重启",
    39: "打开课玩死",
    40: "关闭科外",
    41: "这台电脑的网速",
    42: "给游戏机优先",
    43: "打开防火墙",
    44: "关闭防火墙",
    45: "打开奖场控制",
    46: "打开微偏方",
    47: "寸网",
}


def main() -> int:
    template = ROOT / "asr_debug/router_voice_readthrough.template.csv"
    rows = list(csv.DictReader(template.open(encoding="utf-8")))
    svc = WebCommandService()
    svc.initialize()

    misheard: list[dict] = []
    intent_ok = {
        ("router_wifi_config", "router_network_query"): {"WiFi密码多少"},
        ("router_network_diag", "router_network_query"): {"测一下网速", "网速怎么样"},
    }
    for row in rows:
        seq = int(row["seq"])
        raw = RAW_BY_SEQ.get(seq, "")
        exp = row["expected_text"]
        row["raw_asr"] = raw
        handled = svc.handle_text(raw) if raw else {"success": False, "intent": "unknown"}
        repaired = handled.get("transcript") or handled.get("command") or raw
        intent = handled.get("intent", "unknown")
        ok_intent = intent == row["intent"]
        if not ok_intent:
            for (a, b), texts in intent_ok.items():
                if exp in texts and {intent, row["intent"]} == {a, b}:
                    ok_intent = True
                    break
        ok = bool(handled.get("success")) and ok_intent
        row["repaired_text"] = repaired
        row["nlu_intent"] = intent
        row["success"] = "yes" if ok else "no"
        if raw and raw != exp:
            misheard.append({
                "asr_text": raw,
                "target_text": exp,
                "intent": row["intent"],
                "seq": seq,
            })

    out_csv = ROOT / "asr_debug/router_voice_readthrough.csv"
    fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    out_json = ROOT / "asr_debug/router_readthrough/misheard_pairs.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(misheard, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_n = sum(1 for r in rows if r["success"] == "yes")
    print(f"aligned={len(rows)} ok_after_current_rules={ok_n} misheard={len(misheard)}")
    for r in rows:
        if r["success"] == "no":
            print(f"  {r['seq']:2} {r['expected_text']!r} raw={r['raw_asr']!r} -> {r['nlu_intent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
