"""Inspect traj rows around premature_unload events."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
EVENTS = {
    11: [("Smokescreen", 12527, "Tiger-8")],
    12: [("Sideswipe", 14019, "Rabbit-12"), ("Wheeljack", 14019, "Ox-17")],
}

rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
for slot, evs in EVENTS.items():
    r = next(x for x in rows if x.get("method_id") == "ecbs" and int(x["slot"]) == slot)
    df = pd.read_csv(r["trajectory"], low_memory=False)
    print(f"\n======== SH{slot:02d} max_t={df['timestamp'].max()} ========")
    for agv, t, tid in evs:
        sub = df[df["name"] == agv].sort_values("timestamp")
        win = sub[(sub["timestamp"] >= t - 8) & (sub["timestamp"] <= t + 8)]
        print(f"\n-- {agv} @{t} tid={tid} --")
        print(
            win[
                ["timestamp", "X", "Y", "pitch", "loaded", "destination", "task-id"]
            ].to_string(index=False)
        )
        # first/last with this tid
        carry = sub[sub["task-id"].astype(str) == tid]
        if len(carry):
            print(
                f"carry span t={carry['timestamp'].min()}..{carry['timestamp'].max()} "
                f"n={len(carry)} last_cell=({carry.iloc[-1]['X']},{carry.iloc[-1]['Y']}) "
                f"loaded={carry.iloc[-1]['loaded']}"
            )
