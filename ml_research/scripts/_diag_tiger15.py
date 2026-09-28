"""Find Tiger-15 / Tiger-16 pickup events on SH11 ECBS traj."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
r = next(x for x in rows if x.get("method_id") == "ecbs" and int(x["slot"]) == 11)
df = pd.read_csv(r["trajectory"], low_memory=False)

for tid in ("Tiger-15", "Tiger-16", "Tiger-14", "Tiger-8"):
    sub = df[df["task-id"].astype(str) == tid].sort_values("timestamp")
    if sub.empty:
        print(f"{tid}: NEVER in traj")
        continue
    print(
        f"{tid}: t={sub['timestamp'].min()}..{sub['timestamp'].max()} "
        f"n={len(sub)} agvs={sorted(sub['name'].unique())} "
        f"last=({sub.iloc[-1]['X']},{sub.iloc[-1]['Y']}) ld={sub.iloc[-1]['loaded']}"
    )
# rising edges for Tiger station
df2 = df.sort_values(["name", "timestamp"])
print("\nTiger rising edges:")
for name, g in df2.groupby("name"):
    g = g.sort_values("timestamp")
    prev_ld, prev_tid = False, ""
    for _, row in g.iterrows():
        ld = str(row["loaded"]).lower() in ("true", "1", "yes")
        tid = str(row.get("task-id") or "")
        if ld and tid.startswith("Tiger") and (not prev_ld or prev_tid != tid):
            print(f"  rise {tid} agv={name} t={row['timestamp']} cell=({row['X']},{row['Y']})")
        prev_ld, prev_tid = ld, tid
