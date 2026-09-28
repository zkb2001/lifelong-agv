"""Inspect Smokescreen Tiger-8 → Tiger-15 transition and Tiger-15 end."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
r = next(x for x in rows if x.get("method_id") == "ecbs" and int(x["slot"]) == 11)
df = pd.read_csv(r["trajectory"], low_memory=False)
sm = df[df["name"] == "Smokescreen"].sort_values("timestamp")
win = sm[(sm["timestamp"] >= 12520) & (sm["timestamp"] <= 12620)]
print(win[["timestamp", "X", "Y", "loaded", "destination", "task-id"]].to_string(index=False))
print("--- around Tiger-15 end ---")
win2 = sm[(sm["timestamp"] >= 12590) & (sm["timestamp"] <= 12660)]
print(win2[["timestamp", "X", "Y", "loaded", "destination", "task-id"]].to_string(index=False))
