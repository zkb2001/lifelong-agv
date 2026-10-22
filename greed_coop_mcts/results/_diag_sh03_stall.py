"""One-shot diagnosis of SH03 live stall."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "greed_coop_mcts" / "results"
EXP = ROOT / "ml_research" / "map_editor" / "exports"

live = json.loads((RES / "SH_custom_03_live.json").read_text(encoding="utf-8"))
mj = json.loads((EXP / "SH_custom_03.json").read_text(encoding="utf-8"))
obs = {
    (int(o[0]), int(o[1]))
    for o in (mj.get("extra_obstacles") or [])
    if isinstance(o, (list, tuple)) and len(o) >= 2
}
picks = {p["name"]: (int(p["x"]), int(p["y"])) for p in mj.get("pickups") or []}
drops = {p["name"]: (int(p["x"]), int(p["y"])) for p in mj.get("dropoffs") or []}

print("=== LIVE SNAPSHOT ===")
for k in [
    "t",
    "done",
    "total",
    "initial_total",
    "stream_on",
    "carry",
    "approach",
    "pending",
    "k",
    "facn_delta",
    "facn_k_tbc",
    "stall",
    "wall_s",
]:
    print(f"  {k}: {live.get(k)}")

agvs = live.get("agvs") or []
print("\n=== AGV STATE ===")
busy = []
for a in agvs:
    xy = (a["x"], a["y"])
    role = "carry" if a["loaded"] else ("approach" if a["approaching"] else "idle")
    near_p = [
        n
        for n, c in picks.items()
        if abs(c[0] - xy[0]) + abs(c[1] - xy[1]) <= 2
    ]
    near_d = [
        n
        for n, c in drops.items()
        if abs(c[0] - xy[0]) + abs(c[1] - xy[1]) <= 2
    ]
    print(
        f"  {a['name']:12} {role:8} xy={xy} pitch={a['pitch']:3} "
        f"on_obs={xy in obs} dest={a.get('dest')!r} tid={a.get('task_id')!r} "
        f"nearP={near_p} nearD={near_d}"
    )
    if role != "idle":
        busy.append(a)

print("\n=== CLOSE PAIRS ===")
for i, a in enumerate(agvs):
    for b in agvs[i + 1 :]:
        d = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
        if d <= 3:
            print(
                f"  {a['name']}-{b['name']} manh={d} "
                f"({a['x']},{a['y']})/({b['x']},{b['y']}) "
                f"L={a['loaded']}/{b['loaded']} A={a['approaching']}/{b['approaching']}"
            )

# corridor occupancy: cells with degree-ish from free neighbors
print("\n=== PICKUP / DROPOFF GEOMETRY ===")
for n, (x, y) in sorted(picks.items()):
    nbrs = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    free = [c for c in nbrs if c not in obs and 1 <= c[0] <= 20 and 1 <= c[1] <= 20]
    print(f"  pickup {n}@{x},{y} free_nbrs={len(free)} {free}")

# seed task OD balance
seed = RES / "SH03_lifelong_seed400_task.csv"
od = Counter()
src = Counter()
dst = Counter()
with seed.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        s, e = row["start_point"], row["end_point"]
        od[(s, e)] += 1
        src[s] += 1
        dst[e] += 1
print("\n=== SEED TASK SKEW (top) ===")
print("  by start:", src.most_common())
print("  by end top8:", dst.most_common(8))

# viz: last few seconds of busy AGV motion — are they oscillating?
viz = RES / "trajectories" / "SH_custom_03_viz.csv"
if viz.exists():
    rows = list(csv.DictReader(viz.open(encoding="utf-8")))
    by_t: dict[int, list] = defaultdict(list)
    for r in rows:
        try:
            t = int(r["timestamp"])
        except ValueError:
            continue
        by_t[t].append(r)
    times = sorted(by_t)
    if len(times) >= 40:
        window = times[-40:]
        print(f"\n=== VIZ LAST {len(window)}s MOTION (busy names) ===")
        busy_names = {a["name"] for a in busy} or {a["name"] for a in agvs}
        for name in sorted(busy_names):
            path = []
            for t in window:
                for r in by_t[t]:
                    if r["name"] == name:
                        path.append((int(r["X"]), int(r["Y"]), int(float(r.get("pitch") or 0))))
                        break
            uniq = []
            for p in path:
                if not uniq or uniq[-1][:2] != p[:2]:
                    uniq.append(p)
            # oscillation score: revisit rate
            cells = [p[:2] for p in path]
            revisit = len(cells) - len(set(cells))
            print(
                f"  {name}: unique_cells={len(set(cells))}/{len(cells)} "
                f"revisits={revisit} path_tail={uniq[-8:]}"
            )

# free-space bottleneck: how narrow is SH03
free = {(x, y) for x in range(1, 21) for y in range(1, 21) if (x, y) not in obs}
print(f"\n=== FREE SPACE === free={len(free)}/400 obs_in={sum(1 for o in obs if 1<=o[0]<=20 and 1<=o[1]<=20)}")
