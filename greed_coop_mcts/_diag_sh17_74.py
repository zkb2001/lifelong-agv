"""SH17 endgame freeze at ~74 done."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from greed_coop_mcts.scenario import load_slot_meta

NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main() -> None:
    meta = load_slot_meta(17)
    assert meta
    d = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
    blocked = set()
    for o in d["extra_obstacles"]:
        if isinstance(o, dict):
            blocked.add((int(o["x"]), int(o["y"])))
        else:
            blocked.add((int(o[0]), int(o[1])))
    free = {
        (x, y) for x in range(1, 21) for y in range(1, 21) if (x, y) not in blocked
    }
    deg = {
        c: sum((c[0] + dx, c[1] + dy) in free for dx, dy in NEI) for c in free
    }
    drop = {dd["name"]: (dd["x"], dd["y"]) for dd in d["dropoffs"]}

    def pads(name: str):
        c = drop[name]
        return [
            (c[0] + dx, c[1] + dy)
            for dx, dy in NEI
            if (c[0] + dx, c[1] + dy) in free
        ]

    rows = list(
        csv.DictReader(
            open(
                "greed_coop_mcts/results/trajectories/SH17_gcm_gcm.csv",
                encoding="utf-8",
            )
        )
    )
    by: dict[int, list] = defaultdict(list)
    for r in rows:
        by[int(float(r["timestamp"]))].append(r)

    for t in (4000, 6000, 8000, 10000):
        snap = by[t]
        loaded = [r for r in snap if str(r["loaded"]).upper() == "TRUE"]
        idle = [r for r in snap if str(r["loaded"]).upper() != "TRUE"]
        print(f"=== t={t} L={len(loaded)} I={len(idle)} ===")
        for r in loaded:
            cell = (int(r["X"]), int(r["Y"]))
            dest = r["destination"]
            ps = pads(dest)
            mh = min(abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in ps)
            print(
                f"  C {r['name']:12} @{cell} d={deg.get(cell)} "
                f"->{dest} manh={mh} task={r['task-id']}"
            )
        for r in idle:
            cell = (int(r["X"]), int(r["Y"]))
            print(f"  I {r['name']:12} @{cell} d={deg.get(cell)}")

    adj: Counter = Counter()
    mixed: Counter = Counter()
    cells: dict[str, set] = defaultdict(set)
    for t, snap in by.items():
        if t < 6000:
            continue
        coords = {r["name"]: (int(r["X"]), int(r["Y"])) for r in snap}
        lm = {r["name"]: str(r["loaded"]).upper() == "TRUE" for r in snap}
        for r in snap:
            cells[r["name"]].add((int(r["X"]), int(r["Y"])))
        ns = list(coords)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a, b = ns[i], ns[j]
                xa, ya = coords[a]
                xb, yb = coords[b]
                if abs(xa - xb) + abs(ya - yb) == 1:
                    p = tuple(sorted((a, b)))
                    adj[p] += 1
                    if lm[a] != lm[b]:
                        mixed[p] += 1
    print("sticky", adj.most_common(8))
    print("mixed", mixed.most_common(8))
    print(
        "uniq",
        sorted((a, len(s), sorted(s)) for a, s in cells.items()),
    )
    jd = json.loads(
        Path("greed_coop_mcts/results/SH17_gcm_gcm.json").read_text(encoding="utf-8")
    )
    print(
        "done",
        jd["tasks_completed"],
        "egress",
        jd.get("n_pad_egress"),
        "park",
        jd.get("n_park_freezes"),
        "H",
        jd.get("avg_committed_horizon"),
    )


if __name__ == "__main__":
    main()
