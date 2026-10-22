"""Diagnose SH17 pad-egress livelock after 34-done freeze."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from greed_coop_mcts.pgcp import cell_degree, pad_deg2_corridor
from greed_coop_mcts.scenario import load_slot_meta

NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))
TRAJ = Path("greed_coop_mcts/results/trajectories/SH17_gcm_gcm.csv")


def load_free(meta):
    d = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
    blocked = set()
    for o in d["extra_obstacles"]:
        if isinstance(o, dict):
            blocked.add((int(o["x"]), int(o["y"])))
        else:
            blocked.add((int(o[0]), int(o[1])))
    free = {(x, y) for x in range(1, 21) for y in range(1, 21) if (x, y) not in blocked}
    deg = {c: sum((c[0] + dx, c[1] + dy) in free for dx, dy in NEI) for c in free}
    dropoffs = {dd["name"]: (dd["x"], dd["y"]) for dd in d["dropoffs"]}
    return blocked, free, deg, dropoffs


def main():
    meta = load_slot_meta(17)
    blocked, free, deg, dropoffs = load_free(meta)
    with TRAJ.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # When did done stop increasing? Infer from loaded task-ids uniqueness over time.
    # Use trajectory: count unique completed via... we don't have done in traj.
    # Freeze from batch log: done=34 from t=1000 onward. Focus t=900..2000 and end.
    by_t = defaultdict(list)
    for r in rows:
        by_t[int(float(r["timestamp"]))].append(r)

    times = sorted(by_t)
    print(f"tmax={times[-1]} nagv={len({r['name'] for r in rows})}")

    # Unload pads = neighbors of dropoff centers
    def unload_pads(name):
        c = dropoffs[name]
        return [
            (c[0] + dx, c[1] + dy)
            for dx, dy in NEI
            if (c[0] + dx, c[1] + dy) in free
        ]

    # Corridor sizes for top dests
    with open(meta["task_csv"], encoding="utf-8") as f:
        tasks = list(csv.DictReader(f))
    dest_freq = Counter(r["end_point"] for r in tasks)
    print("dest_top", dest_freq.most_common(6))
    for dest, cnt in dest_freq.most_common(5):
        pads = unload_pads(dest)
        union = set()
        for p in pads:
            union |= pad_deg2_corridor(p, blocked)
            union.add(p)
        print(
            f"  {dest} center={dropoffs[dest]} pads={pads} "
            f"union_corr_n={len(union)} sample={sorted(union)[:12]}"
        )

    # Snapshot analysis
    for t in [500, 900, 1000, 1200, 2000, 5000, 10000]:
        snap = by_t.get(t)
        if not snap:
            continue
        loaded = [r for r in snap if str(r["loaded"]).upper() == "TRUE"]
        idle = [r for r in snap if str(r["loaded"]).upper() != "TRUE"]
        print(f"\n=== t={t} loaded={len(loaded)} idle={len(idle)} ===")
        for r in loaded:
            cell = (int(r["X"]), int(r["Y"]))
            dest = r["destination"]
            pads = unload_pads(dest) if dest in dropoffs else []
            corr = set()
            for p in pads:
                corr |= pad_deg2_corridor(p, blocked)
                corr.add(p)
            print(
                f"  CARRY {r['name']:12} @{cell} d={deg.get(cell)} "
                f"-> {dest} on_throat={cell in corr} manh_pad="
                f"{min((abs(cell[0]-p[0])+abs(cell[1]-p[1]) for p in pads), default=-1)}"
            )
        for r in idle:
            cell = (int(r["X"]), int(r["Y"]))
            on_any = False
            which = []
            for dest, _ in dest_freq.most_common(8):
                pads = unload_pads(dest)
                corr = set()
                for p in pads:
                    corr |= pad_deg2_corridor(p, blocked)
                if cell in corr and deg.get(cell, 99) <= 2:
                    on_any = True
                    which.append(dest)
            print(
                f"  IDLE  {r['name']:12} @{cell} d={deg.get(cell)} "
                f"on_deg2_throat={on_any} dests={which}"
            )

        # adjacency idle-loaded
        coords = {r["name"]: (int(r["X"]), int(r["Y"])) for r in snap}
        loadm = {r["name"]: str(r["loaded"]).upper() == "TRUE" for r in snap}
        destm = {r["name"]: r["destination"] for r in snap}
        pairs = []
        names = list(coords)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                xa, ya = coords[a]
                xb, yb = coords[b]
                if abs(xa - xb) + abs(ya - yb) == 1 and loadm[a] != loadm[b]:
                    pairs.append((a, loadm[a], destm[a], (xa, ya), b, loadm[b], destm[b], (xb, yb)))
        print(f"  idle<->loaded adj: {pairs}")

    # Sticky pairs after freeze (t>=1000)
    adj = Counter()
    mixed = Counter()
    for t in times:
        if t < 1000:
            continue
        snap = by_t[t]
        coords = {r["name"]: (int(r["X"]), int(r["Y"])) for r in snap}
        loadm = {r["name"]: str(r["loaded"]).upper() == "TRUE" for r in snap}
        names = list(coords)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                xa, ya = coords[a]
                xb, yb = coords[b]
                if abs(xa - xb) + abs(ya - yb) == 1:
                    pair = tuple(sorted((a, b)))
                    adj[pair] += 1
                    if loadm[a] != loadm[b]:
                        mixed[pair] += 1
    print("\nsticky_adj t>=1000 top10", adj.most_common(10))
    print("sticky_idle<->loaded top10", mixed.most_common(10))

    # Unique cells per agent after freeze
    cells = defaultdict(set)
    for t in times:
        if t < 1000:
            continue
        for r in by_t[t]:
            cells[r["name"]].add((int(r["X"]), int(r["Y"])))
    print(
        "unique cells t>=1000",
        sorted(((a, len(s), sorted(s)[:6]) for a, s in cells.items()), key=lambda x: x[1]),
    )


if __name__ == "__main__":
    main()
