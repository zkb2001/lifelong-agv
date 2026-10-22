"""Compare SH03 vs SH12 vs SH17: topology + end-state deadlock mode."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from greed_coop_mcts.scenario import load_slot_meta

ROOT = Path(__file__).resolve().parent
TRAJ = ROOT / "results" / "trajectories"
NEI = ((1, 0), (-1, 0), (0, 1), (0, -1))


def load_grid(meta: dict):
    data = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
    gmin = data["grid_min"]
    gmax = data["grid_max"]
    if isinstance(gmin, (list, tuple)):
        x0, y0 = int(gmin[0]), int(gmin[1])
        x1, y1 = int(gmax[0]), int(gmax[1])
    else:
        x0 = y0 = int(gmin)
        x1 = y1 = int(gmax)
    blocked = set()
    for o in data.get("extra_obstacles") or []:
        if isinstance(o, dict):
            blocked.add((int(o["x"]), int(o["y"])))
        else:
            blocked.add((int(o[0]), int(o[1])))
    free = {
        (x, y)
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
        if (x, y) not in blocked
    }
    deg = {}
    for c in free:
        deg[c] = sum((c[0] + dx, c[1] + dy) in free for dx, dy in NEI)
    dropoffs = {}
    for d in data.get("dropoffs") or []:
        if isinstance(d, dict):
            dropoffs[d.get("name", f"{d['x']},{d['y']}")] = (int(d["x"]), int(d["y"]))
        else:
            dropoffs[str(d)] = tuple(d)
    # also from position csv
    with open(meta["position_csv"], encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("type") in ("end_point", "dropoff", "destination"):
                dropoffs[r["name"]] = (int(float(r["x"])), int(float(r["y"])))
    with open(meta["task_csv"], encoding="utf-8") as f:
        tasks = list(csv.DictReader(f))
    dest_freq = Counter(r["end_point"] for r in tasks)
    return {
        "free": free,
        "deg": deg,
        "dropoffs": dropoffs,
        "n_agvs": int(data.get("n_agvs") or meta.get("n_agvs") or 0),
        "w": x1 - x0 + 1,
        "h": y1 - y0 + 1,
        "n_free": len(free),
        "corridor_cells": sum(1 for c, d in deg.items() if d <= 2),
        "dest_freq": dest_freq,
        "data": data,
    }


def pad_corridor_depth(pad, free, deg, max_d=12):
    """BFS along deg<=2 from pad until deg>=3; return depth and whether dead-end."""
    if pad not in free:
        return {"depth": 0, "dead_end": True, "exits": 0}
    # grow corridor component attached to pad
    q = [pad]
    seen = {pad}
    junctions = []
    while q:
        c = q.pop(0)
        if deg.get(c, 0) >= 3 and c != pad:
            junctions.append(c)
            continue
        for dx, dy in NEI:
            nxt = (c[0] + dx, c[1] + dy)
            if nxt in free and nxt not in seen:
                if deg.get(nxt, 0) <= 2 or nxt == pad:
                    seen.add(nxt)
                    q.append(nxt)
                elif deg.get(nxt, 0) >= 3:
                    junctions.append(nxt)
                    seen.add(nxt)
    # depth = max manh in corridor component to pad among deg<=2
    corridor = [c for c in seen if deg.get(c, 0) <= 2]
    depth = max((_manh(c, pad) for c in corridor), default=0)
    return {
        "depth": depth,
        "n_corridor": len(corridor),
        "exits": len(set(junctions)),
        "dead_end": len(set(junctions)) <= 1,
        "pad_deg": deg.get(pad, 0),
    }


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def analyze(slot: int, last_n: int = 3000) -> None:
    meta = load_slot_meta(slot)
    assert meta
    g = load_grid(meta)
    traj = TRAJ / f"SH{slot:02d}_gcm_gcm.csv"
    with traj.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    times = sorted({int(float(r["timestamp"])) for r in rows})
    tmax = max(times)
    t0 = max(0, tmax - last_n)
    tail = [r for r in rows if int(float(r["timestamp"])) >= t0]
    agents = sorted({r["name"] for r in rows})
    dens = len(agents) / max(1, g["n_free"])

    print(f"\n======== SH{slot:02d} ========")
    print(
        f"n_agv={len(agents)} map={g['w']}x{g['h']} free={g['n_free']} "
        f"corr_cells(deg<=2)={g['corridor_cells']} "
        f"corr_ratio={g['corridor_cells']/max(1,g['n_free']):.3f} dens={dens:.4f}"
    )
    print(f"dest_top={g['dest_freq'].most_common(6)}")

    # top destinations: corridor geometry
    for dest, cnt in g["dest_freq"].most_common(5):
        pad = g["dropoffs"].get(dest)
        if not pad:
            print(f"  dest={dest} cnt={cnt} pad=MISSING")
            continue
        info = pad_corridor_depth(pad, g["free"], g["deg"])
        print(f"  dest={dest} cnt={cnt} pad={pad} {info}")

    end = [r for r in rows if int(float(r["timestamp"])) == tmax]
    loaded = [r for r in end if str(r["loaded"]).upper() == "TRUE"]
    idle = [r for r in end if str(r["loaded"]).upper() != "TRUE"]
    print(f"tmax={tmax} end_loaded={len(loaded)} end_idle={len(idle)}")
    print(
        "  loaded:",
        [
            (r["name"], (int(r["X"]), int(r["Y"])), r["destination"], r["task-id"])
            for r in loaded
        ],
    )
    print(
        "  idle:",
        [(r["name"], (int(r["X"]), int(r["Y"])), r["destination"] or "-") for r in idle],
    )

    coords = {r["name"]: (int(r["X"]), int(r["Y"])) for r in end}
    # adjacency
    neigh = []
    names = list(coords)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            xa, ya = coords[a]
            xb, yb = coords[b]
            if abs(xa - xb) + abs(ya - yb) == 1:
                la = next(r for r in end if r["name"] == a)
                lb = next(r for r in end if r["name"] == b)
                neigh.append(
                    (
                        a,
                        str(la["loaded"]).upper() == "TRUE",
                        la["destination"],
                        (xa, ya),
                        b,
                        str(lb["loaded"]).upper() == "TRUE",
                        lb["destination"],
                        (xb, yb),
                    )
                )
    print(f"  adjacent_pairs n={len(neigh)}")
    for p in neigh:
        print(f"    {p}")

    # sticky adjacency in tail
    by_t: dict[int, dict[str, tuple]] = defaultdict(dict)
    load_t: dict[int, dict[str, bool]] = defaultdict(dict)
    dest_t: dict[int, dict[str, str]] = defaultdict(dict)
    for r in tail:
        t = int(float(r["timestamp"]))
        by_t[t][r["name"]] = (int(r["X"]), int(r["Y"]))
        load_t[t][r["name"]] = str(r["loaded"]).upper() == "TRUE"
        dest_t[t][r["name"]] = r.get("destination") or ""
    adj_count = Counter()
    mixed_count = Counter()  # idle-loaded sticky pairs
    for t, cmap in by_t.items():
        ags = list(cmap)
        for i in range(len(ags)):
            for j in range(i + 1, len(ags)):
                a, b = ags[i], ags[j]
                xa, ya = cmap[a]
                xb, yb = cmap[b]
                if abs(xa - xb) + abs(ya - yb) == 1:
                    pair = tuple(sorted((a, b)))
                    adj_count[pair] += 1
                    la, lb = load_t[t][a], load_t[t][b]
                    if la != lb:
                        mixed_count[pair] += 1
    print(f"  sticky_adj top8: {adj_count.most_common(8)}")
    print(f"  sticky_idle<->loaded top8: {mixed_count.most_common(8)}")

    # unmoved + unique cells
    first = {}
    moved = set()
    cells = defaultdict(set)
    for r in sorted(tail, key=lambda rr: int(float(rr["timestamp"]))):
        a = r["name"]
        p = (int(r["X"]), int(r["Y"]))
        cells[a].add(p)
        if a not in first:
            first[a] = p
        elif p != first[a]:
            moved.add(a)
    stuck = [a for a in agents if a not in moved]
    print(f"  unmoved_last_{last_n}: {stuck}")
    print(
        "  unique_cells smallest:",
        sorted(((a, len(s)) for a, s in cells.items()), key=lambda x: x[1])[:10],
    )

    # sample same-dest loaded contention
    step = max(1, (tmax - t0) // 5)
    for t in range(t0, tmax + 1, step):
        snap = [r for r in rows if int(float(r["timestamp"])) == t]
        if not snap:
            continue
        carr = [r for r in snap if str(r["loaded"]).upper() == "TRUE"]
        dc = Counter(r["destination"] for r in carr if r["destination"])
        # how many idle sit on deg<=2 near any loaded dest pad
        idle_on_corr = 0
        for r in snap:
            if str(r["loaded"]).upper() == "TRUE":
                continue
            cell = (int(r["X"]), int(r["Y"]))
            if g["deg"].get(cell, 99) <= 2:
                idle_on_corr += 1
        print(
            f"  t={t} loaded={len(carr)} top_dest={dc.most_common(3)} "
            f"idle_on_deg<=2={idle_on_corr}"
        )


def main():
    for s in (3, 12, 17):
        analyze(s)


if __name__ == "__main__":
    main()
