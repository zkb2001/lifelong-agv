"""Diagnose phantom pickups: loaded without standing on start_point."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POS = ROOT / "data" / "agv_position.csv"
TRAJ = ROOT / "ml_research" / "results" / "lifelong" / "live_trajectory.csv"
TASKS = ROOT / "ml_research" / "results" / "lifelong" / "live_tasks.csv"


def main() -> None:
    starts, ends = {}, {}
    with POS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t, n = r["type"].strip(), r["name"].strip()
            x, y = int(r["x"]), int(r["y"])
            if t == "start_point":
                starts[n] = (x, y)
            elif t == "end_point":
                ends[n] = (x, y)

    tasks = {}
    with TASKS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tasks[r["task_id"]] = r

    by_t: dict[int, dict[str, dict]] = defaultdict(dict)
    with TRAJ.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_t[int(r["timestamp"])][r["name"]] = r

    prev: dict[str, dict] = {}
    ok = 0
    anoms = []
    for t in sorted(by_t):
        for name, row in by_t[t].items():
            loaded = str(row.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(row.get("task-id") or "").strip()
            x, y = int(row["X"]), int(row["Y"])
            p = prev.get(name)
            if p is not None:
                pl = str(p.get("loaded", "")).lower() in ("true", "1", "yes")
                if loaded and not pl:
                    info = tasks.get(tid, {})
                    sp = info.get("start_point", "?")
                    sx, sy = starts.get(sp, (None, None))
                    at = (x, y) == (sx, sy) if sx is not None else False
                    if at:
                        ok += 1
                    else:
                        anoms.append(
                            {
                                "t": t,
                                "agv": name,
                                "tid": tid,
                                "start": sp,
                                "pos": (x, y),
                                "pickup": (sx, sy),
                                "dest": info.get("end_point"),
                                "prev_pos": (int(p["X"]), int(p["Y"])),
                                "prev_tid": str(p.get("task-id") or "").strip(),
                            }
                        )
            prev[name] = row

    print(f"ok_pickups={ok} anomalous_pickups={len(anoms)}")
    for a in anoms[:40]:
        print(a)

    # Also: how many catalog tasks would phantom-render if completed set empty?
    print(f"catalog_tasks={len(tasks)} starts={len(starts)}")


if __name__ == "__main__":
    main()
