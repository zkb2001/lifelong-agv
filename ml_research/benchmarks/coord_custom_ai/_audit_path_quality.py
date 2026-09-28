"""Audit AGV load balance, carry duration, and hotspot parking."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(path: str) -> None:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    by_agv: dict = defaultdict(list)
    for r in rows:
        by_agv[r["name"]].append(r)

    sim_t = max(int(r["timestamp"]) for r in rows)
    print(f"sim_t={sim_t}")

    turns = moves = waits = move_and_turn = 0
    for _name, seq in by_agv.items():
        seq = sorted(seq, key=lambda r: int(r["timestamp"]))
        for a, b in zip(seq, seq[1:]):
            ax, ay = int(a["X"]), int(a["Y"])
            bx, by = int(b["X"]), int(b["Y"])
            pa, pb = int(a["pitch"]) % 360, int(b["pitch"]) % 360
            dist = abs(ax - bx) + abs(ay - by)
            if dist == 0 and pa == pb:
                waits += 1
            elif dist == 0 and pa != pb:
                turns += 1
            elif dist == 1 and pa == pb:
                moves += 1
            elif dist == 1 and pa != pb:
                move_and_turn += 1
    print(f"turns={turns} moves={moves} waits={waits} move_and_turn={move_and_turn}")

    print("\n== per AGV ==")
    for name in sorted(by_agv):
        seq = sorted(by_agv[name], key=lambda r: int(r["timestamp"]))
        tids = set()
        n_move = n_turn = n_idle = 0
        cells: Counter = Counter()
        for a, b in zip(seq, seq[1:]):
            ax, ay = int(a["X"]), int(a["Y"])
            bx, by = int(b["X"]), int(b["Y"])
            pa, pb = int(a["pitch"]) % 360, int(b["pitch"]) % 360
            dist = abs(ax - bx) + abs(ay - by)
            cells[(ax, ay)] += 1
            if dist == 0 and pa == pb:
                n_idle += 1
            elif dist == 0:
                n_turn += 1
            else:
                n_move += 1
            tid = (a.get("task-id") or "").strip()
            if tid and tid.lower() not in ("", "none", "null"):
                tids.add(tid)
        tot = max(1, n_move + n_turn + n_idle)
        hot_cell, hot_n = cells.most_common(1)[0]
        print(
            f"{name}: tasks={len(tids)} move={n_move} turn={n_turn} "
            f"idle={n_idle} idle%={100 * n_idle / tot:.0f} "
            f"hot={hot_cell}x{hot_n}"
        )

    print("\n== carry durations ==")
    carries = []
    for name, seq in by_agv.items():
        seq = sorted(seq, key=lambda r: int(r["timestamp"]))
        i = 0
        while i < len(seq):
            if str(seq[i].get("loaded", "")).lower() not in ("1", "true", "yes"):
                i += 1
                continue
            j = i
            while j + 1 < len(seq) and str(seq[j + 1].get("loaded", "")).lower() in (
                "1",
                "true",
                "yes",
            ):
                j += 1
            dur = int(seq[j]["timestamp"]) - int(seq[i]["timestamp"])
            tid = seq[i].get("task-id")
            carries.append((name, tid, dur))
            i = j + 1
    carries.sort(key=lambda x: x[2])
    if carries:
        durs = [c[2] for c in carries]
        print(
            f"n={len(carries)} min={min(durs)} max={max(durs)} "
            f"avg={sum(durs)/len(durs):.1f} uniq={len(set(durs))}"
        )
        for c in carries:
            print(f"  {c[0]} {c[1]} carry={c[2]}s")


if __name__ == "__main__":
    main(sys.argv[1])
