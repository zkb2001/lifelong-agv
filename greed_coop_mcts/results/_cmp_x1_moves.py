"""At x=1: do AGVs ever step west to x=0? prod vs gen."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

TRAJ = Path(__file__).resolve().parent / "trajectories"


def analyze(slot: int, path: Path) -> None:
    ag: dict[str, list] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ag[r["name"]].append(r)
    moves: Counter = Counter()
    west = 0
    for seq in ag.values():
        seq.sort(key=lambda r: int(r["timestamp"]))
        for a, b in zip(seq, seq[1:]):
            xa, ya = int(a["X"]), int(a["Y"])
            xb, yb = int(b["X"]), int(b["Y"])
            if xa != 1:
                continue
            moves[(xb - xa, yb - ya)] += 1
            if xb == 0:
                west += 1
    print(f"SH{slot:02d}: from x=1 moves={dict(moves)} to_x0={west}")


def main():
    for slot, name in [
        (1, "SH01_gcm_gcm.csv"),
        (12, "SH12_gcm_gcm.csv"),
        (17, "SH17_gcm_gcm.csv"),
        (26, "SH26_gen_gcm.csv"),
        (29, "SH29_gen_gcm.csv"),
        (31, "SH31_gen_gcm.csv"),
    ]:
        p = TRAJ / name
        if p.exists():
            analyze(slot, p)


if __name__ == "__main__":
    main()
