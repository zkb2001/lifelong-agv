"""Under y=1 + north obstacle: which way do AGVs step? prod vs gen."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPORTS = Path(__file__).resolve().parents[2] / "ml_research" / "map_editor" / "exports"
TRAJ = Path(__file__).resolve().parent / "trajectories"


def obs(slot: int):
    j = json.loads((EXPORTS / f"SH_custom_{slot:02d}.json").read_text(encoding="utf-8"))
    out = set()
    for o in j["extra_obstacles"]:
        out.add(tuple(o) if not isinstance(o, dict) else (int(o["x"]), int(o["y"])))
    return out


def analyze(slot: int, path: Path) -> None:
    o = obs(slot)
    ag: dict[str, list] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ag[r["name"]].append(r)
    moves: Counter = Counter()
    for seq in ag.values():
        seq.sort(key=lambda r: int(r["timestamp"]))
        for a, b in zip(seq, seq[1:]):
            xa, ya = int(a["X"]), int(a["Y"])
            xb, yb = int(b["X"]), int(b["Y"])
            if ya != 1 or (xa, 2) not in o:
                continue
            moves[(xb - xa, yb - ya)] += 1
    south = moves.get((0, -1), 0)
    print(f"SH{slot:02d} {path.name}: under N-obs@y1 moves={dict(moves)} south_to_y0={south}")


def main():
    for slot, name in [
        (12, "SH12_gcm_gcm.csv"),
        (3, "SH03_gcm_gcm.csv"),
        (2, "SH02_gcm_gcm.csv"),
        (7, "SH07_gcm_gcm.csv"),
        (23, "SH23_gen_gcm.csv"),
        (25, "SH25_gen_gcm.csv"),
        (28, "SH28_gen_gcm.csv"),
        (39, "SH39_gen_gcm.csv"),
    ]:
        p = TRAJ / name
        if p.exists():
            analyze(slot, p)
        else:
            print(f"missing {p}")


if __name__ == "__main__":
    main()
