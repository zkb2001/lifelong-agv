"""Trace one OOB step: what did rule/local propose around SH23 t=687."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from greed_coop_mcts.rule_obs import GRID, DIR_DELTA
from greed_coop_mcts.pgcp import _neighbors

ROOT = Path(__file__).resolve().parent
EXPORTS = Path(__file__).resolve().parents[2] / "ml_research" / "map_editor" / "exports"


def main():
    slot = 23
    j = json.loads((EXPORTS / f"SH_custom_{slot:02d}.json").read_text(encoding="utf-8"))
    obs = set()
    for o in j["extra_obstacles"]:
        obs.add(tuple(o) if not isinstance(o, dict) else (o["x"], o["y"]))
    # stations are NOT walls in planner? usually blocked for pathing of others
    pick = {(p["x"], p["y"]) for p in j["pickups"]}
    drop = {(p["x"], p["y"]) for p in j["dropoffs"]}
    blocked = set(obs)  # same as free-space check in map_gen

    print(f"GRID={GRID}  planner allows coords 0..{GRID-1}")
    print(f"validator allows 1..20")
    print(f"ghost cells walkable under planner: y=0 or x=0")

    cell = (10, 1)
    nbs = _neighbors(cell, blocked)
    print(f"pgcp._neighbors({cell}, obs-only) = {nbs}")
    # with stations as blocked?
    blocked2 = blocked | pick | drop
    print(f"pgcp._neighbors({cell}, obs|stations) = {_neighbors(cell, blocked2)}")

    # Is (10,0) in blocked? 
    print(f"(10,0) in obs? {(10,0) in obs}")
    print(f"(10,2) in obs? {(10,2) in obs}")

    # show corridor: which y=2 cells blocked
    print("obs on y=2:", sorted(c for c in obs if c[1] == 2))

    # trajectory window around first OOB
    traj = ROOT / "trajectories" / "SH23_gen_gcm.csv"
    rows = [r for r in csv.DictReader(traj.open(encoding="utf-8")) if r["name"] == "Bumblebee"]
    rows.sort(key=lambda r: int(r["timestamp"]))
    for r in rows:
        t = int(r["timestamp"])
        if 680 <= t <= 695:
            print(
                f"t={t} Bumblebee ({r['X']},{r['Y']}) pitch={r.get('pitch')} "
                f"loaded={r.get('loaded')} task={r.get('task_id') or r.get('task')}"
            )

    # Compare SH21: is (10,2) free?
    j21 = json.loads((EXPORTS / "SH_custom_21.json").read_text(encoding="utf-8"))
    obs21 = {tuple(o) for o in j21["extra_obstacles"]}
    print("SH21 obs y=2:", sorted(c for c in obs21 if c[1] == 2))
    print("SH21 _neighbors((10,1)):", _neighbors((10, 1), obs21))


if __name__ == "__main__":
    main()
