"""Compare edge geometry: SH01-20 vs SH21-40 — why prod rarely hits ghost cells."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EXPORTS = Path(__file__).resolve().parents[2] / "ml_research" / "map_editor" / "exports"
TRAJ = Path(__file__).resolve().parent / "trajectories"

# Default AGV home columns on y=1 (from generator BASE_AGVS + extras)
HOME_COLS = [3, 6, 9, 12, 15, 18, 2, 5, 8, 11, 14, 17, 4, 7, 10, 13, 16, 19, 1, 20]


def load_obs(slot: int):
    j = json.loads((EXPORTS / f"SH_custom_{slot:02d}.json").read_text(encoding="utf-8"))
    obs = set()
    for o in j.get("extra_obstacles") or []:
        if isinstance(o, dict):
            obs.add((int(o["x"]), int(o["y"])))
        else:
            obs.add((int(o[0]), int(o[1])))
    n = int(j.get("n_agvs") or 0)
    return obs, n, j.get("map_shape") or j.get("notes") or ""


def edge_metrics(obs, n_agvs: int):
    cols = HOME_COLS[: max(n_agvs, 0)]
    # north of home cell blocked → only escape is y=0 if E/W congested
    north_blocked = sum(1 for x in cols if (x, 2) in obs)
    # any obstacle on y=2 in home-band columns 1..20
    y2 = sorted(c for c in obs if c[1] == 2)
    y2_on_home = [c for c in y2 if c[0] in cols]
    # left/right inward choke near pickups (x=2 / x=19)
    x2 = sorted(c for c in obs if c[0] == 2)
    x19 = sorted(c for c in obs if c[0] == 19)
    return {
        "n_obs": len(obs),
        "n_agvs": n_agvs,
        "y2_count": len(y2),
        "y2_on_home_cols": len(y2_on_home),
        "home_north_blocked": north_blocked,
        "home_north_frac": north_blocked / max(len(cols), 1),
        "x2_count": len(x2),
        "x19_count": len(x19),
        "y2_cells": y2,
        "y2_home": y2_on_home,
    }


def traj_oob(path: Path) -> int:
    if not path.exists():
        return -1
    n = 0
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            x, y = int(r["X"]), int(r["Y"])
            if x < 1 or x > 20 or y < 1 or y > 20:
                n += 1
    return n


def traj_y1_with_north_obs(path: Path, obs) -> int:
    """Ticks where AGV sits on y=1 under a y=2 obstacle (pressure situation)."""
    if not path.exists():
        return -1
    n = 0
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            x, y = int(r["X"]), int(r["Y"])
            if y == 1 and (x, 2) in obs:
                n += 1
    return n


def main():
    print(f"{'slot':>4} {'tag':>8} {'obs':>4} {'n':>3} {'y2':>3} {'y2home':>6} "
          f"{'Nblk':>4} {'frac':>5} {'x2':>3} {'oob':>6} {'y1|Nobs':>8}")
    for slot in list(range(1, 21)) + list(range(21, 41)):
        obs, n, style = load_obs(slot)
        m = edge_metrics(obs, n)
        # prod traj naming
        if slot <= 20:
            tp = TRAJ / f"SH{slot:02d}_gcm_gcm.csv"
            tag = "prod"
        else:
            tp = TRAJ / f"SH{slot:02d}_gen_gcm.csv"
            tag = "gen"
        oob = traj_oob(tp)
        press = traj_y1_with_north_obs(tp, obs)
        print(
            f"{slot:4d} {tag:>8} {m['n_obs']:4d} {m['n_agvs']:3d} {m['y2_count']:3d} "
            f"{m['y2_on_home_cols']:6d} {m['home_north_blocked']:4d} "
            f"{m['home_north_frac']:5.2f} {m['x2_count']:3d} {oob:6d} {press:8d}"
        )

    # summary
    def avg(slots, key):
        vals = []
        for s in slots:
            obs, n, _ = load_obs(s)
            vals.append(edge_metrics(obs, n)[key])
        return sum(vals) / len(vals)

    print("\n--- averages ---")
    for name, slots in (("SH01-20", range(1, 21)), ("SH21-40", range(21, 41))):
        print(
            f"{name}: y2_count={avg(slots,'y2_count'):.1f} "
            f"y2_on_home={avg(slots,'y2_on_home_cols'):.2f} "
            f"home_north_frac={avg(slots,'home_north_frac'):.2f} "
            f"x2_count={avg(slots,'x2_count'):.1f}"
        )


if __name__ == "__main__":
    main()
