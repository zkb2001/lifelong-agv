"""Deep-dive: why gen maps produce OOB (x=0/y=0) while looking connected."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from greed_coop_mcts.scenario import load_slot_meta, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.hier_coord.scene_difficulty.map_gen import (
    free_cells,
    is_connected,
)

ROOT = Path(__file__).resolve().parent
EXPORTS = Path(__file__).resolve().parents[2] / "ml_research" / "map_editor" / "exports"


def load_map(slot: int):
    sid = f"SH_custom_{slot:02d}"
    j = json.loads((EXPORTS / f"{sid}.json").read_text(encoding="utf-8"))
    obs = set()
    for o in j.get("extra_obstacles") or []:
        if isinstance(o, dict):
            obs.add((int(o["x"]), int(o["y"])))
        else:
            obs.add((int(o[0]), int(o[1])))
    pick = {(int(p["x"]), int(p["y"])) for p in j.get("pickups") or []}
    drop = {(int(p["x"]), int(p["y"])) for p in j.get("dropoffs") or []}
    agvs = []
    with (EXPORTS / f"{sid}_position.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("type", "")).lower() == "agv":
                agvs.append((r["name"], int(r["x"]), int(r["y"])))
    return j, obs, pick, drop, agvs


def traj_oob_events(traj: Path):
    """Return list of (t, name, x, y, prev_xy) for OOB ticks."""
    by_agv: dict[str, list] = defaultdict(list)
    with traj.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_agv[r["name"]].append(r)
    events = []
    for name, seq in by_agv.items():
        seq.sort(key=lambda r: int(r["timestamp"]))
        prev = None
        for r in seq:
            x, y = int(r["X"]), int(r["Y"])
            t = int(r["timestamp"])
            if x < 1 or x > 20 or y < 1 or y > 20:
                events.append(
                    {
                        "t": t,
                        "agv": name,
                        "xy": (x, y),
                        "prev": prev,
                        "pitch": int(r.get("pitch") or 0),
                        "loaded": r.get("loaded"),
                        "task": r.get("task_id") or r.get("task"),
                    }
                )
            prev = (x, y)
    return events


def edge_block_stats(obs, pick, drop):
    """How blocked are the inward neighbors of edge stations / AGV row."""
    # AGV home row y=1: cell (x,2) is inward
    home_inward_blocked = []
    for x in range(1, 21):
        if (x, 1) in obs:
            home_inward_blocked.append(("home_cell_blocked", x, 1))
        if (x, 2) in obs:
            home_inward_blocked.append(("home_inward_blocked", x, 2))
    # left pickups x=1 -> inward (2,y); right x=20 -> (19,y)
    pickup_inward = []
    for x, y in pick:
        inward = (x + 1, y) if x == 1 else (x - 1, y) if x == 20 else None
        if inward and inward in obs:
            pickup_inward.append((x, y, inward))
    # dropoffs near bottom: y=4 often — check (x,3)/(x,5)
    drop_south = []
    for x, y in drop:
        if y <= 5:
            for ny in (y - 1, y - 2):
                if ny >= 1 and (x, ny) in obs:
                    drop_south.append(((x, y), (x, ny)))
    return {
        "home_issues": home_inward_blocked,
        "pickup_inward_blocked": pickup_inward,
        "drop_south_blocked": drop_south[:20],
        "obs_on_y1": sorted(c for c in obs if c[1] == 1),
        "obs_on_y2": sorted(c for c in obs if c[1] == 2),
        "obs_on_x1": sorted(c for c in obs if c[0] == 1),
        "obs_on_x2": sorted(c for c in obs if c[0] == 2),
        "obs_on_x19": sorted(c for c in obs if c[0] == 19),
        "obs_on_x20": sorted(c for c in obs if c[0] == 20),
    }


def neighbor_context(obs, pick, drop, cell):
    x, y = cell
    out = {}
    for dx, dy, lab in (
        (0, 1, "N"),
        (0, -1, "S"),
        (1, 0, "E"),
        (-1, 0, "W"),
    ):
        n = (x + dx, y + dy)
        if n[0] < 1 or n[0] > 20 or n[1] < 1 or n[1] > 20:
            out[lab] = "OOB"
        elif n in obs:
            out[lab] = "OBS"
        elif n in pick:
            out[lab] = "PICK"
        elif n in drop:
            out[lab] = "DROP"
        else:
            out[lab] = "FREE"
    return out


def analyze_slot(slot: int):
    j, obs, pick, drop, agvs = load_map(slot)
    traj = ROOT / "trajectories" / f"SH{slot:02d}_gen_gcm.csv"
    if not traj.exists():
        print(f"SH{slot:02d}: no traj")
        return
    events = traj_oob_events(traj)
    connected = is_connected(obs)
    edge = edge_block_stats(obs, pick, drop)
    print("=" * 72)
    print(
        f"SH{slot:02d} style={j.get('map_shape')} diff={j.get('difficulty')} "
        f"obs={len(obs)} n_agv={j.get('n_agvs')} connected={connected} "
        f"oob_events={len(events)}"
    )
    if not events:
        print("  (VALID / no OOB)")
        print(f"  obs_on_y2={edge['obs_on_y2']}")
        print(f"  obs_on_x2={edge['obs_on_x2'][:10]} ...")
        return

    # entry transitions: from in-bounds -> OOB
    entries = [e for e in events if e["prev"] and 1 <= e["prev"][0] <= 20 and 1 <= e["prev"][1] <= 20]
    entry_cells = Counter((e["prev"], e["xy"]) for e in entries)
    print(f"  unique entry transitions (top):")
    for (prev, xy), cnt in entry_cells.most_common(8):
        ctx = neighbor_context(obs, pick, drop, prev)
        print(f"    {prev} -> {xy} x{cnt}  neighbors@{prev}={ctx}")

    oob_cells = Counter(e["xy"] for e in events)
    print(f"  oob cells: {oob_cells.most_common(8)}")
    print(f"  agvs involved: {Counter(e['agv'] for e in events).most_common()}")
    print(f"  obs_on_y1={edge['obs_on_y1']}")
    print(f"  obs_on_y2={edge['obs_on_y2']}")
    print(f"  obs_on_x1={edge['obs_on_x1']}")
    print(f"  obs_on_x2 count={len(edge['obs_on_x2'])} sample={edge['obs_on_x2'][:8]}")
    print(f"  pickup_inward_blocked={edge['pickup_inward_blocked']}")

    # first episode
    e0 = entries[0] if entries else events[0]
    print(f"  first OOB: t={e0['t']} {e0['agv']} prev={e0.get('prev')} xy={e0['xy']} "
          f"pitch={e0['pitch']} loaded={e0['loaded']} task={e0['task']}")


def main():
    rows = json.loads((ROOT / "GEN_CURRICULUM_EVAL.json").read_text(encoding="utf-8"))
    # focus INVALID + a few VALID for contrast
    invalid = [r["slot"] for r in rows if not r["ok"]]
    valid = [r["slot"] for r in rows if r["ok"]]
    print("INVALID slots:", invalid)
    print("VALID slots:", valid)
    for slot in sorted(invalid)[:6]:
        analyze_slot(slot)
    print("\n--- VALID contrast ---")
    for slot in valid[:3]:
        analyze_slot(slot)
    # also SH25 (high mot) and SH31
    print("\n--- HIGH MOT ---")
    for slot in (25, 31):
        analyze_slot(slot)


if __name__ == "__main__":
    main()
