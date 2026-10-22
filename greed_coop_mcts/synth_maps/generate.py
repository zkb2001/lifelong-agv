"""Generate SH-like warehouse maps (json + position + task CSVs).

Pickup stations are FIXED (SH canonical left/right wall pads). Obstacles,
dropoffs, and AGV homes are randomized under strict geometric validity:
  - single free-space connected component (no islands)
  - every AGV reachable from the free component
  - dropoffs not fully boxed (at least one open 4-neighbor)
  - left pickups: cell immediately to the right is free of obstacle/dropoff
  - right pickups: cell immediately to the left is free of obstacle/dropoff
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from greed_coop_mcts.paths import MAP_EXPORTS_SYNTH
from greed_coop_mcts.pgcp import near_pad_deg2_count, throat_band_fleet_cap
from greed_coop_mcts.rule_obs import GRID, LEGAL_MIN, seal_ghost_border

Cell = Tuple[int, int]
HI = GRID - 1  # 20

# SH-canonical pickup pads — never move these.
FIXED_PICKUPS: List[dict] = [
    {"name": "Tiger", "x": 1, "y": 6},
    {"name": "Dragon", "x": 1, "y": 10},
    {"name": "Horse", "x": 1, "y": 14},
    {"name": "Rabbit", "x": 20, "y": 6},
    {"name": "Ox", "x": 20, "y": 10},
    {"name": "Monkey", "x": 20, "y": 14},
]
PICK_NAMES = [p["name"] for p in FIXED_PICKUPS]
DROP_NAMES = [
    "Beijing",
    "Shanghai",
    "Suzhou",
    "Hangzhou",
    "Nanjing",
    "Wuhan",
    "Changsha",
    "Guangzhou",
    "Chengdu",
    "Xiamen",
    "Kunming",
    "Urumqi",
    "Shenzhen",
    "Dalian",
    "Tianjin",
    "Chongqing",
]
AGV_NAMES = [
    "Optimus",
    "Ironhide",
    "Bumblebee",
    "Hound",
    "Jazz",
    "Smokescreen",
    "Sideswipe",
    "Megatron",
    "Ratchet",
    "Prowl",
    "Bluestreak",
    "Sunstreaker",
    "Trailbreaker",
    "Mirage",
    "Wheeljack",
    "Perceptor",
    "HotRod",
    "Kup",
    "Arcee",
    "Springer",
    "UltraMagnus",
    "Rodimus",
    "Grimlock",
    "Slag",
    "Sludge",
    "Snarl",
    "Swoop",
    "Blitzwing",
    "Astrotrain",
    "Octane",
]


def _neighbors(c: Cell) -> List[Cell]:
    x, y = c
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (x + dx, y + dy)
        if LEGAL_MIN <= n[0] <= HI and LEGAL_MIN <= n[1] <= HI:
            out.append(n)
    return out


def _bfs_reachable(start: Cell, blocked: Set[Cell]) -> Set[Cell]:
    if start in blocked:
        return set()
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for n in _neighbors(cur):
            if n in blocked or n in seen:
                continue
            seen.add(n)
            q.append(n)
    return seen


def _all_free(blocked: Set[Cell]) -> Set[Cell]:
    return {
        (x, y)
        for x in range(LEGAL_MIN, HI + 1)
        for y in range(LEGAL_MIN, HI + 1)
        if (x, y) not in blocked
    }


def _add_block(blocked: Set[Cell], cells: List[Cell], protected: Set[Cell]) -> None:
    for c in cells:
        if c in protected:
            continue
        if LEGAL_MIN <= c[0] <= HI and LEGAL_MIN <= c[1] <= HI:
            blocked.add(c)


def _pickup_approach_ok(pickups: List[dict], forbidden: Set[Cell]) -> bool:
    """Left wall: +x free of obstacle/dropoff; right wall: -x free."""
    for p in pickups:
        x, y = int(p["x"]), int(p["y"])
        if x == 1:
            approach = (2, y)
        elif x == HI:
            approach = (HI - 1, y)
        else:
            # interior pickups not expected; require any free neighbor
            if not any(n not in forbidden for n in _neighbors((x, y))):
                return False
            continue
        if approach in forbidden:
            return False
    return True


def _dropoffs_not_boxed(
    dropoffs: List[dict], blocked: Set[Cell], stations: Set[Cell]
) -> bool:
    """Each dropoff must have ≥1 4-neighbor that is neither obstacle nor station."""
    hard = blocked | stations
    for d in dropoffs:
        c = (int(d["x"]), int(d["y"]))
        if not any(n not in hard for n in _neighbors(c)):
            return False
    return True


def _single_component(blocked: Set[Cell], seeds: List[Cell]) -> Optional[Set[Cell]]:
    """All free cells must form one connected component containing every seed."""
    free = _all_free(blocked)
    if not free:
        return None
    start = None
    for s in seeds:
        if s in free:
            start = s
            break
    if start is None:
        return None
    reach = _bfs_reachable(start, blocked)
    if reach != free:
        return None
    for s in seeds:
        if s not in reach:
            return None
    return reach


def _fingerprint(
    blocked: Set[Cell], dropoffs: List[dict], agvs: List[dict]
) -> str:
    obs = sorted(f"{x},{y}" for x, y in blocked)
    drops = sorted(f"{d['name']}:{d['x']},{d['y']}" for d in dropoffs)
    homes = sorted(f"{a['name']}:{a['x']},{a['y']}:{a['pitch']}" for a in agvs)
    raw = "|".join(["OBS"] + obs + ["DROP"] + drops + ["AGV"] + homes)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _validate_layout(
    pickups: List[dict],
    dropoffs: List[dict],
    blocked: Set[Cell],
    agvs: List[dict],
) -> bool:
    pick_cells = {(int(p["x"]), int(p["y"])) for p in pickups}
    drop_cells = {(int(d["x"]), int(d["y"])) for d in dropoffs}
    stations = pick_cells | drop_cells

    # Pads themselves must not be obstacles.
    if any(c in blocked for c in stations):
        return False

    # Approach cells: not obstacle and not dropoff (pickups OK only on pad itself).
    approach_forbidden = blocked | drop_cells
    if not _pickup_approach_ok(pickups, approach_forbidden):
        return False

    if not _dropoffs_not_boxed(dropoffs, blocked, stations):
        return False

    agv_cells = [(int(a["x"]), int(a["y"])) for a in agvs]
    if any(c in blocked or c in stations for c in agv_cells):
        return False
    if len(set(agv_cells)) != len(agv_cells):
        return False

    seeds = list(pick_cells) + list(drop_cells) + agv_cells
    reach = _single_component(blocked, seeds)
    if reach is None:
        return False
    return True


def generate_one(
    rng: random.Random, map_idx: int, *, regime: Optional[str] = None
) -> Optional[dict]:
    """Return one valid map bundle, or None if this attempt fails.

    regime: easy|mid|hard (aliases: open|crowded). Controls density/fleet pressure.
    """
    if regime is None:
        regime = rng.choice(["easy", "mid", "hard"])
    regime = str(regime).lower()
    # Difficulty = obstacle pressure (post-repair n_obs bands), not raw free alone.
    # Grid interior ~400 cells; keep all regimes solvable + idle_pool≥0.2.
    if regime in ("open", "easy"):
        regime = "easy"
        n_agvs = rng.randint(14, 24)
        n_drop = rng.randint(10, 16)
        n_rects = rng.randint(1, 3)
        n_scatter = rng.randint(0, 5)
        aisle_p = 0.2
        obs_lo, obs_hi = 15, 55
    elif regime in ("mid", "medium"):
        regime = "mid"
        n_agvs = rng.randint(10, 18)
        n_drop = rng.randint(8, 12)
        n_rects = rng.randint(3, 6)
        n_scatter = rng.randint(6, 14)
        aisle_p = 0.55
        obs_lo, obs_hi = 56, 100
    else:
        regime = "hard"
        n_agvs = rng.randint(8, 14)
        n_drop = rng.randint(6, 10)
        n_rects = rng.randint(5, 9)
        n_scatter = rng.randint(12, 22)
        aisle_p = 0.8
        obs_lo, obs_hi = 101, 160

    n_agvs = min(n_agvs, len(AGV_NAMES))
    pickups = [dict(p) for p in FIXED_PICKUPS]
    pick_names = PICK_NAMES[:]
    drop_names = DROP_NAMES[:n_drop]

    pick_cells = {(p["x"], p["y"]) for p in pickups}
    # Approach cells + pads protected from obstacles.
    protected: Set[Cell] = set(pick_cells)
    for p in pickups:
        x, y = p["x"], p["y"]
        if x == 1:
            protected.add((2, y))
        elif x == HI:
            protected.add((HI - 1, y))
        protected.update(_neighbors((x, y)))

    used: Set[Cell] = set(pick_cells)
    # Reserve approach cells so dropoffs never sit on them.
    for p in pickups:
        x, y = p["x"], p["y"]
        if x == 1:
            used.add((2, y))
        elif x == HI:
            used.add((HI - 1, y))

    dropoffs: List[dict] = []
    lattice = [(x, y) for x in (6, 9, 12, 15) for y in (4, 8, 12, 16)]
    rng.shuffle(lattice)
    for i, name in enumerate(drop_names):
        placed = False
        for c in lattice:
            if c in used:
                continue
            # Keep dropoffs off pickup 4-neighborhood.
            if any(abs(c[0] - p["x"]) + abs(c[1] - p["y"]) <= 1 for p in pickups):
                continue
            used.add(c)
            dropoffs.append({"name": name, "x": c[0], "y": c[1]})
            placed = True
            break
        if placed:
            continue
        for _ in range(80):
            x = rng.randint(4, HI - 3)
            y = rng.randint(3, HI - 2)
            c = (x, y)
            if c in used:
                continue
            if any(abs(c[0] - p["x"]) + abs(c[1] - p["y"]) <= 1 for p in pickups):
                continue
            used.add(c)
            dropoffs.append({"name": name, "x": x, "y": y})
            placed = True
            break
        if not placed:
            return None

    drop_cells = {(d["x"], d["y"]) for d in dropoffs}
    stations = pick_cells | drop_cells
    # Protect dropoff neighborhoods lightly so they are not fully boxed by default.
    for c in list(drop_cells):
        protected.add(c)
        # keep at least one random neighbor free preference by protecting one
        neigh = _neighbors(c)
        if neigh:
            protected.add(rng.choice(neigh))

    blocked: Set[Cell] = set()
    for _ in range(n_rects):
        w = rng.randint(1, 3 if regime != "hard" else 4)
        h = rng.randint(1, 4 if regime != "hard" else 5)
        x0 = rng.randint(2, HI - w)
        y0 = rng.randint(2, HI - h)
        cells = [(x0 + dx, y0 + dy) for dx in range(w) for dy in range(h)]
        _add_block(blocked, cells, protected)

    if rng.random() < aisle_p:
        x = rng.randint(5, 15)
        gap = 4 if regime == "easy" else (3 if regime == "mid" else 2)
        _add_block(
            blocked,
            [(x, y) for y in range(2, HI) if y % gap != 0],
            protected,
        )
    if regime == "hard" and rng.random() < 0.5:
        y = rng.randint(5, 15)
        _add_block(
            blocked,
            [(x, y) for x in range(3, HI - 1) if x % 3 != 0],
            protected,
        )

    for _ in range(n_scatter):
        c = (rng.randint(3, HI - 2), rng.randint(3, HI - 2))
        _add_block(blocked, [c], protected)

    blocked = seal_ghost_border(blocked)

    # Connectivity repair: carve manhattan tunnels from first pickup to every station.
    seed = (pickups[0]["x"], pickups[0]["y"])
    reach = _bfs_reachable(seed, blocked)
    for st in stations:
        if st in reach:
            continue
        x, y = seed
        tx, ty = st
        while (x, y) != (tx, ty):
            if x < tx:
                x += 1
            elif x > tx:
                x -= 1
            elif y < ty:
                y += 1
            else:
                y -= 1
            blocked.discard((x, y))
        reach = _bfs_reachable(seed, blocked)

    # Clear islands by flooding and removing unreachable free cells → obstacles
    # (better: carve to merge). Prefer carving from reach into orphan pockets.
    free = _all_free(blocked)
    if reach != free:
        orphans = free - reach
        for oc in list(orphans):
            # tunnel orphan → seed
            x, y = oc
            tx, ty = seed
            while (x, y) != (tx, ty):
                if x < tx:
                    x += 1
                elif x > tx:
                    x -= 1
                elif y < ty:
                    y += 1
                else:
                    y -= 1
                blocked.discard((x, y))
            reach = _bfs_reachable(seed, blocked)
            free = _all_free(blocked)
            if reach == free:
                break

    # Ensure dropoff open neighbor after obstacle placement
    for d in dropoffs:
        c = (d["x"], d["y"])
        hard = blocked | stations
        if not any(n not in hard for n in _neighbors(c)):
            for n in _neighbors(c):
                if n not in stations:
                    blocked.discard(n)
                    break

    # Ensure pickup approach clear of obstacles (dropoffs already barred via used)
    for p in pickups:
        x, y = p["x"], p["y"]
        approach = (2, y) if x == 1 else ((HI - 1, y) if x == HI else None)
        if approach is not None:
            blocked.discard(approach)

    def _interior_obs_count() -> int:
        return sum(
            1
            for x, y in blocked
            if LEGAL_MIN <= x <= HI and LEGAL_MIN <= y <= HI
        )

    # Densify to hit regime obstacle band while preserving connectivity & approach rules.
    target_obs = rng.randint(obs_lo, obs_hi)
    densify_guard = protected | stations
    for p in pickups:
        x, y = p["x"], p["y"]
        if x == 1:
            densify_guard.add((2, y))
        elif x == HI:
            densify_guard.add((HI - 1, y))
    # Keep one open neighbor per dropoff forever.
    drop_open: Set[Cell] = set()
    for d in dropoffs:
        c = (d["x"], d["y"])
        hard = blocked | stations
        opens = [n for n in _neighbors(c) if n not in hard]
        if opens:
            keep = rng.choice(opens)
            drop_open.add(keep)
            densify_guard.add(keep)

    seed = (pickups[0]["x"], pickups[0]["y"])
    for _ in range(5000):
        cur = _interior_obs_count()
        if cur >= target_obs:
            break
        c = (rng.randint(LEGAL_MIN, HI), rng.randint(LEGAL_MIN, HI))
        if c in densify_guard or c in blocked:
            continue
        blocked.add(c)
        if not _pickup_approach_ok(pickups, blocked | drop_cells):
            blocked.discard(c)
            continue
        if not _dropoffs_not_boxed(dropoffs, blocked, stations):
            blocked.discard(c)
            continue
        reach = _bfs_reachable(seed, blocked)
        free_now = _all_free(blocked)
        if reach != free_now or any(st not in reach for st in stations):
            blocked.discard(c)
            continue

    # Thin if overshoot (rare; keeps easy maps in band).
    removable = [
        c
        for c in list(blocked)
        if LEGAL_MIN <= c[0] <= HI
        and LEGAL_MIN <= c[1] <= HI
        and c not in densify_guard
    ]
    rng.shuffle(removable)
    for c in removable:
        if _interior_obs_count() <= obs_hi:
            break
        blocked.discard(c)

    free_n = sum(
        1
        for x in range(LEGAL_MIN, HI + 1)
        for y in range(LEGAL_MIN, HI + 1)
        if (x, y) not in blocked
    )
    near = near_pad_deg2_count(blocked | stations, drop_cells)
    k_tbc = throat_band_fleet_cap(n_agvs, near, base=6)
    idle = (n_agvs - k_tbc) / max(1, n_agvs)
    if idle < 0.2:
        n_agvs = min(30, max(n_agvs, k_tbc + max(2, int(0.25 * k_tbc + 1))))
        idle = (n_agvs - k_tbc) / max(1, n_agvs)

    # Place AGVs on free top/bottom rows inside the connected component.
    reach = _bfs_reachable(seed, blocked)
    agvs: List[dict] = []
    home_used: Set[Cell] = set()
    for i in range(n_agvs):
        name = AGV_NAMES[i]
        placed = False
        for _ in range(80):
            y = 1 if i % 2 == 0 else HI
            x = rng.randint(2, HI - 1)
            c = (x, y)
            if c in blocked or c in stations or c in home_used:
                continue
            if c not in reach:
                continue
            home_used.add(c)
            pitch = 90 if y == 1 else 270
            agvs.append({"name": name, "x": x, "y": y, "pitch": pitch})
            placed = True
            break
        if not placed:
            # fall back: any free reachable cell
            candidates = [c for c in reach if c not in stations and c not in home_used]
            if not candidates:
                return None
            c = rng.choice(candidates)
            home_used.add(c)
            pitch = 90 if c[1] <= 10 else 270
            agvs.append({"name": name, "x": c[0], "y": c[1], "pitch": pitch})

    if not _validate_layout(pickups, dropoffs, blocked, agvs):
        return None
    if idle < 0.2 or free_n < 120:
        return None
    n_obs = _interior_obs_count()
    if not (obs_lo <= n_obs <= obs_hi):
        return None



    tasks = []
    for t in range(100):
        sp = rng.choice(pick_names)
        ep = rng.choice(drop_names)
        cnt = sum(1 for row in tasks if row[1] == sp) + 1
        tasks.append((f"{sp}-{cnt}", sp, ep, "Normal", "None"))

    sid = f"SYN_{map_idx:05d}"
    obstacles = sorted(blocked)
    fp = _fingerprint(blocked, dropoffs, agvs)
    jdoc = {
        "id": sid,
        "map_id": sid,
        "format_version": 2,
        "grid_min": 1,
        "grid_max": 20,
        "n_agvs": n_agvs,
        "n_pickups": len(pickups),
        "n_dropoffs": len(dropoffs),
        "n_tasks": 100,
        "pickups": pickups,
        "dropoffs": dropoffs,
        "extra_obstacles": [[x, y] for x, y in obstacles],
        "regime": regime,
        "fingerprint": fp,
        "notes": "synth warehouse for FACN; fixed SH pickups",
    }
    return {
        "id": sid,
        "idx": map_idx,
        "json": jdoc,
        "pickups": pickups,
        "dropoffs": dropoffs,
        "agvs": agvs,
        "tasks": tasks,
        "fingerprint": fp,
        "geom": {
            "n": n_agvs,
            "free": free_n,
            "n_obs": int(n_obs),
            "near": int(near),
            "k_tbc": int(k_tbc),
            "idle_pool": float(idle),
            "regime": regime,
            "fingerprint": fp,
        },
    }


def write_map(bundle: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = bundle["id"]
    jpath = out_dir / f"{sid}.json"
    jpath.write_text(json.dumps(bundle["json"], indent=2), encoding="utf-8")

    pos_path = out_dir / f"{sid}_position.csv"
    with pos_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "name", "x", "y", "pitch"])
        for p in bundle["pickups"]:
            w.writerow(["start_point", p["name"], p["x"], p["y"], ""])
        for d in bundle["dropoffs"]:
            w.writerow(["end_point", d["name"], d["x"], d["y"], ""])
        for a in bundle["agvs"]:
            w.writerow(["agv", a["name"], a["x"], a["y"], a["pitch"]])

    task_path = out_dir / f"{sid}_task.csv"
    with task_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["task_id", "start_point", "end_point", "priority", "remaining_time"]
        )
        for row in bundle["tasks"]:
            w.writerow(row)
    return jpath


def load_synth_meta(map_idx: int, *, exports: Path | None = None) -> dict:
    root = Path(exports or MAP_EXPORTS_SYNTH)
    sid = f"SYN_{int(map_idx):05d}"
    jpath = root / f"{sid}.json"
    task_csv = root / f"{sid}_task.csv"
    pos_csv = root / f"{sid}_position.csv"
    data = json.loads(jpath.read_text(encoding="utf-8"))
    return {
        "id": sid,
        "slot": int(map_idx),
        "source_slot": sid,
        "map_shape": "synth",
        "task_csv": str(task_csv.resolve()),
        "position_csv": str(pos_csv.resolve()),
        "map_json": str(jpath.resolve()),
        "extra_obstacles": data.get("extra_obstacles") or [],
        "n_tasks": int(data.get("n_tasks") or 100),
        "n_agvs": int(data.get("n_agvs") or 0),
        "max_sim_time": 200000,
        "wall_timeout": 600.0,
        "pickups": data.get("pickups") or [],
        "notes": data.get("notes", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synth warehouse maps")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--start-idx",
        type=int,
        default=10001,
        help="synth ids start at 10001 to avoid SH 1-40 collision",
    )
    ap.add_argument("--out", type=str, default=str(MAP_EXPORTS_SYNTH))
    ap.add_argument(
        "--clear",
        action="store_true",
        help="delete existing SYN_* files in out dir before writing",
    )
    ap.add_argument("--max-tries-per-map", type=int, default=200)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        removed = 0
        for pat in ("SYN_*.json", "SYN_*_position.csv", "SYN_*_task.csv", "index.json"):
            for p in out_dir.glob(pat):
                p.unlink(missing_ok=True)
                removed += 1
        print(f"[synth] cleared ~{removed} files in {out_dir}", flush=True)

    rng = random.Random(int(args.seed))
    n_total = int(args.n)
    # Equal easy / mid / hard quotas (remainder → mid).
    n_easy = n_total // 3
    n_hard = n_total // 3
    n_mid = n_total - n_easy - n_hard
    regime_queue: List[str] = (
        ["easy"] * n_easy + ["mid"] * n_mid + ["hard"] * n_hard
    )
    rng.shuffle(regime_queue)

    index = []
    seen_fp: Set[str] = set()
    fails = 0
    counts = {"easy": 0, "mid": 0, "hard": 0}
    i = 0
    while i < n_total:
        idx = int(args.start_idx) + i
        target_regime = regime_queue[i]
        bundle = None
        for _ in range(int(args.max_tries_per_map)):
            cand = generate_one(rng, idx, regime=target_regime)
            if cand is None:
                fails += 1
                continue
            fp = cand["fingerprint"]
            if fp in seen_fp:
                fails += 1
                continue
            seen_fp.add(fp)
            bundle = cand
            break
        if bundle is None:
            print(
                f"[synth] FATAL: could not produce unique valid "
                f"{target_regime} map idx={idx}",
                flush=True,
            )
            return 1
        write_map(bundle, out_dir)
        index.append({"id": bundle["id"], "idx": idx, **bundle["geom"]})
        counts[bundle["geom"]["regime"]] = counts.get(bundle["geom"]["regime"], 0) + 1
        i += 1
        if i % 100 == 0 or i == 1 or i == n_total:
            print(
                f"[synth] wrote {bundle['id']} ({i}/{n_total}) "
                f"regime={bundle['geom']['regime']} "
                f"idle={bundle['geom']['idle_pool']:.2f} "
                f"counts={counts} fails={fails}",
                flush=True,
            )
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(
        f"[synth] done n={len(index)} unique={len(seen_fp)} "
        f"counts={counts} fails={fails} → {out_dir}",
        flush=True,
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
