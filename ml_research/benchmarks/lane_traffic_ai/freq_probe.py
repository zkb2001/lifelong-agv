"""Frequency probe: single-AGV cartesian A* → artery / parking labels.

Protocol (per map):
1. Cartesian cover: every AGV alone delivers every (pickup × dropoff).
2. Visited cells → artery (highest-frequency trunk by construction).
3. Unload cell + walkable 4-neighbors → artery; pickup 8-neighbors → artery.
4. Remaining walkable → parking.
5. Export training samples for ParkArterySemaphoreNet BC.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import GRID, PLAY_HI, PLAY_LO, RESULTS_DIR, ensure_dirs
from .features import build_park_artery_channels, topology_degree_field
from .teacher import (
    build_sem_features,
    teacher_connectivity_field,
    teacher_semaphore_label,
)

Cell = Tuple[int, int]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

FREQ_DIR = RESULTS_DIR / "freq_probe"


def stations_dests_from_meta(meta: dict) -> Tuple[List[dict], List[dict]]:
    jpath = Path(meta.get("map_json") or "")
    pickups: List[dict] = list(meta.get("pickups") or [])
    dropoffs: List[dict] = []
    if jpath.exists():
        data = json.loads(jpath.read_text(encoding="utf-8"))
        if not pickups:
            pickups = list(data.get("pickups") or [])
        dropoffs = list(data.get("dropoffs") or [])
    if not dropoffs:
        # fallback to scenario generator names if JSON missing dropoffs
        from ml_research.benchmarks.scenarios.generator import END_POINTS

        dropoffs = [{"name": n, "x": x, "y": y} for n, x, y in END_POINTS]
    return pickups, dropoffs


def walkable_and_obstacles(
    meta: dict,
    grid: int = GRID,
    *,
    block_dropoffs: bool = True,
    block_pickups: bool = False,
) -> Tuple[np.ndarray, Set[Cell]]:
    """walkable (H,W) float, obstacle set from map JSON + stations.

    Unload (dropoff) cells are obstacles — AGVs only use neighboring pads,
    matching engine ``get_end_points`` behavior.
    """
    walk = np.zeros((grid, grid), dtype=np.float32)
    for y in range(PLAY_LO, min(PLAY_HI, grid - 1) + 1):
        for x in range(PLAY_LO, min(PLAY_HI, grid - 1) + 1):
            walk[y, x] = 1.0
    obs: Set[Cell] = set()
    jpath = Path(meta.get("map_json") or "")
    data: dict = {}
    if jpath.exists():
        data = json.loads(jpath.read_text(encoding="utf-8"))
        for p in data.get("extra_obstacles") or []:
            if isinstance(p, dict):
                obs.add((int(p["x"]), int(p["y"])))
            else:
                obs.add((int(p[0]), int(p[1])))
    for p in meta.get("extra_obstacles") or []:
        if isinstance(p, dict):
            obs.add((int(p["x"]), int(p["y"])))
        else:
            obs.add((int(p[0]), int(p[1])))

    pickups, dropoffs = stations_dests_from_meta(meta)
    if block_dropoffs:
        for d in dropoffs:
            obs.add((int(d["x"]), int(d["y"])))
    if block_pickups:
        for p in pickups:
            obs.add((int(p["x"]), int(p["y"])))

    for c in obs:
        x, y = c
        if 0 <= x < grid and 0 <= y < grid:
            walk[y, x] = 0.0
    return walk, obs


def nearest_walkable(walk: np.ndarray, cell: Cell) -> Optional[Cell]:
    x0, y0 = int(cell[0]), int(cell[1])
    H, W = walk.shape
    if 0 <= x0 < W and 0 <= y0 < H and walk[y0, x0] > 0.5:
        return (x0, y0)
    for r in range(1, 6):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) != r:
                    continue
                x, y = x0 + dx, y0 + dy
                if 0 <= x < W and 0 <= y < H and walk[y, x] > 0.5:
                    return (x, y)
    return None


def static_astar_path(
    walk: np.ndarray,
    start: Cell,
    goal: Cell,
) -> List[Cell]:
    """Unweighted shortest path on walkable grid (A*/BFS)."""
    s = nearest_walkable(walk, start)
    g = nearest_walkable(walk, goal)
    if s is None or g is None:
        return []
    if s == g:
        return [s]
    H, W = walk.shape
    prev: Dict[Cell, Optional[Cell]] = {s: None}
    q = deque([s])
    found = False
    while q:
        x, y = q.popleft()
        if (x, y) == g:
            found = True
            break
        for dx, dy in DIR4:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if walk[ny, nx] < 0.5:
                continue
            if (nx, ny) in prev:
                continue
            prev[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if not found:
        return []
    path: List[Cell] = []
    cur: Optional[Cell] = g
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def build_coverage_tasks(
    pickups: Sequence[dict],
    dropoffs: Sequence[dict],
    *,
    rounds: int = 1,
) -> List[dict]:
    """Every pickup→every dropoff, optionally repeated ``rounds`` times."""
    rows: List[dict] = []
    counters: Dict[str, int] = {}
    for _ in range(max(1, int(rounds))):
        for pk in pickups:
            st = str(pk.get("name") or "")
            if not st:
                continue
            for dp in dropoffs:
                dest = str(dp.get("name") or "")
                if not dest:
                    continue
                counters[st] = counters.get(st, 0) + 1
                rows.append(
                    {
                        "task_id": f"{st}-{counters[st]}",
                        "start_point": st,
                        "end_point": dest,
                        "priority": "Normal",
                        "remaining_time": "None",
                    }
                )
    return rows


def write_task_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "start_point",
                "end_point",
                "priority",
                "remaining_time",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "task_id": r["task_id"],
                    "start_point": r["start_point"],
                    "end_point": r["end_point"],
                    "priority": r.get("priority") or "Normal",
                    "remaining_time": r.get("remaining_time") or "None",
                }
            )


def load_agv_starts(meta: dict) -> List[Tuple[str, Cell]]:
    """AGV initial poses from position CSV."""
    path = Path(meta.get("position_csv") or "")
    out: List[Tuple[str, Cell]] = []
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = str(row.get("type") or "").strip().lower()
            if t not in ("agv", "robot", "vehicle"):
                continue
            name = str(row.get("name") or f"agv{len(out)}")
            out.append((name, (int(row["x"]), int(row["y"]))))
    return out


def accumulate_path(freq: np.ndarray, path: Sequence[Cell]) -> None:
    H, W = freq.shape
    for c in path:
        x, y = int(c[0]), int(c[1])
        if 0 <= x < W and 0 <= y < H:
            freq[y, x] += 1.0


def cartesian_delivery_frequency(
    meta: dict,
    pickups: Sequence[dict],
    dropoffs: Sequence[dict],
) -> np.ndarray:
    """Cartesian cover: every AGV alone delivers every (pickup×dropoff) task.

    For each (agv_start, pickup, dropoff):
      leg1 = A*(agv → pickup); leg2 = A*(pickup → dropoff)
    If a later leg fails, **still count legs planned before the failure**
    (do not discard the prefix).
    """
    walk, _ = walkable_and_obstacles(meta)
    freq = np.zeros((GRID, GRID), dtype=np.float32)
    agvs = load_agv_starts(meta)
    if not agvs:
        agvs = [
            (str(p.get("name") or i), (int(p["x"]), int(p["y"])))
            for i, p in enumerate(pickups)
        ]

    tasks = [
        (
            (int(pk["x"]), int(pk["y"])),
            (int(dp["x"]), int(dp["y"])),
            str(pk.get("name")),
            str(dp.get("name")),
        )
        for pk in pickups
        for dp in dropoffs
    ]
    n_full = 0
    n_prefix = 0
    n_empty = 0
    total = len(agvs) * len(tasks)
    for _agv_name, agv_xy in agvs:
        for pk_xy, dp_xy, _pk_n, _dp_n in tasks:
            leg1 = static_astar_path(walk, agv_xy, pk_xy)
            if not leg1:
                n_empty += 1
                continue
            accumulate_path(freq, leg1)
            leg2 = static_astar_path(walk, pk_xy, dp_xy)
            if not leg2:
                # planning failed after pickup leg — keep prefix only
                n_prefix += 1
                continue
            accumulate_path(freq, leg2)
            n_full += 1
    print(
        f"[FREQ] cartesian AGV×task full={n_full}/{total} "
        f"prefix_only={n_prefix} empty={n_empty} "
        f"agvs={len(agvs)} tasks={len(tasks)} map={meta.get('id')}",
        flush=True,
    )
    return freq


def labels_from_frequency(
    freq: np.ndarray,
    walk: np.ndarray,
    *,
    pickups: Optional[Sequence[dict]] = None,
    dropoffs: Optional[Sequence[dict]] = None,
    artery_q: float = 0.50,
    min_artery_abs: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Cartesian-visit + station pads → artery; remaining walkable → parking.

    Rules (no multi-AGV, no frequency quantiles):
    1. Every cell visited by single-AGV cartesian deliveries → artery
    2. Unload cell + walkable 4-neighbors (NSEW) → artery
    3. Pickup **8-neighbors only** (N/S/E/W + diagonals) that are walkable → artery.
       The pickup cell itself is NOT painted here; obstacle neighbors are skipped.
    4. All other walkable cells → parking
    5. Pickup cells are cleared from both artery and parking (station, not trunk).
    """
    del artery_q, min_artery_abs  # legacy kwargs ignored
    H, W = walk.shape
    walk_m = walk > 0.5
    artery = np.zeros((H, W), dtype=np.float32)
    parking = np.zeros((H, W), dtype=np.float32)
    station = np.zeros((H, W), dtype=np.float32)
    freq_norm = np.zeros((H, W), dtype=np.float32)
    mx = float(np.max(freq)) + 1e-6
    freq_norm[walk_m] = freq[walk_m] / mx

    # 1) cartesian visited paths = trunk
    artery[walk_m & (freq > 0)] = 1.0

    def _paint(x: int, y: int) -> None:
        # Obstacle / OOB → skip (not artery).
        if 0 <= x < W and 0 <= y < H and walk[y, x] > 0.5:
            artery[y, x] = 1.0

    # 2) unload cell + walkable 4-neighbors
    for d in dropoffs or ():
        x, y = int(d["x"]), int(d["y"])
        _paint(x, y)
        for dx, dy in DIR4:
            _paint(x + dx, y + dy)

    # 3) pickup mouth: 8-neighbors ONLY (never the pickup cell itself)
    dir8 = DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    pickup_cells: List[Cell] = []
    for p in pickups or ():
        x, y = int(p["x"]), int(p["y"])
        pickup_cells.append((x, y))
        for dx, dy in dir8:
            _paint(x + dx, y + dy)

    parking[walk_m & (artery < 0.5)] = 1.0
    artery[parking >= 0.5] = 0.0

    # Pickup cell ≠ 主干道, ≠ 停车点 (even if cartesian visited it).
    for x, y in pickup_cells:
        if 0 <= x < W and 0 <= y < H and walk[y, x] > 0.5:
            artery[y, x] = 0.0
            parking[y, x] = 0.0
            station[y, x] = 1.0

    return {
        "artery": artery.astype(np.float32),
        "parking": parking.astype(np.float32),
        "station": station.astype(np.float32),
        "freq_norm": freq_norm.astype(np.float32),
    }


def make_training_samples_from_labels(
    meta: dict,
    labels: Dict[str, np.ndarray],
    pickups: Sequence[dict],
    dropoffs: Sequence[dict],
    *,
    n_aug: int = 16,
    seed: int = 0,
) -> List[dict]:
    """Build ParkArtery channel+label dicts for BC."""
    rng = np.random.default_rng(seed)
    walk, _ = walkable_and_obstacles(meta)
    artery = labels["artery"]
    parking = labels["parking"]
    samples: List[dict] = []

    unload_cells = [(int(d["x"]), int(d["y"])) for d in dropoffs]
    pickup_cells = [(int(p["x"]), int(p["y"])) for p in pickups]

    for i in range(int(n_aug)):
        # random failed AGV pose on parking or walkable
        ys, xs = np.where(parking > 0.5)
        if len(xs) == 0:
            ys, xs = np.where(walk > 0.5)
        if len(xs) == 0:
            break
        j = int(rng.integers(0, len(xs)))
        fpos = (int(xs[j]), int(ys[j]))
        unload = unload_cells[int(rng.integers(0, len(unload_cells)))]
        # optional active-path band from artery
        active = np.zeros_like(walk)
        if rng.random() < 0.6:
            active = (artery >= 0.85).astype(np.float32) * float(rng.uniform(0.3, 0.8))

        ch = np.zeros((12, GRID, GRID), dtype=np.float32)
        ch[0] = 1.0 - walk
        ch[1] = walk
        ch[2, fpos[1], fpos[0]] = 1.0
        ch[3] = active * walk
        ch[4, fpos[1], fpos[0]] = 1.0
        # unload attractor
        for yy in range(GRID):
            for xx in range(GRID):
                if walk[yy, xx] < 0.5:
                    continue
                d = abs(xx - unload[0]) + abs(yy - unload[1])
                ch[5, yy, xx] = float(np.exp(-0.35 * d))
        for c in pickup_cells:
            for yy in range(GRID):
                for xx in range(GRID):
                    if walk[yy, xx] < 0.5:
                        continue
                    d = abs(xx - c[0]) + abs(yy - c[1])
                    ch[6, yy, xx] = max(ch[6, yy, xx], float(np.exp(-0.4 * d)))
        ch[7] = ch[6]
        ch[8, unload[1], unload[0]] = float(rng.random() > 0.4)
        # congestion soft
        ch[9] = np.clip(labels.get("freq_norm", artery) * float(rng.uniform(0.2, 1.0)), 0, 1)
        ch[10] = topology_degree_field(walk)
        ch[11] = (parking > 0.5).astype(np.float32) * float(rng.random() > 0.5)

        conn = teacher_connectivity_field(
            walk, artery, [unload], active_path=active if active.sum() > 0 else None
        )
        pad_free = 1.0 - float(ch[8, unload[1], unload[0]])
        path_clear = 1.0 if active[fpos[1], fpos[0]] < 0.5 else 0.0
        sem = teacher_semaphore_label(
            connectivity=conn,
            pos=fpos,
            goal=unload,
            pad_free=pad_free,
            path_clear=path_clear,
            congestion=float(ch[9, fpos[1], fpos[0]]),
        )
        sem_feat = build_sem_features(
            artery=artery,
            connectivity=conn,
            congestion=ch[9],
            pad_free=pad_free,
            path_clear=path_clear,
            pos=fpos,
            goal=unload,
        )
        samples.append(
            {
                "channels": ch,
                "parking": parking,
                "artery": artery,
                "connectivity": conn,
                "semaphore": float(sem),
                "sem_feat": sem_feat,
                "map_id": str(meta.get("id") or ""),
                "goals": [unload],
                "sem_pos": fpos,
                "sem_goal": unload,
                "pad_free": pad_free,
                "path_clear": path_clear,
                "source": "freq_probe",
            }
        )
    return samples


def probe_one_map(
    meta: dict,
    out_dir: Path,
    *,
    rounds: int = 1,
    n_aug: int = 24,
) -> dict:
    """Single-AGV cartesian cover → artery/parking labels."""
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    pickups, dropoffs = stations_dests_from_meta(meta)
    walk, _ = walkable_and_obstacles(meta)

    tasks = build_coverage_tasks(pickups, dropoffs, rounds=rounds)
    task_csv = out_dir / "coverage_tasks.csv"
    write_task_csv(task_csv, tasks)

    freq = cartesian_delivery_frequency(meta, pickups, dropoffs)
    labels = labels_from_frequency(
        freq, walk, pickups=pickups, dropoffs=dropoffs
    )
    samples = make_training_samples_from_labels(
        meta, labels, pickups, dropoffs, n_aug=n_aug, seed=hash(str(meta.get("id"))) % 10_000
    )

    np.savez_compressed(
        out_dir / "frequency.npz",
        freq=freq,
        freq_single=freq,
        artery=labels["artery"],
        parking=labels["parking"],
        freq_norm=labels["freq_norm"],
        walkable=walk,
    )
    np.savez_compressed(
        out_dir / "samples.npz",
        samples=np.asarray(samples, dtype=object),
    )
    summary = {
        "map_id": meta.get("id"),
        "n_pickups": len(pickups),
        "n_dropoffs": len(dropoffs),
        "n_tasks": len(tasks),
        "n_samples": len(samples),
        "freq_sum": float(freq.sum()),
        "artery_cells": int((labels["artery"] >= 0.5).sum()),
        "parking_cells": int((labels["parking"] >= 0.5).sum()),
        "mode": "cartesian_single_only",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        from .render_freq_map import render_artery_parking_png

        render_artery_parking_png(
            out_dir / "frequency.npz",
            out_dir / "artery_parking.png",
            title=str(meta.get("id") or ""),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FREQ] WARN png render failed: {exc}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def probe_slots(
    slots: Sequence[int],
    *,
    rounds: int = 1,
    n_aug: int = 24,
) -> Path:
    from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta

    ensure_dirs()
    root = FREQ_DIR / f"run_{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    merged_samples: List[dict] = []
    for slot in slots:
        meta = load_custom_meta(int(slot))
        if not meta:
            print(f"[FREQ] skip missing slot={slot}", flush=True)
            continue
        out = root / str(meta["id"])
        summary = probe_one_map(meta, out, rounds=rounds, n_aug=n_aug)
        all_summaries.append(summary)
        sp = out / "samples.npz"
        if sp.exists():
            data = np.load(sp, allow_pickle=True)
            merged_samples.extend(list(data["samples"]))
    np.savez_compressed(
        root / "all_samples.npz",
        samples=np.asarray(merged_samples, dtype=object),
    )
    (root / "index.json").write_text(
        json.dumps(
            {"summaries": all_summaries, "n_samples": len(merged_samples), "root": str(root)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"[FREQ] done maps={len(all_summaries)} samples={len(merged_samples)} → {root}",
        flush=True,
    )
    return root


def load_freq_samples(path: Path) -> List[dict]:
    path = Path(path)
    if path.is_dir():
        cand = path / "all_samples.npz"
        if not cand.exists():
            # merge per-map
            samples: List[dict] = []
            for sp in path.glob("*/samples.npz"):
                samples.extend(list(np.load(sp, allow_pickle=True)["samples"]))
            return samples
        path = cand
    if not path.exists():
        return []
    return list(np.load(path, allow_pickle=True)["samples"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cartesian single-AGV freq probe → artery/parking labels"
    )
    ap.add_argument("--slots", type=str, default="1,2,3", help="comma slots e.g. 1,2,3")
    ap.add_argument("--rounds", type=int, default=1, help="coverage task rounds")
    ap.add_argument("--n-aug", type=int, default=24, help="BC samples per map")
    args = ap.parse_args()
    slots = [int(x) for x in str(args.slots).split(",") if x.strip()]
    probe_slots(slots, rounds=int(args.rounds), n_aug=int(args.n_aug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
