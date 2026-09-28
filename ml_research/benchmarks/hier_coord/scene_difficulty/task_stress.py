"""Controlled non-random task stress patterns for difficulty labeling."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ml_research.benchmarks.scenarios.generator import (
    ALL_DESTS,
    ALL_STATIONS,
    END_POINTS,
    START_POINTS,
    write_position_csv,
)

TASK_PATTERNS = (
    "uniform",
    "same_pickup_10",
    "same_pickup_20",
    "same_dropoff_10",
    "same_dropoff_30",
    "pickup10_dropoff30",
    "fifo_deep",
    "cross_flow",
)


def _tid(station: str, counters: Dict[str, int]) -> str:
    counters[station] = counters.get(station, 0) + 1
    return f"{station}-{counters[station]}"


def gen_stress_tasks(
    pattern: str,
    n_tasks: int,
    seed: int,
) -> List[dict]:
    """Generate task rows with intentional bursts (same pickup / same dropoff)."""
    rng = random.Random(seed)
    counters: Dict[str, int] = {}
    rows: List[dict] = []

    def add(station: str, dest: str, priority: str = "Normal", rt="None"):
        rows.append(
            {
                "task_id": _tid(station, counters),
                "start_point": station,
                "end_point": dest,
                "priority": priority,
                "remaining_time": rt,
            }
        )

    pat = pattern.lower().strip()
    if pat == "uniform":
        for _ in range(n_tasks):
            add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS))
        return rows

    if pat.startswith("same_pickup_"):
        burst = int(pat.split("_")[-1])
        hot = rng.choice(ALL_STATIONS)
        for i in range(n_tasks):
            st = hot if i < burst else rng.choice(ALL_STATIONS)
            add(st, rng.choice(ALL_DESTS))
        return rows

    if pat.startswith("same_dropoff_"):
        burst = int(pat.split("_")[-1])
        hub = rng.choice(ALL_DESTS)
        for i in range(n_tasks):
            dest = hub if i < burst else rng.choice(ALL_DESTS)
            add(rng.choice(ALL_STATIONS), dest)
        return rows

    if pat in ("pickup10_dropoff30", "dual_stress"):
        hot = rng.choice(ALL_STATIONS)
        hub = rng.choice(ALL_DESTS)
        for i in range(n_tasks):
            st = hot if i < 10 else rng.choice(ALL_STATIONS)
            dest = hub if i < 30 else rng.choice(ALL_DESTS)
            add(st, dest)
        return rows

    if pat == "fifo_deep":
        # Deep same-station queues then rotate
        order = list(ALL_STATIONS)
        rng.shuffle(order)
        i = 0
        while len(rows) < n_tasks:
            st = order[i % len(order)]
            for _ in range(min(8, n_tasks - len(rows))):
                add(st, rng.choice(ALL_DESTS))
            i += 1
        return rows[:n_tasks]

    if pat == "cross_flow":
        left = ["Tiger", "Dragon", "Horse"]
        right = ["Rabbit", "Ox", "Monkey"]
        left_d = ["Beijing", "Shanghai", "Suzhou", "Hangzhou", "Nanjing", "Wuhan"]
        right_d = ["Shenzhen", "Dalian", "Tianjin", "Chongqing", "Chengdu", "Xiamen"]
        for i in range(n_tasks):
            if i % 2 == 0:
                add(rng.choice(left), rng.choice(right_d))
            else:
                add(rng.choice(right), rng.choice(left_d))
        return rows

    raise ValueError(f"unknown task pattern: {pattern}")


def gen_phased_tasks(
    *,
    n_random_pre: int = 100,
    n_burst: int = 100,
    n_random_post: int = 100,
    burst_mode: str = "same_dropoff",
    seed: int = 0,
) -> List[dict]:
    """Lifelong stress: uniform → concentrated (pickup or dropoff) → uniform.

    ``burst_mode``: ``same_dropoff`` | ``same_pickup`` | ``dual``
    Each row gets ``_phase`` ∈ {random_pre, burst, random_post} for logging.
    """
    rng = random.Random(seed)
    counters: Dict[str, int] = {}
    rows: List[dict] = []

    def add(station: str, dest: str, phase: str, priority: str = "Normal"):
        rows.append(
            {
                "task_id": _tid(station, counters),
                "start_point": station,
                "end_point": dest,
                "priority": priority,
                "remaining_time": "None",
                "_phase": phase,
            }
        )

    for _ in range(int(n_random_pre)):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_pre")

    mode = burst_mode.lower().strip()
    hot = rng.choice(ALL_STATIONS)
    hub = rng.choice(ALL_DESTS)
    for _ in range(int(n_burst)):
        if mode in ("same_dropoff", "dropoff"):
            add(rng.choice(ALL_STATIONS), hub, "burst")
        elif mode in ("same_pickup", "pickup"):
            add(hot, rng.choice(ALL_DESTS), "burst")
        elif mode in ("dual", "pickup_dropoff"):
            add(hot, hub, "burst")
        else:
            raise ValueError(f"unknown burst_mode: {burst_mode}")

    for _ in range(int(n_random_post)):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_post")

    return rows


def write_task_csv(path: Path, rows: Sequence[dict]) -> None:
    path = Path(path)
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
            w.writerow({k: r.get(k, "") for k in w.fieldnames})


def write_scene_bundle(
    out_dir: Path,
    *,
    scene_id: str,
    obstacles: List,
    n_agvs: int,
    tasks: List[dict],
    meta_extra: Optional[dict] = None,
) -> dict:
    """Write position/task CSV + scene JSON consumable by run_m0."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = out_dir / f"{scene_id}_position.csv"
    task = out_dir / f"{scene_id}_task.csv"
    meta_path = out_dir / f"{scene_id}.json"
    write_position_csv(pos, n_agvs)
    write_task_csv(task, tasks)
    meta = {
        "id": scene_id,
        "extra_obstacles": obstacles,
        "n_agvs": n_agvs,
        "n_tasks": len(tasks),
        "position_csv": str(pos.resolve()),
        "task_csv": str(task.resolve()),
        **(meta_extra or {}),
    }
    meta_path.write_text(
        json_dumps(meta),
        encoding="utf-8",
    )
    return meta


def json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)
