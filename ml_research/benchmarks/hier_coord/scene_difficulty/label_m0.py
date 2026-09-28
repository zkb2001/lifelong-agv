"""Label scenes easy/hard by whether M0 spacetime-A* can finish."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple  # noqa: F401 — Set used by wave label

import numpy as np

from ml_research.benchmarks.hier_coord.obs import GRID, N_HIER_CHANNELS, build_hier_map_obs
from ml_research.benchmarks.runner import run_m0
from ml_research.benchmarks.scenarios.generator import END_POINTS, START_POINTS

Cell = Tuple[int, int]


def stations_set() -> Set[Cell]:
    return {(int(x), int(y)) for _, x, y in START_POINTS + END_POINTS}


def static_from_obstacles(obstacles: Sequence) -> Set[Cell]:
    out: Set[Cell] = set()
    for o in obstacles:
        if isinstance(o, dict):
            out.add((int(o["x"]), int(o["y"])))
        else:
            out.add((int(o[0]), int(o[1])))
    return out


def default_pose_from_position_csv(position_csv: Path) -> Dict[str, Tuple[int, int, int]]:
    import csv

    pose: Dict[str, Tuple[int, int, int]] = {}
    with open(position_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("type") or "").lower() != "agv":
                continue
            name = str(row["name"])
            pitch = int(float(row["pitch"] or 90))
            pose[name] = (int(row["x"]), int(row["y"]), pitch)
    return pose


def enrich_task_row(t: dict) -> dict:
    """Attach pickup/drop geometry used by hier map obs / wave probes."""
    st = str(t.get("start_point") or "")
    pk = t.get("pickup_point")
    if pk is None:
        for name, x, y in START_POINTS:
            if name == st:
                pk = (x + 1, y) if x <= 10 else (x - 1, y)
                break
    ends = list(t.get("end_points") or [])
    dest = str(t.get("end_point") or t.get("destination") or "")
    if not ends and dest:
        for name, x, y in END_POINTS:
            if name == dest:
                ends = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                break
    return {
        **dict(t),
        "task_id": t.get("task_id"),
        "pickup_point": pk,
        "end_points": ends,
        "destination": dest,
        "start_point": st,
    }


def build_label_obs(
    *,
    obstacles: Sequence,
    position_csv: Path,
    task_rows: Sequence[dict],
    n_agvs: Optional[int] = None,
) -> np.ndarray:
    """Build hier map obs from near-horizon queues (no assigned yet)."""
    from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
        take_horizon_task_rows,
    )

    static = static_from_obstacles(obstacles)
    stations = stations_set()
    pose = default_pose_from_position_csv(Path(position_csv))
    na = int(n_agvs) if n_agvs is not None else max(1, len(pose))
    rows = take_horizon_task_rows(task_rows, na)
    queues: Dict[str, list] = {}
    for t in rows:
        info = enrich_task_row(t)
        st = str(info.get("start_point") or "")
        queues.setdefault(st, []).append(info)
    return build_hier_map_obs(
        static=static,
        stations=stations,
        pose=pose,
        assigned={},
        queues=queues,
    )


def _pickup_cell(task: dict) -> Optional[Tuple[int, int]]:
    info = enrich_task_row(task)
    pk = info.get("pickup_point")
    if pk is None:
        return None
    return (int(pk[0]), int(pk[1]))


def label_first_wave_st(
    *,
    obstacles: Sequence,
    position_csv: Path,
    task_rows: Sequence[dict],
    n_agvs: Optional[int] = None,
) -> Dict[str, Any]:
    """Label by whether prioritized spacetime A* can finish the first wave.

    Wave size = min(n_agvs, n_tasks). Steps:
    1) Greedy AGV→pickup assignment
    2) ST-A* to pickups (idle AGVs treated as static blockers — crowded fleets)
    3) ST-A* from pickups to first dropoff cell
    easy iff both stages succeed. Matches gate: astar vs ecbs for *this* wave.
    """
    from ml_research.benchmarks.coord_custom_ai.solve_ecbs import _prioritized_st_paths

    static0 = static_from_obstacles(obstacles) | stations_set()
    pose = default_pose_from_position_csv(Path(position_csv))
    na = int(n_agvs) if n_agvs is not None else len(pose)
    agents = sorted(pose.keys())[: max(1, na)]
    tasks = [enrich_task_row(t) for t in task_rows]
    k = min(len(agents), len(tasks))
    if k <= 0:
        return {
            "label": "easy",
            "label_int": 0,
            "wave_ok": True,
            "wave_k": 0,
            "label_rule": "wave_st_empty",
        }

    starts = {n: (int(pose[n][0]), int(pose[n][1])) for n in agents}
    used: Set[str] = set()
    goals: Dict[str, Tuple[int, int]] = {}
    movers: Set[str] = set()
    task_of: Dict[str, dict] = {}
    for t in tasks[:k]:
        pk = _pickup_cell(t)
        if pk is None:
            continue
        best = None
        best_d = 10**9
        for n in agents:
            if n in used:
                continue
            d = abs(starts[n][0] - pk[0]) + abs(starts[n][1] - pk[1])
            if d < best_d:
                best_d = d
                best = n
        if best is None:
            break
        used.add(best)
        goals[best] = pk
        movers.add(best)
        task_of[best] = t

    if not movers:
        return {
            "label": "hard",
            "label_int": 1,
            "wave_ok": False,
            "wave_k": 0,
            "label_rule": "wave_st_no_goals",
        }

    # Idle fleet bodies block corridors (important when n_agvs >> n_tasks).
    idle_cells = {starts[n] for n in agents if n not in movers}
    static = set(static0) | idle_cells

    paths_pk = _prioritized_st_paths(starts, goals, movers, static)
    if paths_pk is None:
        return {
            "label": "hard",
            "label_int": 1,
            "wave_ok": False,
            "wave_k": int(len(movers)),
            "label_rule": "wave_st_pickup_fail",
        }

    # Delivery stage: start at pickup, goal = first end_point.
    starts_d = {n: goals[n] for n in movers}
    for n in agents:
        if n not in movers:
            starts_d[n] = starts[n]
    goals_d: Dict[str, Tuple[int, int]] = {}
    for n, t in task_of.items():
        ends = t.get("end_points") or []
        if not ends:
            continue
        goals_d[n] = (int(ends[0][0]), int(ends[0][1]))
    if len(goals_d) < len(movers):
        return {
            "label": "hard",
            "label_int": 1,
            "wave_ok": False,
            "wave_k": int(len(movers)),
            "label_rule": "wave_st_no_drop",
        }

    paths_d = _prioritized_st_paths(starts_d, goals_d, movers, static)
    if paths_d is None:
        return {
            "label": "hard",
            "label_int": 1,
            "wave_ok": False,
            "wave_k": int(len(movers)),
            "label_rule": "wave_st_delivery_fail",
        }

    return {
        "label": "easy",
        "label_int": 0,
        "wave_ok": True,
        "wave_k": int(len(movers)),
        "label_rule": "wave_st_ok",
    }


def label_with_m0(
    *,
    task_csv: Path,
    position_csv: Path,
    obstacles: Sequence,
    scenario_tag: str,
    max_time: int = 1200,
    wall_timeout: float = 90.0,
    save_trajectory: bool = False,
) -> Dict[str, Any]:
    """Run M0 baseline; easy iff finishes without forced stop."""
    rep = run_m0(
        Path(task_csv),
        position_csv=Path(position_csv),
        max_time=int(max_time),
        wall_timeout=float(wall_timeout),
        scenario_tag=str(scenario_tag),
        extra_obstacles=list(obstacles),
        save_trajectory=bool(save_trajectory),
    )
    cr = float(rep.get("completion_ratio") or 0.0)
    forced = bool(rep.get("forced_stop"))
    easy = (cr >= 0.999) and (not forced)
    return {
        "label": "easy" if easy else "hard",
        "label_int": 0 if easy else 1,
        "completion_ratio": cr,
        "forced_stop": forced,
        "sim_time": int(rep.get("sim_time") or 0),
        "wall_seconds": float(rep.get("wall_seconds") or 0.0),
        "tasks_completed": int(rep.get("tasks_completed") or 0),
        "tasks_total": int(rep.get("tasks_total") or 0),
        "conflict_free": bool(rep.get("conflict_free")),
        "method": rep.get("method"),
    }
