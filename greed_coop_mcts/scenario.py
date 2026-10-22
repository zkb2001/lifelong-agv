"""Scenario I/O: map exports + simulation engine (no ECBS / Hier / A*)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from greed_coop_mcts.paths import ENGINE_PY, MAP_EXPORTS

Cell = Tuple[int, int]

TRAJ_HEADER = [
    "timestamp",
    "name",
    "X",
    "Y",
    "pitch",
    "loaded",
    "destination",
    "Emergency",
    "task-id",
]


def slot_id(n: int) -> str:
    return f"SH_custom_{n:02d}"


def load_engine(force_reload: bool = False):
    name = "greed_coop_mcts_engine"
    if force_reload and name in sys.modules:
        del sys.modules[name]
    if name in sys.modules:
        return sys.modules[name]
    if not ENGINE_PY.is_file():
        raise FileNotFoundError(f"missing engine: {ENGINE_PY}")
    spec = importlib.util.spec_from_file_location(name, ENGINE_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ENGINE_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_slot_meta(
    slot: int,
    *,
    max_sim_time: int = 200000,
    wall_timeout: float = 1800.0,
    n_tasks: int = 100,
    exports: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    root = Path(exports) if exports else MAP_EXPORTS
    sid = slot_id(slot)
    jpath = root / f"{sid}.json"
    task_csv = root / f"{sid}_task.csv"
    pos_csv = root / f"{sid}_position.csv"
    if not jpath.is_file() or not task_csv.is_file() or not pos_csv.is_file():
        return None
    data = json.loads(jpath.read_text(encoding="utf-8"))
    return {
        "id": sid,
        "slot": int(slot),
        "source_slot": sid,
        "map_shape": "custom_editor",
        "task_csv": str(task_csv.resolve()),
        "position_csv": str(pos_csv.resolve()),
        "map_json": str(jpath.resolve()),
        "extra_obstacles": data.get("extra_obstacles") or [],
        "n_tasks": int(n_tasks if n_tasks is not None else data.get("n_tasks") or 100),
        "n_agvs": int(data.get("n_agvs") or 0),
        "max_sim_time": int(max_sim_time),
        "wall_timeout": float(wall_timeout),
        "pickups": data.get("pickups") or [],
        "notes": data.get("notes", ""),
    }


def patch_extra_obstacles(env, extra_obstacles: List[Tuple[int, int]]) -> None:
    if not extra_obstacles:
        return
    orig = env.get_static_obstacles
    extra = [tuple(p) for p in extra_obstacles]

    def _combined():
        return list(orig()) + extra

    env.get_static_obstacles = _combined


def load_scenario(
    task_csv: Path,
    position_csv: Path,
    *,
    force_reload: bool = False,
):
    """Return (mod, env, agv_states, task_states, n_tasks)."""
    mod = load_engine(force_reload=force_reload)
    task_csv = Path(task_csv)
    position_csv = Path(position_csv)
    all_tasks = mod.get_task_list(str(task_csv))
    start_points, end_points, agv_list = mod.get_object_position(str(position_csv))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = mod.get_task_state(start_points, all_tasks, end_points)
    n_tasks = sum(len(v) for v in task_states.values())
    return mod, env, agv_states, task_states, n_tasks
