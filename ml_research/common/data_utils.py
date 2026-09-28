"""Load map / trajectories and import expert planner without editing main copy.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .paths import GRID_SIZE, MAIN_COPY, POSITION_CSV, TRAJECTORIES


def load_main_copy(force_reload: bool = False):
    """Import `simulation/engine.py` as a module (read-only usage)."""
    name = "mioverse_main_copy"
    if force_reload and name in sys.modules:
        del sys.modules[name]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MAIN_COPY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {MAIN_COPY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_static_map(path: Path = POSITION_CSV) -> Dict[str, Any]:
    start_points, end_points, obstacles = {}, {}, []
    agvs = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            typ, name, x, y = parts[0], parts[1], int(parts[2]), int(parts[3])
            pitch = int(parts[4]) if len(parts) > 4 and parts[4] else 0
            if typ == "start_point":
                start_points[name] = (x, y)
                obstacles.append((x, y))
            elif typ == "end_point":
                end_points[name] = (x, y)
                obstacles.append((x, y))
            elif typ == "agv":
                agvs[name] = (x, y, pitch)
    return {
        "start_points": start_points,
        "end_points": end_points,
        "obstacles": obstacles,
        "agvs": agvs,
        "grid_size": GRID_SIZE,
    }


def load_trajectory(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop duplicate header rows if any
    df = df[df["timestamp"].astype(str) != "timestamp"].copy()
    df["timestamp"] = df["timestamp"].astype(int)
    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)
    df["pitch"] = df["pitch"].astype(int)
    return df.sort_values(["timestamp", "name"]).reset_index(drop=True)


def available_trajectories() -> List[Path]:
    return [p for p in TRAJECTORIES if p.exists() and p.stat().st_size > 100]


def pitch_to_dir(pitch: int) -> int:
    """Map pitch degrees to discrete direction index: 0E,1N,2W,3S."""
    p = pitch % 360
    return {0: 0, 90: 1, 180: 2, 270: 3}.get(p, 0)


def action_from_transition(
    x0: int, y0: int, p0: int, x1: int, y1: int, p1: int
) -> int:
    """
    Discrete actions aligned with expert AGV kinematics:
      0 wait, 1 forward, 2 turn_left(+90), 3 turn_right(-90), 4 turn_180
    """
    if (x0, y0) == (x1, y1):
        dp = (p1 - p0) % 360
        if dp == 0:
            return 0
        if dp == 90:
            return 2
        if dp == 270:
            return 3
        if dp == 180:
            return 4
        return 0
    # moved: treat as forward (possibly after prior turn in same step — rare in logs)
    return 1


ACTION_NAMES = ["wait", "forward", "turn_left", "turn_right", "turn_180"]


def build_occupancy(
    frame: pd.DataFrame, grid_size: int = GRID_SIZE
) -> np.ndarray:
    occ = np.zeros((grid_size + 1, grid_size + 1), dtype=np.float32)
    for _, r in frame.iterrows():
        occ[int(r["Y"]), int(r["X"])] = 1.0
    return occ


def local_patch(
    occ: np.ndarray,
    x: int,
    y: int,
    radius: int,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    size = 2 * radius + 1
    patch = np.ones((size, size), dtype=np.float32)  # 1 = blocked outside
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = x + dx, y + dy
            if 1 <= xx <= grid_size and 1 <= yy <= grid_size:
                patch[dy + radius, dx + radius] = occ[yy, xx]
            # else keep 1
    return patch
