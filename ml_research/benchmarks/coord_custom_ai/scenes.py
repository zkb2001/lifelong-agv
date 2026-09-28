"""Load custom map metas from map_editor exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml_research.map_editor.model import default_exports_dir, slot_id

Cell = Tuple[int, int]


def load_custom_meta(
    slot: int,
    *,
    max_sim_time: int = 4000,
    wall_timeout: float = 4000.0,
    n_tasks: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    exports = default_exports_dir()
    sid = slot_id(slot)
    jpath = exports / f"{sid}.json"
    if not jpath.exists():
        return None
    data = json.loads(jpath.read_text(encoding="utf-8"))
    task_csv = exports / f"{sid}_task.csv"
    pos_csv = exports / f"{sid}_position.csv"
    if not task_csv.exists() or not pos_csv.exists():
        return None
    nt = int(n_tasks if n_tasks is not None else data.get("n_tasks") or 100)
    return {
        "id": sid,
        "slot": slot,
        "source_slot": sid,
        "map_shape": "custom_editor",
        "task_csv": str(task_csv.resolve()),
        "position_csv": str(pos_csv.resolve()),
        "map_json": str(jpath.resolve()),
        "extra_obstacles": data.get("extra_obstacles") or [],
        "n_tasks": nt,
        "n_agvs": int(data.get("n_agvs") or 0),
        "max_sim_time": int(max_sim_time),
        "wall_timeout": float(wall_timeout),
        "pickups": data.get("pickups") or [],
        "notes": data.get("notes", ""),
    }


def load_custom_metas(
    slots: Optional[List[int]] = None,
    *,
    max_sim_time: int = 4000,
    wall_timeout: float = 4000.0,
    n_tasks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    slots = slots if slots is not None else list(range(1, 21))
    out: List[Dict[str, Any]] = []
    for n in slots:
        meta = load_custom_meta(
            n, max_sim_time=max_sim_time, wall_timeout=wall_timeout, n_tasks=n_tasks
        )
        if meta:
            out.append(meta)
    return out


def pickup_map_from_meta(meta: Dict[str, Any]) -> Dict[str, Cell]:
    m: Dict[str, Cell] = {}
    for p in meta.get("pickups") or []:
        name = str(p.get("name") or "")
        if name:
            m[name] = (int(p["x"]), int(p["y"]))
    return m


def task_start_points(task_csv: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(task_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = str(row.get("task_id") or "")
            sp = str(row.get("start_point") or "")
            if tid and sp:
                out[tid] = sp
    return out
