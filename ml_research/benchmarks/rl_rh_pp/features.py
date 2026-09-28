"""Feature tokens for RL-RH-PP style priority Transformer."""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from ml_research.benchmarks.common import enhanced_pair_cost, is_urgent_task

AGV_DIM = 12
TASK_DIM = 14
MAX_AGV = 16
MAX_TASK = 24


def _manh(a, b) -> float:
    return float(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def _xy(agv) -> Tuple[int, int]:
    st = agv.state
    return int(st[0]), int(st[1])


def total_backlog(sim) -> int:
    return sum(len(v) for v in (getattr(sim, "task_states", None) or {}).values())


def build_agv_tokens(sim, agvs: Sequence[Any]):
    feats = np.zeros((MAX_AGV, AGV_DIM), dtype=np.float32)
    mask = np.zeros((MAX_AGV,), dtype=np.bool_)
    names: List[str] = []
    bl = total_backlog(sim) / 40.0
    surf = len(getattr(sim, "surface_tasks", None) or {}) / float(MAX_TASK)
    for i, agv in enumerate(list(agvs)[:MAX_AGV]):
        x, y = _xy(agv)
        dens = fleet_density(sim, (x, y), 3)
        loaded = 1.0 if getattr(agv, "loaded", False) or getattr(agv, "carrying", False) else 0.0
        idle = 1.0 if not getattr(agv, "current_task", None) else 0.0
        feats[i] = [x / 20.0, y / 20.0, dens, loaded, idle, bl, surf, 0, 0, 0, 0, 0]
        mask[i] = True
        names.append(agv.name)
    return feats, mask, names


def teacher_m0_order(sim, free_agvs: Sequence[Any], tids: Sequence[str] | None = None) -> List[str]:
    """Greedy baseline priority: lower min pair cost first."""
    ids = list(
        tids if tids is not None else (sim.surface_tasks or {}).keys()
    )[:MAX_TASK]

    def _key(tid: str):
        info = sim.surface_tasks.get(tid)
        if not info:
            return 1e9
        costs = [
            enhanced_pair_cost(sim, a, tid, info, urgency_mode="m0_plus")
            for a in free_agvs
        ]
        return min(costs) if costs else 1e9

    return sorted(ids, key=_key)


def fleet_density(sim, cell, radius: int = 3) -> float:
    n = 0
    for a in getattr(sim, "agvs", []) or []:
        try:
            if _manh(_xy(a), cell) <= radius:
                n += 1
        except Exception:
            pass
    return float(n) / 8.0


def build_task_tokens(sim, free_agvs: Sequence[Any], task_ids: Sequence[str] | None = None):
    tids = list(task_ids if task_ids is not None else (sim.surface_tasks or {}).keys())[
        :MAX_TASK
    ]
    feats = np.zeros((MAX_TASK, TASK_DIM), dtype=np.float32)
    mask = np.zeros((MAX_TASK,), dtype=np.bool_)
    for i, tid in enumerate(tids):
        info = sim.surface_tasks[tid]
        pk = tuple(info["pickup_point"])
        ends = info.get("end_points") or (
            [tuple(info["unload_point"])] if info.get("unload_point") else [pk]
        )
        ek = tuple(ends[0])
        urgent = 1.0 if is_urgent_task(info) else 0.0
        rt = info.get("remaining_time")
        rt_n = float(rt) / 500.0 if rt is not None else 2.0
        before = info.get("numbers_before_urgent", -1)
        before_n = float(before) / 10.0 if before is not None and before >= 0 else 1.0
        dens_p = fleet_density(sim, pk, 3)
        dens_e = fleet_density(sim, ek, 3)
        min_c = 99.0
        for a in free_agvs:
            c = enhanced_pair_cost(sim, a, tid, info, urgency_mode="m0_plus")
            if c < min_c:
                min_c = float(c)
        min_c_n = min(min_c, 80.0) / 80.0
        trip = _manh(pk, ek) / 40.0
        pickup = info.get("pickup_name") or ""
        q_len = len((sim.task_states or {}).get(pickup, [])) / 20.0
        bl = total_backlog(sim) / 40.0
        feats[i] = [
            pk[0] / 20.0,
            pk[1] / 20.0,
            ek[0] / 20.0,
            ek[1] / 20.0,
            urgent,
            rt_n,
            before_n,
            dens_p,
            dens_e,
            min_c_n,
            trip,
            q_len,
            bl,
            0.0,
        ]
        mask[i] = True
    return feats, mask, tids


def teacher_task_order(sim, free_agvs: Sequence[Any]) -> List[str]:
    from ml_research.benchmarks.allocators import _task_priority_key

    return sorted(
        (sim.surface_tasks or {}).keys(),
        key=lambda tid: _task_priority_key(sim, tid, free_agvs),
    )
