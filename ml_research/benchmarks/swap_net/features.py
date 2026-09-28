"""SwapNet feature builder — includes Emergency/Urgent SLA signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ml_research.benchmarks.common import is_urgent_task

FEAT_DIM = 16
FEAT_NAMES = (
    "far_eta_n",
    "near_eta_n",
    "eta_gap_n",
    "far_dist_n",
    "near_dist_n",
    "task_urgent",
    "remaining_time_n",
    "sla_slack_n",
    "urgent_gain_n",
    "near_idle",
    "near_holds_urgent",
    "queue_depth_n",
    "margin_ratio",
    "prefer_near_for_urgent",
    "far_misses_sla",
    "near_meets_sla",
)


@dataclass
class SwapContext:
    far_eta: float
    near_eta: float
    margin: float
    far_dist: float
    near_dist: float
    task_info: dict
    queue_depth: int = 0
    near_has_task: bool = False
    near_task_info: Optional[dict] = None
    grid_scale: float = 40.0


def _remaining_time(task_info: dict) -> Optional[float]:
    rt = task_info.get("remaining_time")
    if rt is None or rt == "":
        return None
    try:
        return float(rt)
    except (TypeError, ValueError):
        return None


def build_swap_features(ctx: SwapContext) -> np.ndarray:
    """16-dim vector for SwapNet v3 (urgent / SLA aware)."""
    gs = max(1.0, float(ctx.grid_scale))
    margin = max(1.0, float(ctx.margin))
    gap = float(ctx.far_eta - ctx.near_eta)
    urgent = is_urgent_task(ctx.task_info)
    rt = _remaining_time(ctx.task_info)
    rt_n = 0.0
    sla_slack_n = 0.0
    urgent_gain_n = 0.0
    far_misses = 0.0
    near_meets = 0.0
    if urgent and rt is not None and rt > 0:
        rt_n = min(1.0, rt / 300.0)
        sla_slack_n = max(-1.0, min(1.0, (rt - ctx.near_eta) / rt))
        urgent_gain_n = min(1.0, gap / rt)
        far_misses = 1.0 if ctx.far_eta > rt else 0.0
        near_meets = 1.0 if ctx.near_eta <= rt else 0.0
    else:
        urgent_gain_n = min(1.0, gap / gs)

    near_urgent = (
        is_urgent_task(ctx.near_task_info) if ctx.near_task_info else False
    )
    return np.array(
        [
            ctx.far_eta / gs,
            ctx.near_eta / gs,
            gap / gs,
            ctx.far_dist / gs,
            ctx.near_dist / gs,
            1.0 if urgent else 0.0,
            rt_n,
            sla_slack_n,
            urgent_gain_n,
            1.0 if not ctx.near_has_task else 0.0,
            1.0 if near_urgent else 0.0,
            min(1.0, float(ctx.queue_depth) / 8.0),
            gap / margin,
            1.0 if urgent and gap >= margin else 0.0,
            far_misses,
            near_meets,
        ],
        dtype=np.float32,
    )


def manhattan(a, b) -> float:
    return float(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def eta_to_pickup(sim: Any, agv: Any, pickup_xy) -> float:
    if hasattr(sim, "manhattan_distance"):
        return float(sim.manhattan_distance(agv.state[:2], pickup_xy))
    return manhattan(agv.state[:2], pickup_xy)


def agv_is_loaded(sim: Any, agv: Any) -> bool:
    t = int(getattr(sim, "time", 0))
    if hasattr(sim, "_step_at_time"):
        row = sim._step_at_time(agv, t)
        if row:
            return str(row.get("loaded", "")).lower() in ("true", "1", "yes")
    return bool(getattr(agv, "load_state", {}).get("loaded"))


def build_context_from_pair(
    sim: Any,
    far_agv: Any,
    near_agv: Any,
    task_info: dict,
    *,
    margin: float,
    pickup_xy,
) -> SwapContext:
    far_eta = eta_to_pickup(sim, far_agv, pickup_xy)
    near_eta = eta_to_pickup(sim, near_agv, pickup_xy)
    station = task_info.get("pickup_name") or task_info.get("start_point")
    qdepth = len(sim.task_states.get(station, [])) if station else 0
    near_tid = getattr(near_agv, "task_id", None)
    near_info = None
    if near_tid:
        near_info = task_info if str(near_tid) == str(task_info.get("task_id")) else None
        if near_info is None:
            for q in sim.task_states.values():
                for t in q:
                    if str(t.get("task_id")) == str(near_tid):
                        near_info = t
                        break
    return SwapContext(
        far_eta=far_eta,
        near_eta=near_eta,
        margin=margin,
        far_dist=far_eta,
        near_dist=near_eta,
        task_info=task_info,
        queue_depth=qdepth,
        near_has_task=bool(near_tid),
        near_task_info=near_info,
    )
