"""WaveNet features — score which AGVs should enter the current joint wave."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

FEAT_DIM = 12
FEAT_NAMES = (
    "eta_n",
    "work_n",
    "local_obs_n",
    "conflict_proxy_n",
    "urgency",
    "is_seed",
    "map_hardness",
    "k_budget_n",
    "n_cand_n",
    "dist_rank_n",
    "pair_overlap_n",
    "dest_share",  # fraction of wave candidates sharing this destination
)


@dataclass
class WaveAgentContext:
    agv: str
    task: dict
    eta: float
    work_count: int
    local_obs: float
    conflict_proxy: float
    urgency: float
    is_seed: bool
    map_hardness: float
    k_budget: int
    n_cand: int
    dist_rank: float  # 0 = nearest among cand
    pair_overlap: float
    dest_share: float = 0.0
    was_idle: bool = False  # legacy; prefer dest_share


def _local_obstacle_density(
    cell: Cell,
    static: Set[Cell],
    *,
    radius: int = 2,
) -> float:
    x0, y0 = cell
    total = 0
    blocked = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = x0 + dx, y0 + dy
            if x < 1 or x > 20 or y < 1 or y > 20:
                continue
            total += 1
            if (x, y) in static:
                blocked += 1
    return float(blocked) / float(max(1, total))


def _bbox_overlap(a0: Cell, a1: Cell, b0: Cell, b1: Cell) -> float:
    """Cheap corridor-overlap proxy in [0, 1]."""
    ax0, ax1 = min(a0[0], a1[0]), max(a0[0], a1[0])
    ay0, ay1 = min(a0[1], a1[1]), max(a0[1], a1[1])
    bx0, bx1 = min(b0[0], b1[0]), max(b0[0], b1[0])
    by0, by1 = min(b0[1], b1[1]), max(b0[1], b1[1])
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    if ix0 > ix1 or iy0 > iy1:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    area_a = max(1, (ax1 - ax0 + 1) * (ay1 - ay0 + 1))
    area_b = max(1, (bx1 - bx0 + 1) * (by1 - by0 + 1))
    return float(inter) / float(max(area_a, area_b))


def build_wave_contexts(
    assigned: Dict[str, dict],
    pose: Dict[str, Pose],
    static: Set[Cell],
    bfs_len: Callable[[Cell, Cell, Set[Cell]], int],
    work_count: Dict[str, int],
    *,
    map_hardness: float,
    k_budget: int,
    seed_agvs: Optional[Set[str]] = None,
    grid_scale: float = 40.0,
) -> List[WaveAgentContext]:
    seed_agvs = seed_agvs or set()
    if not assigned:
        return []
    etas: Dict[str, float] = {}
    for a, task in assigned.items():
        pk = tuple(task["pickup_point"])
        etas[a] = float(bfs_len(pose[a][:2], pk, static))
    ranked = sorted(etas, key=lambda a: (etas[a], a))
    rank_of = {a: i / max(1, len(ranked) - 1) for i, a in enumerate(ranked)}

    overlap: Dict[str, float] = {a: 0.0 for a in assigned}
    items = list(assigned.items())
    for i, (a, ta) in enumerate(items):
        pa, ga = pose[a][:2], tuple(ta["pickup_point"])
        for b, tb in items[i + 1 :]:
            pb, gb = pose[b][:2], tuple(tb["pickup_point"])
            ov = _bbox_overlap(pa, ga, pb, gb)
            overlap[a] += ov
            overlap[b] += ov
    n_other = max(1, len(assigned) - 1)
    for a in overlap:
        overlap[a] /= float(n_other)

    # also fold dropoff-corridor overlap into conflict proxy
    drop_ov: Dict[str, float] = {a: 0.0 for a in assigned}
    for i, (a, ta) in enumerate(items):
        ga = tuple(ta["pickup_point"])
        da = tuple((ta.get("end_points") or [ga])[0])
        for b, tb in items[i + 1 :]:
            gb = tuple(tb["pickup_point"])
            db = tuple((tb.get("end_points") or [gb])[0])
            ov = _bbox_overlap(ga, da, gb, db)
            drop_ov[a] += ov
            drop_ov[b] += ov
    for a in drop_ov:
        drop_ov[a] /= float(n_other)

    from collections import Counter

    dest_counts = Counter(str(t.get("destination") or "") for t in assigned.values())
    n_cand = len(assigned)

    out: List[WaveAgentContext] = []
    for a, task in assigned.items():
        urg = 1.0 if str(task.get("priority") or "").lower() in (
            "urgent",
            "emergency",
            "high",
        ) else 0.0
        conflict = 0.55 * overlap[a] + 0.45 * drop_ov[a]
        dest = str(task.get("destination") or "")
        share = float(dest_counts[dest]) / float(max(1, n_cand))
        out.append(
            WaveAgentContext(
                agv=a,
                task=task,
                eta=etas[a],
                work_count=int(work_count.get(a, 0)),
                local_obs=_local_obstacle_density(pose[a][:2], static),
                conflict_proxy=conflict,
                urgency=urg,
                is_seed=a in seed_agvs,
                map_hardness=float(map_hardness),
                k_budget=int(k_budget),
                n_cand=n_cand,
                dist_rank=float(rank_of.get(a, 0.5)),
                pair_overlap=conflict,
                dest_share=share,
                was_idle=int(work_count.get(a, 0)) == 0,
            )
        )
    # silence unused
    _ = grid_scale
    return out


def build_wave_features(ctx: WaveAgentContext, *, grid_scale: float = 40.0) -> np.ndarray:
    g = max(1.0, float(grid_scale))
    work_n = min(1.0, float(ctx.work_count) / 20.0)
    feat = np.array(
        [
            min(1.0, float(ctx.eta) / g),
            work_n,
            float(ctx.local_obs),
            float(ctx.conflict_proxy),
            float(ctx.urgency),
            1.0 if ctx.is_seed else 0.0,
            float(ctx.map_hardness),
            min(1.0, float(ctx.k_budget) / 8.0),
            min(1.0, float(ctx.n_cand) / 8.0),
            float(ctx.dist_rank),
            float(ctx.pair_overlap),
            float(getattr(ctx, "dest_share", 1.0 if ctx.was_idle else 0.0)),
        ],
        dtype=np.float32,
    )
    assert feat.shape[0] == FEAT_DIM
    return feat
