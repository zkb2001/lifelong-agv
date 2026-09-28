"""Hierarchical AI observation: real map + AGVs + tasks as multi-channel grid."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

GRID = 21
N_HIER_CHANNELS = 8
HIER_CHANNEL_NAMES = (
    "obstacle",
    "walkable",
    "station",
    "agv",
    "assigned_pickup",
    "assigned_dropoff",
    "queue_pickup",
    "hotspot_dest",
)


def _paint_disk(field: np.ndarray, cells: Iterable[Cell], *, radius: int = 0, value: float = 1.0) -> None:
    H, W = field.shape
    for c in cells:
        x, y = int(c[0]), int(c[1])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    field[ny, nx] = max(field[ny, nx], value)


def _attractor(field: np.ndarray, cells: Sequence[Cell], walk: np.ndarray, *, decay: float = 0.35) -> None:
    H, W = field.shape
    for cx, cy in cells:
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        for y in range(H):
            for x in range(W):
                if walk[y, x] < 0.5:
                    continue
                d = abs(x - cx) + abs(y - cy)
                field[y, x] = max(field[y, x], float(np.exp(-decay * d)))


def build_hier_map_obs(
    *,
    static: Set[Cell],
    stations: Set[Cell],
    pose: Mapping[str, Pose],
    assigned: Optional[Mapping[str, dict]] = None,
    queues: Optional[Mapping[str, Sequence[dict]]] = None,
    grid: int = GRID,
) -> np.ndarray:
    """Return float32 (C, H, W) observation for hierarchical nets.

    Channels
    --------
    0 obstacle (static)
    1 walkable free cells
    2 station cells
    3 AGV occupancy
    4 assigned pickup attractor
    5 assigned dropoff attractor
    6 remaining queue pickup attractor
    7 dominant assigned destination blob (hotspot cue)
    """
    H = W = int(grid)
    ch = np.zeros((N_HIER_CHANNELS, H, W), dtype=np.float32)
    blocked = set(static) | set(stations)
    for y in range(H):
        for x in range(W):
            if not (1 <= x <= 20 and 1 <= y <= 20):
                ch[0, y, x] = 1.0
            elif (x, y) in static:
                ch[0, y, x] = 1.0
            elif (x, y) in stations:
                ch[2, y, x] = 1.0
            else:
                ch[1, y, x] = 1.0

    for p in pose.values():
        x, y = int(p[0]), int(p[1])
        if 0 <= x < W and 0 <= y < H:
            ch[3, y, x] = 1.0

    assigned = assigned or {}
    picks: list[Cell] = []
    drops: list[Cell] = []
    dest_counts: Dict[str, int] = {}
    dest_cells: Dict[str, Cell] = {}
    for t in assigned.values():
        pk = t.get("pickup_point")
        if pk is not None:
            picks.append((int(pk[0]), int(pk[1])))
        eps = t.get("end_points") or []
        for e in eps:
            drops.append((int(e[0]), int(e[1])))
        key = str(t.get("destination") or "").strip()
        if not key and eps:
            e0 = eps[0]
            key = f"cell:{int(e0[0])},{int(e0[1])}"
            dest_cells[key] = (int(e0[0]), int(e0[1]))
        elif key and eps:
            dest_cells.setdefault(key, (int(eps[0][0]), int(eps[0][1])))
        if key:
            dest_counts[key] = dest_counts.get(key, 0) + 1

    q_picks: list[Cell] = []
    for q in (queues or {}).values():
        for t in q or []:
            pk = t.get("pickup_point") if isinstance(t, dict) else None
            if pk is not None:
                q_picks.append((int(pk[0]), int(pk[1])))

    _attractor(ch[4], picks, ch[1])
    _attractor(ch[5], drops, ch[1])
    _attractor(ch[6], q_picks[:24], ch[1])

    if dest_counts:
        top = max(dest_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        cell = dest_cells.get(top)
        if cell is not None:
            _paint_disk(ch[7], [cell], radius=2, value=1.0)
            _attractor(ch[7], [cell], ch[1], decay=0.25)

    # silence unused
    _ = blocked
    return ch
