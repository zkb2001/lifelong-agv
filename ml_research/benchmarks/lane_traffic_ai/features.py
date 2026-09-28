"""Map-centric feature channels for ParkArterySemaphoreNet (v5) + legacy v4."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .config import (
    ACTIVE_PATH_HORIZON,
    DIR_DELTA,
    GRID,
    N_MAP_CHANNELS,
    N_MAP_CHANNELS_V4,
    PLAY_HI,
    PLAY_LO,
)

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]
AssignedLike = Union[dict, Sequence[dict], Mapping[str, dict]]


def zone_id(x: int, y: int, grid: int = GRID) -> int:
    hx = (PLAY_LO + PLAY_HI) // 2
    hy = (PLAY_LO + PLAY_HI) // 2
    return (0 if y < hy else 2) + (0 if x < hx else 1)


def _attractor(cells: Sequence[Cell], walk: np.ndarray, H: int, W: int) -> np.ndarray:
    field = np.zeros((H, W), dtype=np.float32)
    if not cells:
        return field
    for cx, cy in cells:
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        for y in range(H):
            for x in range(W):
                if walk[y, x] < 0.5:
                    continue
                d = abs(x - cx) + abs(y - cy)
                field[y, x] = max(field[y, x], float(np.exp(-0.35 * d)))
    return field


def _as_task_list(assigned_tasks: Optional[AssignedLike]) -> list:
    if assigned_tasks is None:
        return []
    if isinstance(assigned_tasks, Mapping):
        return list(assigned_tasks.values())
    if isinstance(assigned_tasks, dict) and (
        "pickup_point" in assigned_tasks or "end_points" in assigned_tasks
    ):
        return [assigned_tasks]
    return list(assigned_tasks)


def snapshot_agv_positions(sim) -> Dict[str, Cell]:
    out: Dict[str, Cell] = {}
    states = getattr(sim, "agv_states", None) or {}
    for name, agv in states.items():
        st = agv["state"] if isinstance(agv, dict) else getattr(agv, "state", None)
        if st is None:
            continue
        out[str(name)] = (int(st[0]), int(st[1]))
    return out


def topology_degree_field(walkable: np.ndarray) -> np.ndarray:
    """Normalized degree / corridor hint in [0,1]."""
    H, W = walkable.shape
    out = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            if walkable[y, x] < 0.5:
                continue
            deg = 0
            for dx, dy in DIR_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] > 0.5:
                    deg += 1
            out[y, x] = min(1.0, deg / 4.0)
    return out


def project_active_paths(
    sim=None,
    *,
    path_cells: Optional[Sequence[Cell]] = None,
    horizon: int = ACTIVE_PATH_HORIZON,
    grid: int = GRID,
) -> np.ndarray:
    """2D mask of cells reserved by active AGV trajectories (next H steps)."""
    H = W = grid
    mask = np.zeros((H, W), dtype=np.float32)
    if path_cells is not None:
        for c in path_cells:
            x, y = int(c[0]), int(c[1])
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = 1.0
        return mask
    if sim is None:
        return mask
    t0 = int(getattr(sim, "time", 0) or 0)
    for agv in getattr(sim, "agvs", []) or []:
        tid = getattr(agv, "task_id", None)
        # recovery / escape holding is not "active committed work" for hard block
        if tid is not None and str(tid).startswith(("escape_", "park_", "recover_")):
            continue
        path = getattr(agv, "path", None) or []
        if not path:
            st = getattr(agv, "state", None)
            if st is not None and len(st) >= 2:
                x, y = int(st[0]), int(st[1])
                if 0 <= x < W and 0 <= y < H:
                    mask[y, x] = max(mask[y, x], 0.5)
            continue
        for i in range(t0, min(len(path), t0 + max(1, int(horizon)))):
            p = path[i]
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = 1.0
    return mask


def build_map_channels(
    sim=None,
    *,
    obstacles: Optional[Sequence] = None,
    stations: Optional[Set[Cell]] = None,
    agv_positions: Optional[Sequence[Cell]] = None,
    agv_poses: Optional[Mapping[str, Pose]] = None,
    prev_positions: Optional[Dict[str, Cell]] = None,
    goal_cells: Optional[Sequence[Cell]] = None,
    assigned_tasks: Optional[AssignedLike] = None,
    open_pickups: Optional[Sequence[Cell]] = None,
    construction: Optional[np.ndarray] = None,
    grid: int = GRID,
) -> np.ndarray:
    """Legacy v4 channels (C=11)."""
    H = W = grid
    ch = np.zeros((N_MAP_CHANNELS_V4, H, W), dtype=np.float32)

    obs: Set[Cell] = set()
    if obstacles is not None:
        for p in obstacles:
            obs.add((int(p[0]), int(p[1])))
    elif sim is not None:
        env = getattr(sim, "env", None)
        if env is not None and hasattr(env, "get_static_obstacles"):
            for p in env.get_static_obstacles() or []:
                obs.add((int(p[0]), int(p[1])))
        extra = getattr(sim, "extra_obstacles", None) or []
        for p in extra:
            obs.add((int(p[0]), int(p[1])))

    st_cells: Set[Cell] = set(stations or ())
    blocked = obs | st_cells
    for y in range(H):
        for x in range(W):
            if (x, y) in blocked or not (PLAY_LO <= x <= PLAY_HI and PLAY_LO <= y <= PLAY_HI):
                ch[0, y, x] = 1.0
            else:
                ch[1, y, x] = 1.0

    poses: Dict[str, Cell] = {}
    if agv_poses is not None:
        for n, p in agv_poses.items():
            poses[str(n)] = (int(p[0]), int(p[1]))
    elif agv_positions is not None:
        for i, c in enumerate(agv_positions):
            poses[str(i)] = (int(c[0]), int(c[1]))
    elif sim is not None:
        poses = snapshot_agv_positions(sim)

    for c in poses.values():
        x, y = c
        if 0 <= x < W and 0 <= y < H:
            ch[2, y, x] = 1.0

    if prev_positions:
        for name, cur in poses.items():
            prev = prev_positions.get(name)
            if not prev:
                continue
            dx, dy = cur[0] - prev[0], cur[1] - prev[1]
            for di, (adx, ady) in enumerate(DIR_DELTA):
                if (dx, dy) == (adx, ady):
                    x, y = cur
                    if 0 <= x < W and 0 <= y < H:
                        ch[3 + di, y, x] += 1.0
                    break

    tasks = _as_task_list(assigned_tasks)
    assigned_pickups: list[Cell] = []
    assigned_dropoffs: list[Cell] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        pk = t.get("pickup_point")
        if pk is not None:
            assigned_pickups.append((int(pk[0]), int(pk[1])))
        for e in t.get("end_points") or []:
            assigned_dropoffs.append((int(e[0]), int(e[1])))

    pool: list[Cell] = list(open_pickups or ())
    if not pool and goal_cells:
        pool = [(int(c[0]), int(c[1])) for c in goal_cells]
    if not pool and sim is not None:
        for q in (getattr(sim, "task_states", None) or {}).values():
            for t in q or []:
                pk = t.get("pickup_point") if isinstance(t, dict) else None
                if pk is not None:
                    pool.append((int(pk[0]), int(pk[1])))

    ch[7] = _attractor(pool, ch[1], H, W)
    ch[8] = _attractor(assigned_pickups, ch[1], H, W)
    ch[9] = _attractor(assigned_dropoffs, ch[1], H, W)
    if construction is not None:
        c = np.asarray(construction, dtype=np.float32)
        if c.ndim == 3:
            ch[10] = np.clip(c.max(axis=-1), 0, 1) * ch[1]
        elif c.shape == (H, W):
            ch[10] = np.clip(c, 0, 1) * ch[1]
    elif sim is not None:
        overlay = getattr(sim, "_lane_construction", None)
        if overlay is not None:
            c = np.asarray(overlay, dtype=np.float32)
            if c.ndim == 3:
                ch[10] = np.clip(c.max(axis=-1), 0, 1) * ch[1]
            elif c.shape == (H, W):
                ch[10] = np.clip(c, 0, 1) * ch[1]

    return ch


def build_park_artery_channels(
    sim=None,
    *,
    obstacles: Optional[Sequence] = None,
    stations: Optional[Set[Cell]] = None,
    failed_positions: Optional[Sequence[Cell]] = None,
    failed_unloads: Optional[Sequence[Cell]] = None,
    open_pickups: Optional[Sequence[Cell]] = None,
    occupied_parking: Optional[Sequence[Cell]] = None,
    active_path_mask: Optional[np.ndarray] = None,
    unload_blocked: Optional[Sequence[Cell]] = None,
    assigned_tasks: Optional[AssignedLike] = None,
    grid: int = GRID,
    horizon: int = ACTIVE_PATH_HORIZON,
) -> np.ndarray:
    """v5 channels (C=12, H, W).

    0 obstacle, 1 walkable, 2 AGV occ, 3 active-path, 4 failed pos,
    5 failed unload attractor, 6 pickup attractor, 7 open pool,
    8 unload pad blocked, 9 congestion, 10 topology, 11 occupied parking.
    """
    H = W = grid
    ch = np.zeros((N_MAP_CHANNELS, H, W), dtype=np.float32)

    obs: Set[Cell] = set()
    if obstacles is not None:
        for p in obstacles:
            obs.add((int(p[0]), int(p[1])))
    elif sim is not None:
        env = getattr(sim, "env", None)
        if env is not None and hasattr(env, "get_static_obstacles"):
            for p in env.get_static_obstacles() or []:
                obs.add((int(p[0]), int(p[1])))
        extra = getattr(sim, "extra_obstacles", None) or []
        for p in extra:
            obs.add((int(p[0]), int(p[1])))

    st_cells: Set[Cell] = set(stations or ())
    blocked = obs | st_cells
    for y in range(H):
        for x in range(W):
            if (x, y) in blocked or not (PLAY_LO <= x <= PLAY_HI and PLAY_LO <= y <= PLAY_HI):
                ch[0, y, x] = 1.0
            else:
                ch[1, y, x] = 1.0

    poses = snapshot_agv_positions(sim) if sim is not None else {}
    for c in poses.values():
        x, y = int(c[0]), int(c[1])
        if 0 <= x < W and 0 <= y < H:
            ch[2, y, x] = 1.0

    if active_path_mask is not None:
        ap = np.asarray(active_path_mask, dtype=np.float32)
        if ap.shape == (H, W):
            ch[3] = np.clip(ap, 0, 1) * ch[1]
    elif sim is not None:
        ch[3] = project_active_paths(sim, horizon=horizon, grid=grid) * ch[1]

    for c in failed_positions or ():
        x, y = int(c[0]), int(c[1])
        if 0 <= x < W and 0 <= y < H:
            ch[4, y, x] = 1.0

    ch[5] = _attractor(list(failed_unloads or ()), ch[1], H, W)

    tasks = _as_task_list(assigned_tasks)
    pickups: list[Cell] = list(open_pickups or ())
    for t in tasks:
        if not isinstance(t, dict):
            continue
        pk = t.get("pickup_point")
        if pk is not None:
            pickups.append((int(pk[0]), int(pk[1])))
    if not pickups and sim is not None:
        for q in (getattr(sim, "task_states", None) or {}).values():
            for t in q or []:
                pk = t.get("pickup_point") if isinstance(t, dict) else None
                if pk is not None:
                    pickups.append((int(pk[0]), int(pk[1])))
        for t in (getattr(sim, "surface_tasks", None) or {}).values():
            pk = t.get("pickup_point") if isinstance(t, dict) else None
            if pk is not None:
                pickups.append((int(pk[0]), int(pk[1])))

    ch[6] = _attractor(pickups, ch[1], H, W)
    ch[7] = _attractor(pickups, ch[1], H, W)

    pads = list(unload_blocked or ())
    if not pads and sim is not None and hasattr(sim, "_unload_pad_set"):
        try:
            pads = list(sim._unload_pad_set())
        except Exception:  # noqa: BLE001
            pads = []
    for c in pads:
        x, y = int(c[0]), int(c[1])
        if 0 <= x < W and 0 <= y < H:
            # mark blocked if occupied by any AGV
            ch[8, y, x] = 1.0 if ch[2, y, x] > 0.5 else 0.35

    # congestion = local AGV density blur
    occ = ch[2].copy()
    cong = np.zeros_like(occ)
    for y in range(H):
        for x in range(W):
            if ch[1, y, x] < 0.5:
                continue
            s = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= nx < W and 0 <= ny < H:
                        s += float(occ[ny, nx])
            cong[y, x] = min(1.0, s / 5.0)
    ch[9] = cong
    ch[10] = topology_degree_field(ch[1])

    for c in occupied_parking or ():
        x, y = int(c[0]), int(c[1])
        if 0 <= x < W and 0 <= y < H:
            ch[11, y, x] = 1.0
    if sim is not None:
        parked = getattr(sim, "_traffic_parked_cells", None) or set()
        for c in parked:
            x, y = int(c[0]), int(c[1])
            if 0 <= x < W and 0 <= y < H:
                ch[11, y, x] = 1.0

    return ch


def build_map_channels_ecbs(
    *,
    static: Set[Cell],
    stations: Set[Cell],
    pose: Mapping[str, Pose],
    assigned: Mapping[str, dict],
    open_pickups: Optional[Iterable[Cell]] = None,
    prev_positions: Optional[Dict[str, Cell]] = None,
    grid: int = GRID,
) -> np.ndarray:
    return build_map_channels(
        obstacles=list(static),
        stations=set(stations),
        agv_poses=pose,
        prev_positions=prev_positions,
        assigned_tasks=assigned,
        open_pickups=list(open_pickups or ()),
        grid=grid,
    )


def bfs_dist_field(
    walkable: np.ndarray,
    goals: Sequence[Cell],
    *,
    blocked: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Manhattan BFS distance field; unreachable = -1."""
    H, W = walkable.shape
    dist = np.full((H, W), -1.0, dtype=np.float32)
    from collections import deque

    q: deque = deque()
    for g in goals:
        x, y = int(g[0]), int(g[1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        if walkable[y, x] < 0.5:
            continue
        if blocked is not None and blocked[y, x] > 0.5:
            continue
        dist[y, x] = 0.0
        q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in DIR_DELTA:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if walkable[ny, nx] < 0.5:
                continue
            if blocked is not None and blocked[ny, nx] > 0.5:
                continue
            if dist[ny, nx] >= 0:
                continue
            dist[ny, nx] = dist[y, x] + 1.0
            q.append((nx, ny))
    return dist
