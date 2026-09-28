"""Hand-crafted unidirectional / signal teachers (warehouse traffic rules).

Inspired by unidirectional guide-path literature and Neural Traffic Rules IL stage.
Also labels yield/parking, artery/side cost zones, no-wait mouths, and sparse construction.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .config import BLOCK, DIR_DELTA, GRID, N_DIRS, N_ZONES, PLAY_HI, PLAY_LO
from .features import zone_id


def _degree(walkable: np.ndarray, x: int, y: int) -> int:
    H, W = walkable.shape
    deg = 0
    for dx, dy in DIR_DELTA:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] > 0.5:
            deg += 1
    return deg


def teacher_direction_field(walkable: np.ndarray, grid: int = GRID) -> np.ndarray:
    """(H,W) int in {0..3} preferred direction; -1 if not walkable.

    Classic warehouse loop:
      - even rows prefer East (0), odd rows prefer West (2)
      - on even columns reinforce North (1), odd columns South (3) when row rule conflicts
        at junctions use column rule to create circulating flow.
    """
    H = W = grid
    pref = np.full((H, W), -1, dtype=np.int64)
    for y in range(PLAY_LO, PLAY_HI + 1):
        for x in range(PLAY_LO, PLAY_HI + 1):
            if walkable[y, x] < 0.5:
                continue
            # primary: row one-way
            d = 0 if (y % 2 == 0) else 2
            # at "junction-like" cells (multi-neighbor), blend with column one-way
            deg = _degree(walkable, x, y)
            if deg >= 3:
                d = 1 if (x % 2 == 0) else 3
            # if preferred neighbor blocked, pick any free neighbor
            dx, dy = DIR_DELTA[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H and walkable[ny, nx] > 0.5):
                for alt in range(N_DIRS):
                    adx, ady = DIR_DELTA[alt]
                    ax, ay = x + adx, y + ady
                    if 0 <= ax < W and 0 <= ay < H and walkable[ay, ax] > 0.5:
                        d = alt
                        break
            pref[y, x] = d
    return pref


def teacher_phase_field(walkable: np.ndarray, grid: int = GRID, block: int = BLOCK) -> np.ndarray:
    """(Jy, Jx) phase: 0 = NS green, 1 = EW green. Checkerboard by block."""
    jy = max(1, grid // block)
    jx = max(1, grid // block)
    phase = np.zeros((jy, jx), dtype=np.int64)
    for j in range(jy):
        for i in range(jx):
            phase[j, i] = (i + j) % 2
    return phase


def teacher_zone_field(grid: int = GRID) -> np.ndarray:
    z = np.zeros((grid, grid), dtype=np.int64)
    for y in range(grid):
        for x in range(grid):
            z[y, x] = zone_id(x, y) if PLAY_LO <= x <= PLAY_HI and PLAY_LO <= y <= PLAY_HI else 0
    return z


def teacher_artery_field(walkable: np.ndarray, grid: int = GRID, block: int = BLOCK) -> np.ndarray:
    """Continuous [0,1] cost-zone field: artery≈1 (main corridor), side≈0.

    Block-aligned corridors + high-degree junctions form the main network;
    interior low-degree cells are side roads (分流 via higher step cost).
    """
    H = W = grid
    artery = np.zeros((H, W), dtype=np.float32)
    for y in range(PLAY_LO, PLAY_HI + 1):
        for x in range(PLAY_LO, PLAY_HI + 1):
            if walkable[y, x] < 0.5:
                continue
            on_grid = (x % block == 0) or (y % block == 0)
            deg = _degree(walkable, x, y)
            if on_grid and deg >= 2:
                artery[y, x] = 1.0
            elif deg >= 3:
                artery[y, x] = 0.85
            elif on_grid:
                artery[y, x] = 0.7
            elif deg == 2:
                artery[y, x] = 0.25
            else:
                artery[y, x] = 0.1
    return artery


def teacher_yield_field(
    walkable: np.ndarray,
    artery: Optional[np.ndarray] = None,
    grid: int = GRID,
) -> np.ndarray:
    """Yield / parking mask: prefer waiting on side pockets beside arteries.

    Junction centers (deg≥3) are excluded — those go to nowait.
    """
    H = W = grid
    if artery is None:
        artery = teacher_artery_field(walkable, grid=grid)
    yld = np.zeros((H, W), dtype=np.float32)
    for y in range(PLAY_LO, PLAY_HI + 1):
        for x in range(PLAY_LO, PLAY_HI + 1):
            if walkable[y, x] < 0.5:
                continue
            deg = _degree(walkable, x, y)
            if deg >= 3:
                continue
            near_artery = False
            for dx, dy in DIR_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and artery[ny, nx] >= 0.7:
                    near_artery = True
                    break
            # parking / yield: low-artery side cell next to main corridor
            if near_artery and artery[y, x] < 0.55 and deg <= 2:
                yld[y, x] = 1.0
            elif deg == 1 and near_artery:
                yld[y, x] = 1.0
    return yld


def teacher_nowait_field(
    walkable: np.ndarray,
    channels: Optional[np.ndarray] = None,
    grid: int = GRID,
) -> np.ndarray:
    """No-long-wait mask: junction centers + station mouths (assigned attractors)."""
    H = W = grid
    nowait = np.zeros((H, W), dtype=np.float32)
    for y in range(PLAY_LO, PLAY_HI + 1):
        for x in range(PLAY_LO, PLAY_HI + 1):
            if walkable[y, x] < 0.5:
                continue
            if _degree(walkable, x, y) >= 3:
                nowait[y, x] = 1.0
    if channels is not None and channels.shape[0] >= 10:
        # station mouths: strong assigned pickup/dropoff attractor
        mouth = np.maximum(channels[8], channels[9])
        nowait = np.maximum(nowait, (mouth >= 0.85).astype(np.float32) * walkable)
    return nowait


def teacher_construction_field(
    walkable: np.ndarray,
    grid: int = GRID,
    *,
    density: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """(H,W,4) sparse closed-edge mask for paper-style construction ablation.

    density=0 → all open. When density>0, randomly close directed edges on walkable cells.
    """
    H = W = grid
    closed = np.zeros((H, W, N_DIRS), dtype=np.float32)
    if density <= 0:
        return closed
    rng = rng or np.random.default_rng(seed)
    for y in range(PLAY_LO, PLAY_HI + 1):
        for x in range(PLAY_LO, PLAY_HI + 1):
            if walkable[y, x] < 0.5:
                continue
            for d, (dx, dy) in enumerate(DIR_DELTA):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H and walkable[ny, nx] > 0.5):
                    continue
                if rng.random() < density:
                    closed[y, x, d] = 1.0
    return closed


def teacher_from_channels(
    channels: np.ndarray,
    *,
    construction_density: float = 0.0,
    construction: Optional[np.ndarray] = None,
    urgency: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Build teacher labels from map channels (uses walkable ch1)."""
    walk = channels[1]
    artery = teacher_artery_field(walk)
    if construction is None:
        construction = teacher_construction_field(
            walk, density=construction_density, rng=rng
        )
    # Prefer map-channel construction hint if present and non-zero
    if channels.shape[0] >= 11 and float(channels[10].sum()) > 0:
        # expand cell hint to all outgoing edges from that cell
        hint = channels[10]
        for y in range(hint.shape[0]):
            for x in range(hint.shape[1]):
                if hint[y, x] > 0.5:
                    construction[y, x, :] = np.maximum(construction[y, x, :], 1.0)
    return {
        "dir": teacher_direction_field(walk),
        "phase": teacher_phase_field(walk),
        "zone": teacher_zone_field(),
        "yield": teacher_yield_field(walk, artery),
        "artery": artery,
        "nowait": teacher_nowait_field(walk, channels),
        "construction": construction.astype(np.float32),
        "urgency": float(urgency),
    }


def teacher_parking_field(
    walkable: np.ndarray,
    artery: Optional[np.ndarray] = None,
    *,
    active_path: Optional[np.ndarray] = None,
    unload_blocked: Optional[np.ndarray] = None,
    grid: int = GRID,
) -> np.ndarray:
    """Independent parking mask (not merged into yield).

    Prefer low-degree pockets beside artery; exclude junctions, active paths,
    and unload pad cells.
    """
    H = W = grid
    if artery is None:
        artery = teacher_artery_field(walkable, grid=grid)
    park = np.zeros((H, W), dtype=np.float32)
    for y in range(PLAY_LO, PLAY_HI + 1):
        for x in range(PLAY_LO, PLAY_HI + 1):
            if walkable[y, x] < 0.5:
                continue
            if active_path is not None and float(active_path[y, x]) > 0.5:
                continue
            if unload_blocked is not None and float(unload_blocked[y, x]) > 0.5:
                continue
            deg = _degree(walkable, x, y)
            if deg >= 3:
                continue
            near_artery = False
            for dx, dy in DIR_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and artery[ny, nx] >= 0.7:
                    near_artery = True
                    break
            if near_artery and artery[y, x] < 0.55 and deg <= 2:
                park[y, x] = 1.0
            elif deg == 1 and near_artery:
                park[y, x] = 1.0
    return park


def teacher_connectivity_field(
    walkable: np.ndarray,
    artery: np.ndarray,
    goals: Sequence[Tuple[int, int]],
    *,
    active_path: Optional[np.ndarray] = None,
    artery_thresh: float = 0.55,
    grid: int = GRID,
    parking: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Connectivity toward goals on walkable grid (blocked by active paths).

    Parking cells are included: agents recover from parking, so excluding them
    makes semaphore ``c_here`` always 0 and never releases.
    ``artery`` is kept for API compatibility (soft preference unused in BFS).
    """
    from .features import bfs_dist_field

    del artery_thresh  # reserved
    H = W = int(grid)
    walk_conn = (np.asarray(walkable, dtype=np.float32) > 0.5).astype(np.float32)
    if parking is not None:
        # ensure parking pockets stay open even if walk mask glitches
        walk_conn = np.maximum(walk_conn, (np.asarray(parking) > 0.5).astype(np.float32))
    # Snap station / blocked goals onto walkable pads (all maps).
    snapped: list[Tuple[int, int]] = []
    for g in goals or ():
        gx, gy = int(g[0]), int(g[1])
        if 0 <= gx < W and 0 <= gy < H and walk_conn[gy, gx] > 0.5:
            snapped.append((gx, gy))
            continue
        found = None
        for r in range(1, 6):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) + abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < W and 0 <= ny < H and walk_conn[ny, nx] > 0.5:
                        found = (nx, ny)
                        break
                if found:
                    break
            if found:
                break
        if found:
            snapped.append(found)
    _ = artery  # API compat
    blocked = None
    if active_path is not None:
        blocked = (np.asarray(active_path) > 0.5).astype(np.float32)
    dist = bfs_dist_field(walk_conn, snapped, blocked=blocked)
    conn = np.zeros((H, W), dtype=np.float32)
    finite = dist >= 0
    if not np.any(finite):
        return conn
    dmax = float(np.max(dist[finite])) + 1e-6
    conn[finite] = 1.0 - (dist[finite] / dmax)
    conn *= (walkable > 0.5).astype(np.float32)
    return conn


def teacher_semaphore_label(
    *,
    connectivity: np.ndarray,
    pos: Tuple[int, int],
    goal: Tuple[int, int],
    pad_free: float,
    path_clear: float,
    congestion: float = 0.0,
) -> float:
    """Binary-ish teacher: can activate from pos toward goal without interference."""
    x, y = int(pos[0]), int(pos[1])
    gx, gy = int(goal[0]), int(goal[1])
    H, W = connectivity.shape
    if not (0 <= x < W and 0 <= y < H and 0 <= gx < W and 0 <= gy < H):
        return 0.0
    c_here = float(connectivity[y, x])
    c_goal = float(connectivity[gy, gx])
    if c_here < 0.15 or c_goal < 0.15:
        return 0.0
    if float(path_clear) < 0.5:
        return 0.0
    if float(pad_free) < 0.5 and abs(x - gx) + abs(y - gy) <= 2:
        # near pad but pad occupied
        return 0.0
    if float(congestion) > 0.85:
        return 0.0
    return 1.0 if (c_here + c_goal) * 0.5 >= 0.35 else 0.0


def build_sem_features(
    *,
    artery: np.ndarray,
    connectivity: np.ndarray,
    congestion: np.ndarray,
    pad_free: float,
    path_clear: float,
    pos: Tuple[int, int],
    goal: Tuple[int, int],
) -> np.ndarray:
    """Fixed-length semaphore MLP features (N_SEM_FEATURES=8)."""
    x, y = int(pos[0]), int(pos[1])
    gx, gy = int(goal[0]), int(goal[1])
    H, W = artery.shape
    a_mean = float(np.mean(artery * (connectivity > 0))) if artery.size else 0.0
    c_mean = float(np.mean(connectivity)) if connectivity.size else 0.0
    cong = float(congestion[y, x]) if 0 <= x < W and 0 <= y < H else 0.0
    c_pos = float(connectivity[y, x]) if 0 <= x < W and 0 <= y < H else 0.0
    c_goal = float(connectivity[gy, gx]) if 0 <= gx < W and 0 <= gy < H else 0.0
    manh = abs(x - gx) + abs(y - gy)
    manh_n = min(1.0, manh / 40.0)
    return np.asarray(
        [
            a_mean,
            c_mean,
            cong,
            float(pad_free),
            float(path_clear),
            c_pos,
            c_goal,
            manh_n,
        ],
        dtype=np.float32,
    )


def teacher_park_artery_from_channels(
    channels: np.ndarray,
    *,
    goals: Optional[Sequence[Tuple[int, int]]] = None,
    sem_pos: Optional[Tuple[int, int]] = None,
    sem_goal: Optional[Tuple[int, int]] = None,
    pad_free: float = 1.0,
    path_clear: float = 1.0,
) -> dict:
    """v5 teacher labels from park_artery channels (C=12)."""
    walk = channels[1]
    artery = teacher_artery_field(walk)
    active = channels[3] if channels.shape[0] > 3 else None
    unload_b = channels[8] if channels.shape[0] > 8 else None
    parking = teacher_parking_field(
        walk, artery, active_path=active, unload_blocked=unload_b
    )
    goal_list: list[Tuple[int, int]] = list(goals or ())
    if not goal_list and channels.shape[0] > 5:
        # peaks of failed-unload attractor
        att = channels[5]
        if float(att.max()) > 0:
            ys, xs = np.where(att >= float(att.max()) * 0.9)
            goal_list = [(int(x), int(y)) for y, x in zip(ys, xs)]
    connectivity = teacher_connectivity_field(
        walk, artery, goal_list, active_path=active, parking=parking
    )
    cong = channels[9] if channels.shape[0] > 9 else np.zeros_like(walk)
    if sem_pos is None:
        # default: any failed AGV cell
        if channels.shape[0] > 4 and float(channels[4].sum()) > 0:
            ys, xs = np.where(channels[4] > 0.5)
            sem_pos = (int(xs[0]), int(ys[0]))
        else:
            sem_pos = (PLAY_LO + 1, PLAY_LO + 1)
    if sem_goal is None:
        sem_goal = goal_list[0] if goal_list else sem_pos
    sem = teacher_semaphore_label(
        connectivity=connectivity,
        pos=sem_pos,
        goal=sem_goal,
        pad_free=pad_free,
        path_clear=path_clear,
        congestion=float(cong[sem_pos[1], sem_pos[0]])
        if 0 <= sem_pos[0] < cong.shape[1] and 0 <= sem_pos[1] < cong.shape[0]
        else 0.0,
    )
    sem_feat = build_sem_features(
        artery=artery,
        connectivity=connectivity,
        congestion=cong,
        pad_free=pad_free,
        path_clear=path_clear,
        pos=sem_pos,
        goal=sem_goal,
    )
    return {
        "parking": parking,
        "artery": artery,
        "connectivity": connectivity,
        "semaphore": float(sem),
        "sem_feat": sem_feat,
    }
