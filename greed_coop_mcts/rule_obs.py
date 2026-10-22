"""Global state tensor for the realtime traffic-rule AI.

Each AGV is an agent. The AI sees the whole map plus poses / goals and
emits a single direction per walkable cell.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]  # x, y, pitch

GRID = 21
# Competition / validator arena is [1, 20]. GRID=21 keeps index 20 addressable
# for tensors; row/col 0 are ghost border and must never be walkable at runtime.
LEGAL_MIN = 1
# Direction index: 0E 1N 2W 3S (matches warehouse one-way teachers)
DIR_DELTA = ((1, 0), (0, 1), (-1, 0), (0, -1))
PITCH_TO_DIR = {0: 0, 90: 1, 180: 2, 270: 3}
DIR_TO_PITCH = {0: 0, 1: 90, 2: 180, 3: 270}


def in_arena(cell: Cell, grid: int = GRID) -> bool:
    """True iff cell is inside the competition grid [1, grid-1] (= [1,20])."""
    x, y = int(cell[0]), int(cell[1])
    return LEGAL_MIN <= x < grid and LEGAL_MIN <= y < grid


def seal_ghost_border(blocked: Set[Cell], grid: int = GRID) -> Set[Cell]:
    """Mark x=0 and y=0 as blocked (matches train_rule_system / validator)."""
    out = set(blocked)
    for i in range(grid):
        out.add((i, 0))
        out.add((0, i))
    return out

# Channels (match global_rule_transformer_v1.pt stem: 10ch):
# 0 walkable  1 occupied  2..5 heading E/N/W/S
# 6 deliver_goal (loaded→unload attractor)  7 pickup_goal (empty→door attractor)
# 8 loaded_body  9 conflict_heat
N_RULE_CH = 10


def pitch_to_dir(pitch: int) -> int:
    return PITCH_TO_DIR.get(int(pitch) % 360, 1)


def bfs_field(goal: Cell, blocked: Set[Cell], grid: int = GRID) -> Dict[Cell, int]:
    if goal in blocked:
        return {}
    dist: Dict[Cell, int] = {goal: 0}
    q = deque([goal])
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for dx, dy in DIR_DELTA:
            nxt = (x + dx, y + dy)
            if not in_arena(nxt, grid=grid):
                continue
            if nxt in blocked or nxt in dist:
                continue
            dist[nxt] = d + 1
            q.append(nxt)
    return dist


def attractor_from_goals(
    goals: Sequence[Cell],
    blocked: Set[Cell],
    grid: int = GRID,
) -> np.ndarray:
    """Normalized inverted BFS attractor in [0,1]; obstacles 0."""
    out = np.zeros((grid, grid), dtype=np.float32)
    if not goals:
        return out
    acc = np.zeros((grid, grid), dtype=np.float32)
    for g in goals:
        field = bfs_field(g, blocked, grid=grid)
        if not field:
            continue
        dmax = max(field.values()) or 1
        for (x, y), d in field.items():
            acc[y, x] += 1.0 - (float(d) / float(dmax))
    m = float(acc.max()) if acc.max() > 0 else 1.0
    return (acc / m).astype(np.float32)


def build_rule_state(
    *,
    blocked: Set[Cell],
    poses: Dict[str, Pose],
    deliver_goals: Dict[str, Cell],  # loaded AGVs → unload pad
    pickup_goals: Dict[str, Cell],  # empty assigned → pickup door
    loaded: Set[str],
    conflict_heat: Optional[np.ndarray] = None,
    grid: int = GRID,
) -> np.ndarray:
    """Return (C,H,W) float32 global observation."""
    walk = np.ones((grid, grid), dtype=np.float32)
    for x, y in blocked:
        if 0 <= x < grid and 0 <= y < grid:
            walk[y, x] = 0.0
    # keep border as blocked typically
    walk[0, :] = 0.0
    walk[:, 0] = 0.0

    occ = np.zeros((grid, grid), dtype=np.float32)
    head = np.zeros((4, grid, grid), dtype=np.float32)
    loaded_body = np.zeros((grid, grid), dtype=np.float32)
    for name, (x, y, pitch) in poses.items():
        if not (0 <= x < grid and 0 <= y < grid):
            continue
        occ[y, x] = 1.0
        d = pitch_to_dir(pitch)
        head[d, y, x] = 1.0
        if name in loaded:
            loaded_body[y, x] = 1.0

    deliv = attractor_from_goals(list(deliver_goals.values()), blocked, grid=grid)
    pick = attractor_from_goals(list(pickup_goals.values()), blocked, grid=grid)
    heat = (
        np.asarray(conflict_heat, dtype=np.float32)
        if conflict_heat is not None
        else np.zeros((grid, grid), dtype=np.float32)
    )
    if heat.shape != (grid, grid):
        heat = np.zeros((grid, grid), dtype=np.float32)

    return np.stack(
        [walk, occ, head[0], head[1], head[2], head[3], deliv, pick, loaded_body, heat],
        axis=0,
    ).astype(np.float32)


def teacher_oneway_field(walkable: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Circulation one-way labels: (H,W) int in {0..3}, -1 if not walkable."""
    pref = np.full((grid, grid), -1, dtype=np.int64)
    for y in range(1, grid):
        for x in range(1, grid):
            if walkable[y, x] < 0.5:
                continue
            d = 0 if (y % 2 == 0) else 2
            # junction: column rule
            deg = 0
            for dx, dy in DIR_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid and 0 <= ny < grid and walkable[ny, nx] > 0.5:
                    deg += 1
            if deg >= 3:
                d = 1 if (x % 2 == 0) else 3
            dx, dy = DIR_DELTA[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid and 0 <= ny < grid and walkable[ny, nx] > 0.5):
                for alt in range(4):
                    adx, ady = DIR_DELTA[alt]
                    ax, ay = x + adx, y + ady
                    if 0 <= ax < grid and 0 <= ay < grid and walkable[ay, ax] > 0.5:
                        d = alt
                        break
            pref[y, x] = d
    return pref


def conflict_aware_teacher(
    walkable: np.ndarray,
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    *,
    grid: int = GRID,
) -> np.ndarray:
    """Base circulation, then flip corridors that currently head-on collide."""
    pref = teacher_oneway_field(walkable, grid=grid)
    # Detect pairs moving toward each other on same row/col and flip a strip.
    names = list(poses)
    for i, a in enumerate(names):
        ax, ay, ap = poses[a]
        ga = goals.get(a)
        if ga is None:
            continue
        for b in names[i + 1 :]:
            bx, by, bp = poses[b]
            gb = goals.get(b)
            if gb is None:
                continue
            # same row, approaching
            if ay == by and abs(ax - bx) <= 4:
                da = pitch_to_dir(ap)
                db = pitch_to_dir(bp)
                if {da, db} == {0, 2}:  # E vs W
                    # set row to goal-aligned: prefer east if more goals to the east
                    y = ay
                    prefer_e = sum(1 for g in (ga, gb) if g[0] >= (ax + bx) / 2)
                    d = 0 if prefer_e >= 1 else 2
                    for x in range(1, grid):
                        if walkable[y, x] > 0.5:
                            pref[y, x] = d
            if ax == bx and abs(ay - by) <= 4:
                da = pitch_to_dir(ap)
                db = pitch_to_dir(bp)
                if {da, db} == {1, 3}:
                    x = ax
                    prefer_n = sum(1 for g in (ga, gb) if g[1] >= (ay + by) / 2)
                    d = 1 if prefer_n >= 1 else 3
                    for y in range(1, grid):
                        if walkable[y, x] > 0.5:
                            pref[y, x] = d
    return pref


def goal_flow_teacher(
    walkable: np.ndarray,
    blocked: Set[Cell],
    goals: Dict[str, Cell],
    *,
    grid: int = GRID,
    blend_circulation: float = 0.25,
    goal_weight: float = 1.0,
) -> np.ndarray:
    """Completion-oriented field: each cell points along BFS to nearest goal.

    Votes from all active goals; ties broken by circulation one-way. This is the
    main teacher for high task throughput (not TrafficNet).
    """
    circ = teacher_oneway_field(walkable, grid=grid)
    if not goals:
        return circ
    # votes[y,x,dir]
    votes = np.zeros((grid, grid, 4), dtype=np.float32)
    gw = float(goal_weight)
    for g in goals.values():
        field = bfs_field(g, blocked, grid=grid)
        for (x, y), d0 in field.items():
            if walkable[y, x] < 0.5:
                continue
            # pick neighbor that decreases distance
            best_di = -1
            best_d = d0
            for di, (dx, dy) in enumerate(DIR_DELTA):
                nx, ny = x + dx, y + dy
                nd = field.get((nx, ny))
                if nd is None:
                    continue
                if nd < best_d:
                    best_d = nd
                    best_di = di
            if best_di >= 0:
                # closer cells (near goal) vote stronger
                w = gw / (1.0 + float(d0))
                votes[y, x, best_di] += w
    pref = np.full((grid, grid), -1, dtype=np.int64)
    for y in range(grid):
        for x in range(grid):
            if walkable[y, x] < 0.5:
                continue
            if float(votes[y, x].sum()) <= 1e-8:
                pref[y, x] = int(circ[y, x]) if circ[y, x] >= 0 else 0
                continue
            # blend tiny circulation prior for uniqueness / loops
            scores = votes[y, x].copy()
            if circ[y, x] >= 0 and blend_circulation > 0:
                scores[int(circ[y, x])] += float(blend_circulation) * float(scores.max() + 1e-6)
            pref[y, x] = int(np.argmax(scores))
    return pref


def _phase_proximity(
    poses: Dict[str, Pose],
    loaded: Set[str],
    *,
    grid: int = GRID,
    radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local density of loaded vs empty AGVs (soft discs)."""
    load_prox = np.zeros((grid, grid), dtype=np.float32)
    empty_prox = np.zeros((grid, grid), dtype=np.float32)
    r = max(1, int(radius))
    for name, (x, y, _p) in poses.items():
        if not (0 <= x < grid and 0 <= y < grid):
            continue
        target = load_prox if name in loaded else empty_prox
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid and 0 <= ny < grid):
                    continue
                dist = abs(dx) + abs(dy)
                if dist > r:
                    continue
                target[ny, nx] += 1.0 / (1.0 + float(dist))
    return load_prox, empty_prox


def load_aware_goal_flow_teacher(
    walkable: np.ndarray,
    blocked: Set[Cell],
    deliver_goals: Dict[str, Cell],
    pickup_goals: Dict[str, Cell],
    poses: Dict[str, Pose],
    loaded: Set[str],
    *,
    grid: int = GRID,
    blend_circulation: float = 0.10,
    deliver_priority: float = 1.35,
) -> np.ndarray:
    """Separate loaded→drop and empty→pickup flows; resolve head-on by phase.

    Mixing both goal sets into one vote (plain goal_flow) makes corridor cells
    flip between opposite dirs as AGVs pass — classic SH17 mutual block. Here
    each phase keeps its own field; on disagreement we bias toward the locally
    denser phase, with a mild deliver (loaded) priority because unload pads are
    fewer / more constraining than pickup doors.
    """
    circ = teacher_oneway_field(walkable, grid=grid)
    if not deliver_goals and not pickup_goals:
        return circ
    if not deliver_goals:
        return goal_flow_teacher(
            walkable, blocked, pickup_goals, grid=grid, blend_circulation=blend_circulation
        )
    if not pickup_goals:
        return goal_flow_teacher(
            walkable, blocked, deliver_goals, grid=grid, blend_circulation=blend_circulation
        )

    flow_d = goal_flow_teacher(
        walkable, blocked, deliver_goals, grid=grid, blend_circulation=0.0
    )
    flow_p = goal_flow_teacher(
        walkable, blocked, pickup_goals, grid=grid, blend_circulation=0.0
    )
    load_prox, empty_prox = _phase_proximity(poses, loaded, grid=grid, radius=3)

    pref = np.full((grid, grid), -1, dtype=np.int64)
    for y in range(grid):
        for x in range(grid):
            if walkable[y, x] < 0.5:
                continue
            dd = int(flow_d[y, x])
            dp = int(flow_p[y, x])
            if dd < 0 and dp < 0:
                pref[y, x] = int(circ[y, x]) if circ[y, x] >= 0 else 0
                continue
            if dd < 0:
                pref[y, x] = dp
                continue
            if dp < 0:
                pref[y, x] = dd
                continue
            if dd == dp:
                pref[y, x] = dd
                continue
            # Opposite / disagreeing phases: pick by local load vs empty density.
            score_d = float(deliver_priority) * (1.0 + float(load_prox[y, x]))
            score_p = 1.0 + float(empty_prox[y, x])
            # Exact reverse (head-on) → slightly stronger deliver bias.
            if (dd + 2) % 4 == dp:
                score_d *= 1.15
            chosen = dd if score_d >= score_p else dp
            if circ[y, x] >= 0 and blend_circulation > 0 and abs(score_d - score_p) < 0.25:
                # Near-tie: tiny circulation prior for loop-breaking.
                if float(blend_circulation) > 0.5:
                    chosen = int(circ[y, x])
            pref[y, x] = chosen
    return pref


def patch_flow_on_phase_conflict(
    base_flow: np.ndarray,
    walkable: np.ndarray,
    blocked: Set[Cell],
    deliver_goals: Dict[str, Cell],
    pickup_goals: Dict[str, Cell],
    poses: Dict[str, Pose],
    loaded: Set[str],
    *,
    grid: int = GRID,
    radius: int = 2,
    prox_thresh: float = 0.35,
) -> np.ndarray:
    """Overwrite base_flow only on cells where loaded and empty both nearby."""
    if not deliver_goals or not pickup_goals:
        return base_flow
    out = base_flow.copy()
    la = load_aware_goal_flow_teacher(
        walkable,
        blocked,
        deliver_goals,
        pickup_goals,
        poses,
        loaded,
        grid=grid,
        blend_circulation=0.10,
    )
    load_prox, empty_prox = _phase_proximity(poses, loaded, grid=grid, radius=radius)
    for y in range(grid):
        for x in range(grid):
            if walkable[y, x] < 0.5:
                continue
            if float(load_prox[y, x]) < prox_thresh or float(empty_prox[y, x]) < prox_thresh:
                continue
            if int(la[y, x]) >= 0 and int(la[y, x]) != int(out[y, x]):
                out[y, x] = int(la[y, x])
    return out
