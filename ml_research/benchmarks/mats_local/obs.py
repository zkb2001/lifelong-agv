"""Local CostTracer-style observation (MATS-LP inspired).

Two channels on an (2R+1)×(2R+1) patch centered on the ego agent:
  0) agents: 1 if another AGV occupies the cell else 0
  1) cost-to-go: normalized inverted BFS distance to goal; obstacles = -1
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
Action = str
ACTIONS: Tuple[Action, ...] = ("N", "S", "E", "W", "Wait")
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
DIRS: Dict[Action, Cell] = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
    "Wait": (0, 0),
}

OBS_RADIUS = 5  # 11×11 like MATS-LP / PRIMAL2
OBS_SIZE = 2 * OBS_RADIUS + 1
N_CHANNELS = 2
N_ACTIONS = len(ACTIONS)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_dist(
    start: Cell,
    goal: Cell,
    obstacles: Set[Cell],
    *,
    limit: int = 400,
) -> Dict[Cell, int]:
    """BFS distances from start; stops early once goal reached or limit nodes."""
    if start in obstacles:
        return {}
    q: deque[Cell] = deque([start])
    dist: Dict[Cell, int] = {start: 0}
    while q and len(dist) < limit:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (x + dx, y + dy)
            if nxt in obstacles or nxt in dist:
                continue
            dist[nxt] = dist[(x, y)] + 1
            q.append(nxt)
    return dist


def cost_to_go_field(
    goal: Optional[Cell],
    obstacles: Set[Cell],
    center: Cell,
    *,
    radius: int = OBS_RADIUS,
) -> np.ndarray:
    """Normalized inverted cost-to-go on local patch; obstacles=-1, empty=0..1."""
    side = 2 * radius + 1
    out = np.zeros((side, side), dtype=np.float32)
    if goal is None:
        for i in range(side):
            for j in range(side):
                cell = (center[0] + i - radius, center[1] + j - radius)
                if cell in obstacles:
                    out[i, j] = -1.0
        return out

    # BFS from goal so field is true cost-to-go
    dist = bfs_dist(goal, center, obstacles, limit=800)
    vals: List[float] = []
    raw = np.full((side, side), np.nan, dtype=np.float32)
    for i in range(side):
        for j in range(side):
            cell = (center[0] + i - radius, center[1] + j - radius)
            if cell in obstacles:
                out[i, j] = -1.0
                continue
            d = dist.get(cell)
            if d is None:
                out[i, j] = 0.0
            else:
                raw[i, j] = float(d)
                vals.append(float(d))
    if not vals:
        return out
    dmax = max(vals) if vals else 1.0
    dmax = max(dmax, 1.0)
    for i in range(side):
        for j in range(side):
            if out[i, j] < 0:
                continue
            if np.isnan(raw[i, j]):
                out[i, j] = 0.0
            else:
                # 1 = closest to goal in patch
                out[i, j] = 1.0 - (raw[i, j] / dmax)
    return out


def agent_field(
    center: Cell,
    others: Sequence[Cell],
    *,
    radius: int = OBS_RADIUS,
) -> np.ndarray:
    side = 2 * radius + 1
    out = np.zeros((side, side), dtype=np.float32)
    for ox, oy in others:
        i = ox - center[0] + radius
        j = oy - center[1] + radius
        if 0 <= i < side and 0 <= j < side:
            out[i, j] = 1.0
    return out


def build_observation(
    ego: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Iterable[Cell],
    *,
    radius: int = OBS_RADIUS,
) -> np.ndarray:
    """Return (C, H, W) float32 observation."""
    obs_set: Set[Cell] = {tuple(c) for c in obstacles}  # type: ignore[misc]
    a = agent_field(ego, others, radius=radius)
    c = cost_to_go_field(goal, obs_set, ego, radius=radius)
    return np.stack([a, c], axis=0).astype(np.float32)


def legal_actions(pos: Cell, obstacles: Set[Cell]) -> List[Action]:
    out: List[Action] = []
    for a, (dx, dy) in DIRS.items():
        nxt = (pos[0] + dx, pos[1] + dy)
        if a == "Wait" or nxt not in obstacles:
            out.append(a)
    return out or ["Wait"]


def action_mask(pos: Cell, obstacles: Set[Cell]) -> np.ndarray:
    legal = set(legal_actions(pos, obstacles))
    return np.array([1.0 if a in legal else 0.0 for a in ACTIONS], dtype=np.float32)


def apply_action(pos: Cell, action: Action, obstacles: Set[Cell]) -> Cell:
    dx, dy = DIRS.get(action, (0, 0))
    nxt = (pos[0] + dx, pos[1] + dy)
    if action != "Wait" and nxt in obstacles:
        return pos
    return nxt
