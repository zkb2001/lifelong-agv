"""4-channel FOV observations for the Transformer policy."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from greed_coop_mcts.obs import (
    ACTIONS,
    OBS_RADIUS,
    OBS_SIZE,
    Action,
    Cell,
    action_mask,
    apply_action,
    bfs_dist,
    legal_actions,
)
from greed_coop_mcts.transformer_net import XFMR_CHANNELS

N_CHANNELS_XFMR = XFMR_CHANNELS


def build_observation_xfmr(
    ego: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Iterable[Cell],
    *,
    radius: int = OBS_RADIUS,
) -> np.ndarray:
    """(C,H,W): agents, obstacles, cost-to-go, goal-in-patch."""
    obs_set: Set[Cell] = {tuple(c) for c in obstacles}  # type: ignore[misc]
    side = 2 * radius + 1
    agents = np.zeros((side, side), dtype=np.float32)
    walls = np.zeros((side, side), dtype=np.float32)
    cost = np.zeros((side, side), dtype=np.float32)
    gchan = np.zeros((side, side), dtype=np.float32)

    for ox, oy in others:
        i = ox - ego[0] + radius
        j = oy - ego[1] + radius
        if 0 <= i < side and 0 <= j < side:
            agents[i, j] = 1.0

    dist: Dict[Cell, int] = {}
    if goal is not None:
        dist = bfs_dist(goal, ego, obs_set, limit=900)
        gi = goal[0] - ego[0] + radius
        gj = goal[1] - ego[1] + radius
        if 0 <= gi < side and 0 <= gj < side:
            gchan[gi, gj] = 1.0

    vals: List[float] = []
    raw = np.full((side, side), np.nan, dtype=np.float32)
    for i in range(side):
        for j in range(side):
            cell = (ego[0] + i - radius, ego[1] + j - radius)
            if cell in obs_set:
                walls[i, j] = 1.0
                cost[i, j] = -1.0
                continue
            d = dist.get(cell) if goal is not None else None
            if d is None:
                cost[i, j] = 0.0
            else:
                raw[i, j] = float(d)
                vals.append(float(d))
    if vals:
        dmax = max(max(vals), 1.0)
        for i in range(side):
            for j in range(side):
                if cost[i, j] < 0:
                    continue
                if np.isnan(raw[i, j]):
                    cost[i, j] = 0.0
                else:
                    cost[i, j] = 1.0 - (raw[i, j] / dmax)

    return np.stack([agents, walls, cost, gchan], axis=0).astype(np.float32)


def bfs_expert_action(
    pos: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Set[Cell],
) -> Action:
    """Hard expert: follow BFS cost-to-go; wait if next cell occupied."""
    if goal is None:
        return "Wait"
    dist = bfs_dist(goal, pos, obstacles, limit=900)
    if pos not in dist and goal not in dist:
        return "Wait"
    cur_d = dist.get(pos, 10_000)
    other_set = set(others)
    best_a: Action = "Wait"
    best_score = cur_d + 0.5
    for a in legal_actions(pos, obstacles):
        nxt = apply_action(pos, a, obstacles)
        if a != "Wait" and nxt in other_set:
            continue
        d = dist.get(nxt, 10_000)
        score = float(d) + (0.05 if a == "Wait" else 0.0)
        if score < best_score:
            best_score = score
            best_a = a
    return best_a


def bfs_expert_probs(
    pos: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Set[Cell],
    *,
    temp: float = 0.25,
) -> np.ndarray:
    """Soft distribution peaked on BFS-improving legal moves."""
    mask = action_mask(pos, obstacles)
    scores = np.full(len(ACTIONS), -1e9, dtype=np.float64)
    if goal is None:
        scores[ACTIONS.index("Wait")] = 0.0
    else:
        dist = bfs_dist(goal, pos, obstacles, limit=900)
        cur_d = float(dist.get(pos, 10_000))
        other_set = set(others)
        for i, a in enumerate(ACTIONS):
            if mask[i] < 0.5:
                continue
            nxt = apply_action(pos, a, obstacles)
            if a != "Wait" and nxt in other_set:
                scores[i] = -4.0
                continue
            d = float(dist.get(nxt, 10_000))
            prog = cur_d - d
            scores[i] = prog - (0.15 if a == "Wait" else 0.0)
    xs = np.where(mask > 0.5, scores / max(temp, 1e-6), -1e9)
    xs = xs - np.max(xs)
    ex = np.exp(xs) * mask
    z = ex.sum()
    if z <= 0:
        return (mask / max(mask.sum(), 1.0)).astype(np.float32)
    return (ex / z).astype(np.float32)


def bfs_expert_value(
    pos: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Set[Cell],
) -> float:
    if goal is None:
        return 0.0
    dist = bfs_dist(goal, pos, obstacles, limit=900)
    d = float(dist.get(pos, 40.0))
    base = 1.0 - min(1.0, d / 40.0)
    near = sum(1 for o in others if abs(o[0] - pos[0]) + abs(o[1] - pos[1]) <= 2)
    v = base - 0.12 * float(near)
    return float(max(-1.0, min(1.0, 2.0 * v - 1.0)))
