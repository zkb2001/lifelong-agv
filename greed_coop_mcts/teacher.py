"""Heuristic CostTracer teacher for IL (progress + avoid collisions)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from greed_coop_mcts.obs import (
    ACTIONS,
    Action,
    Cell,
    DIRS,
    action_mask,
    apply_action,
    legal_actions,
    manhattan,
)


def teacher_action_probs(
    pos: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Set[Cell],
    *,
    temp: float = 0.35,
) -> np.ndarray:
    """Softmax over legal moves: prefer progress, penalize stepping onto others."""
    legal = legal_actions(pos, obstacles)
    other_set = set(others)
    scores: Dict[Action, float] = {a: -1e9 for a in ACTIONS}
    cur = float(manhattan(pos, goal)) if goal is not None else 0.0
    for a in legal:
        nxt = apply_action(pos, a, obstacles)
        prog = 0.0
        if goal is not None:
            prog = cur - float(manhattan(nxt, goal))
        coll = -3.0 if nxt in other_set and a != "Wait" else 0.0
        wait_bias = -0.2 if a == "Wait" and prog > 0 else (0.4 if a == "Wait" and prog <= 0 else 0.0)
        # Mild preference to wait when blocked toward goal
        scores[a] = prog + coll + wait_bias
    # Softmax with temperature
    xs = np.array([scores[a] for a in ACTIONS], dtype=np.float64)
    mask = action_mask(pos, obstacles)
    xs = np.where(mask > 0.5, xs / max(1e-6, temp), -1e9)
    xs = xs - np.max(xs)
    ex = np.exp(xs)
    ex = ex * mask
    z = ex.sum()
    if z <= 0:
        p = mask / max(mask.sum(), 1.0)
        return p.astype(np.float32)
    return (ex / z).astype(np.float32)


def teacher_value(
    pos: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Set[Cell],
) -> float:
    """Scalar in [-1,1]: closer to goal and fewer nearby conflicts → higher."""
    if goal is None:
        base = 0.0
    else:
        d = float(manhattan(pos, goal))
        base = 1.0 - min(1.0, d / 20.0)
    # Local crowding penalty
    near = sum(1 for o in others if manhattan(pos, o) <= 2)
    v = base - 0.15 * float(near)
    if pos in set(others):
        v -= 0.5
    return float(max(-1.0, min(1.0, 2.0 * v - 1.0)))


def greedy_action(
    pos: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    obstacles: Set[Cell],
) -> Action:
    p = teacher_action_probs(pos, goal, others, obstacles, temp=0.15)
    return ACTIONS[int(np.argmax(p))]
