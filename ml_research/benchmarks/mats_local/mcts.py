"""Per-AGV local neural MCTS with greed–cooperation reward balance.

Inspired by:
  - MATS-LP (Skrynnik et al., AAAI 2024): decentralized neural MCTS; proximal
    agents expand joint actions, distant agents act greedily.
  - MAPF-MCTS reward shaping (Pitanov et al., 2023): individual progress +
    freedom to leave the individual path to avoid collisions.
  - Self-interested MAPF / social welfare (Amir et al.; IJCAI'24 mechanisms):
    trade private progress against shared conflict cost.

Each AGV is an agent. Leaf reward =
  greed_w * personal_progress  -  coop_w * conflict_cost  +  yield_w * yield_bonus
where yield_bonus favors giving way when another agent is closer to its goal
along a contested cell (soft cooperation without a central auctioneer).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml_research.benchmarks.mats_local.network import CostTracerNet
from ml_research.benchmarks.mats_local.obs import (
    ACTIONS,
    Action,
    Cell,
    action_mask,
    apply_action,
    build_observation,
    legal_actions,
    manhattan,
)
from ml_research.benchmarks.mats_local.teacher import teacher_action_probs, teacher_value


@dataclass
class LocalAgent:
    name: str
    pos: Cell
    goal: Optional[Cell] = None


@dataclass
class MCTSConfig:
    num_simulations: int = 64
    horizon: int = 6
    k_proximal: int = 3
    cpuct: float = 2.0
    conflict_pen: float = 3.5
    # Greed–cooperation mix (normalized to sum ≈ 1 in leaf blend).
    greed_w: float = 0.55
    coop_w: float = 0.35
    yield_w: float = 0.10
    seed: int = 0


@dataclass
class _Edge:
    prior: float
    n: int = 0
    w: float = 0.0
    child: Optional["_Node"] = None

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


@dataclass
class _Node:
    positions: Dict[str, Cell]
    depth: int
    agents: Tuple[str, ...]
    edges: Dict[Tuple[Action, ...], _Edge] = field(default_factory=dict)
    expanded: bool = False
    v: float = 0.0
    n: int = 0


def _greed_coop_value(
    positions: Dict[str, Cell],
    goals: Dict[str, Optional[Cell]],
    *,
    greed_w: float,
    coop_w: float,
    yield_w: float,
    conflict_pen: float,
) -> float:
    """Scalar leaf reward: greed toward own goal vs cooperation under conflicts."""
    names = list(positions)
    conf = 0
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            if positions[n1] == positions[n2]:
                conf += 1

    # Personal progress (greed): closer to own goal is better.
    prog = 0.0
    dist: Dict[str, float] = {}
    for n, p in positions.items():
        g = goals.get(n)
        if g is None:
            dist[n] = 0.0
            continue
        d = float(manhattan(p, g))
        dist[n] = d
        prog -= d / 20.0

    # Soft yield bonus: if A is on B's shortest-ish approach and B is closer
    # to its goal than A is to A's goal, reward A for leaving (coop).
    yield_bonus = 0.0
    for i, n1 in enumerate(names):
        g1 = goals.get(n1)
        p1 = positions[n1]
        for n2 in names[i + 1 :]:
            g2 = goals.get(n2)
            p2 = positions[n2]
            if g1 is None or g2 is None:
                continue
            # Contested neighborhood: agents within Manhattan 1.
            if manhattan(p1, p2) > 1:
                continue
            d1, d2 = dist[n1], dist[n2]
            if d2 + 1e-6 < d1 and manhattan(p1, g2) <= manhattan(p2, g2) + 1:
                # n1 is in n2's way and n2 is closer to goal → encourage n1 yield.
                yield_bonus += 0.15
            elif d1 + 1e-6 < d2 and manhattan(p2, g1) <= manhattan(p1, g1) + 1:
                yield_bonus += 0.15

    raw = (
        float(greed_w) * prog
        - float(coop_w) * float(conflict_pen) * float(conf)
        + float(yield_w) * yield_bonus
    )
    return float(np.tanh(raw))


def _state_value(
    positions: Dict[str, Cell],
    goals: Dict[str, Optional[Cell]],
    *,
    conflict_pen: float,
    greed_w: float = 0.55,
    coop_w: float = 0.35,
    yield_w: float = 0.10,
) -> float:
    return _greed_coop_value(
        positions,
        goals,
        greed_w=greed_w,
        coop_w=coop_w,
        yield_w=yield_w,
        conflict_pen=conflict_pen,
    )


def _agent_prior(
    name: str,
    positions: Dict[str, Cell],
    goals: Dict[str, Optional[Cell]],
    obstacles: set,
    net: Optional[CostTracerNet],
    device: str,
) -> np.ndarray:
    pos = positions[name]
    others = [positions[n] for n in positions if n != name]
    mask = action_mask(pos, obstacles)
    if net is None:
        return teacher_action_probs(pos, goals.get(name), others, obstacles)
    obs = build_observation(pos, goals.get(name), others, obstacles)
    p, _ = net.prob_value(obs, mask, device=device)
    p = p * mask
    z = float(p.sum())
    if z <= 1e-8:
        return mask / max(mask.sum(), 1.0)
    return (p / z).astype(np.float32)


def _joint_priors(
    positions: Dict[str, Cell],
    goals: Dict[str, Optional[Cell]],
    agents: Sequence[str],
    obstacles: set,
    net: Optional[CostTracerNet],
    device: str,
) -> Dict[Tuple[Action, ...], float]:
    marg = [
        _agent_prior(n, positions, goals, obstacles, net, device) for n in agents
    ]
    out: Dict[Tuple[Action, ...], float] = {}
    for combo in product(*[list(ACTIONS) for _ in agents]):
        # Skip illegal for any agent
        ok = True
        p = 1.0
        for i, a in enumerate(combo):
            pos = positions[agents[i]]
            if a not in legal_actions(pos, obstacles):
                ok = False
                break
            p *= float(marg[i][ACTIONS.index(a)])
        if not ok or p <= 0:
            continue
        out[combo] = p
    z = sum(out.values()) or 1.0
    return {k: v / z for k, v in out.items()}


def _apply_joint(
    positions: Dict[str, Cell],
    agents: Sequence[str],
    joint: Tuple[Action, ...],
    obstacles: set,
) -> Dict[str, Cell]:
    nxt = dict(positions)
    proposed: Dict[str, Cell] = {}
    for n, a in zip(agents, joint):
        proposed[n] = apply_action(positions[n], a, obstacles)
    claimed: Dict[Cell, str] = {}
    for n in sorted(proposed):
        c = proposed[n]
        if c in claimed:
            nxt[n] = positions[n]
        else:
            claimed[c] = n
            nxt[n] = c
    return nxt


class NeuralLocalMCTS:
    """Egocentric joint PUCT; net (or teacher) supplies priors + leaf values."""

    def __init__(
        self,
        net: Optional[CostTracerNet] = None,
        cfg: Optional[MCTSConfig] = None,
        device: str = "cpu",
    ):
        self.net = net
        self.cfg = cfg or MCTSConfig()
        self.device = device
        self.rng = random.Random(int(self.cfg.seed))

    def plan(
        self,
        agents: Sequence[LocalAgent],
        obstacles: Sequence[Cell],
    ) -> Dict[str, Action]:
        obs_set = {tuple(c) for c in obstacles}  # type: ignore[misc]
        if not agents:
            return {}
        ego = agents[0]
        others = sorted(agents[1:], key=lambda a: manhattan(ego.pos, a.pos))
        proximal = [ego, *others[: max(0, int(self.cfg.k_proximal) - 1)]]
        names = tuple(a.name for a in proximal)
        positions = {a.name: a.pos for a in agents}
        goals = {a.name: a.goal for a in agents}

        root = _Node(positions=dict(positions), depth=0, agents=names)
        self._expand(root, goals, obs_set)

        for _ in range(int(self.cfg.num_simulations)):
            node = root
            path: List[Tuple[_Node, Tuple[Action, ...]]] = []
            while node.expanded and node.edges and node.depth < int(self.cfg.horizon):
                ja = self._select(node)
                path.append((node, ja))
                edge = node.edges[ja]
                if edge.child is None:
                    child_pos = _apply_joint(node.positions, names, ja, obs_set)
                    edge.child = _Node(
                        positions=child_pos,
                        depth=node.depth + 1,
                        agents=names,
                    )
                node = edge.child
                if not node.expanded:
                    break
            leaf_v = self._expand(node, goals, obs_set)
            for parent, ja in reversed(path):
                edge = parent.edges[ja]
                edge.n += 1
                edge.w += leaf_v
                parent.n += 1
                leaf_v *= 0.95

        if not root.edges:
            return {a.name: "Wait" for a in proximal}
        best = max(root.edges.items(), key=lambda kv: kv[1].n)
        return {n: a for n, a in zip(names, best[0])}

    def _leaf_value(
        self,
        positions: Dict[str, Cell],
        goals: Dict[str, Optional[Cell]],
        obstacles: set,
    ) -> float:
        # Average net/teacher values over proximal agents + conflict term
        vs = []
        for n, pos in positions.items():
            others = [positions[m] for m in positions if m != n]
            if self.net is None:
                vs.append(teacher_value(pos, goals.get(n), others, obstacles))
            else:
                obs = build_observation(pos, goals.get(n), others, obstacles)
                mask = action_mask(pos, obstacles)
                _, v = self.net.prob_value(obs, mask, device=self.device)
                vs.append(v)
        # Net/teacher value ≈ individual (greedy) estimate; mix with explicit
        # greed–coop social term so search balances private progress and yielding.
        base = float(np.mean(vs)) if vs else 0.0
        social = _state_value(
            positions,
            goals,
            conflict_pen=self.cfg.conflict_pen,
            greed_w=self.cfg.greed_w,
            coop_w=self.cfg.coop_w,
            yield_w=self.cfg.yield_w,
        )
        return 0.45 * base + 0.55 * social

    def _expand(
        self,
        node: _Node,
        goals: Dict[str, Optional[Cell]],
        obstacles: set,
    ) -> float:
        if node.depth >= int(self.cfg.horizon):
            node.v = self._leaf_value(node.positions, goals, obstacles)
            node.expanded = True
            return node.v
        if node.expanded:
            return node.v
        priors = _joint_priors(
            node.positions,
            goals,
            node.agents,
            obstacles,
            self.net,
            self.device,
        )
        for ja, p in priors.items():
            node.edges[ja] = _Edge(prior=float(p))
        node.v = self._leaf_value(node.positions, goals, obstacles)
        node.expanded = True
        return node.v

    def _select(self, node: _Node) -> Tuple[Action, ...]:
        total = max(1, node.n)
        best_ja = None
        best_score = -1e18
        for ja, edge in node.edges.items():
            u = self.cfg.cpuct * edge.prior * math.sqrt(total) / (1 + edge.n)
            q = edge.q if edge.n > 0 else node.v
            score = q + u
            if score > best_score:
                best_score = score
                best_ja = ja
        assert best_ja is not None
        return best_ja
