"""RRSC-U motion: soft guidance + correct PIBT (no illegal edge-swap).

Root cause of prior freeze: planned A↔B swaps that ``_apply_desired`` rejects,
so agents only spin in place. Classic PIBT (Okumura et al.) forbids the pushed
agent from moving into the pusher's current cell; if no escape, push returns
invalid and the requester tries another candidate / waits.

No map-type branching (no corridor / hutong / deg detectors).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, Pose, bfs_field, in_arena

Cell = Tuple[int, int]

EMA_ALPHA = 0.10
EDGE_W_LO = 1.0
EDGE_W_HI = 3.5
PROGRESS_STALL_ON = 64
HEUR_PENALTY = 1.5


@dataclass
class SoftGuidance:
    """Online occupancy / opposing-flow / goal-contention costs."""

    occ: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID, GRID), dtype=np.float32)
    )
    edge: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID, GRID, 4), dtype=np.float32)
    )
    goal_heat: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID, GRID), dtype=np.float32)
    )
    heur_pen: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID, GRID), dtype=np.float32)
    )
    last_potential: float = 1e18
    no_progress: int = 0

    def decay_heur(self) -> None:
        self.heur_pen *= 0.94

    def observe(
        self,
        *,
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        desired: Optional[Dict[str, Cell]],
        blocked: Set[Cell],
    ) -> None:
        a = float(EMA_ALPHA)
        self.occ *= 1.0 - a
        for p in poses.values():
            x, y = int(p[0]), int(p[1])
            if 0 <= x < GRID and 0 <= y < GRID:
                self.occ[y, x] += a
        self.edge *= 1.0 - a
        if desired:
            for n, nxt in desired.items():
                cur = (int(poses[n][0]), int(poses[n][1]))
                if nxt == cur:
                    continue
                di = _dir_index(cur, nxt)
                if di < 0:
                    continue
                x, y = cur
                if 0 <= x < GRID and 0 <= y < GRID:
                    self.edge[y, x, di] += a
        self.goal_heat *= 1.0 - a
        counts: Dict[Cell, int] = defaultdict(int)
        for g in goals.values():
            counts[g] += 1
        for g, c in counts.items():
            if c >= 2 and g not in blocked:
                gx, gy = g
                if 0 <= gx < GRID and 0 <= gy < GRID:
                    self.goal_heat[gy, gx] += a * float(c)

    def stay_cost(self, cell: Cell) -> float:
        x, y = cell
        if not (0 <= x < GRID and 0 <= y < GRID):
            return 10.0
        return (
            0.25 * float(self.occ[y, x])
            + 0.40 * float(self.goal_heat[y, x])
            + float(self.heur_pen[y, x])
        )

    def edge_cost(self, cur: Cell, nxt: Cell) -> float:
        di = _dir_index(cur, nxt)
        base = 1.0
        if di >= 0:
            x, y = cur
            if 0 <= x < GRID and 0 <= y < GRID:
                base += 1.2 * float(self.edge[y, x, di])
                rev = (di + 2) % 4
                nx, ny = nxt
                if 0 <= nx < GRID and 0 <= ny < GRID:
                    # Opposing flow — soft diversion (works open + narrow).
                    base += 2.0 * float(self.edge[ny, nx, rev])
        base += 0.35 * self.stay_cost(nxt)
        return float(np.clip(base, EDGE_W_LO, EDGE_W_HI))

    def update_progress(
        self,
        *,
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        blocked: Set[Cell],
        busy: Set[str],
    ) -> None:
        self.decay_heur()
        pot = 0.0
        for n in busy:
            if n not in poses or n not in goals:
                continue
            cur = (int(poses[n][0]), int(poses[n][1]))
            field = bfs_field(goals[n], blocked)
            pot += float(field.get(cur, 200))
        if pot >= self.last_potential - 0.25:
            self.no_progress += 1
        else:
            self.no_progress = 0
            self.last_potential = pot
        if pot < self.last_potential:
            self.last_potential = pot
        if self.no_progress >= PROGRESS_STALL_ON:
            for n in busy:
                if n not in poses:
                    continue
                x, y = int(poses[n][0]), int(poses[n][1])
                if 0 <= x < GRID and 0 <= y < GRID:
                    self.heur_pen[y, x] = min(
                        6.0, float(self.heur_pen[y, x]) + HEUR_PENALTY
                    )
            self.last_potential = pot + 1.0
            self.no_progress = 0


def _dir_index(cur: Cell, nxt: Cell) -> int:
    dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
    for i, (ox, oy) in enumerate(DIR_DELTA):
        if (ox, oy) == (dx, dy):
            return i
    return -1


def _neighbors(cell: Cell, blocked: Set[Cell]) -> List[Cell]:
    x, y = cell
    out: List[Cell] = []
    for dx, dy in DIR_DELTA:
        nxt = (x + dx, y + dy)
        if not in_arena(nxt):
            continue
        if nxt in blocked:
            continue
        out.append(nxt)
    return out


def pibt_push_noswap(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    priority: Dict[str, float],
    soft: SoftGuidance,
) -> Dict[str, Cell]:
    """PIBT with priority inheritance + backtracking; **never** plan edge-swaps.

    When ``ai`` pushes ``ak`` occupying candidate ``v``, ``ak`` must not move
    into ``ai``'s current cell (Okumura et al. IJCAI'19). If ``ak`` returns
    invalid, ``ai`` tries the next candidate.
    """
    occupied: Dict[Cell, str] = {
        (int(poses[n][0]), int(poses[n][1])): n for n in names
    }
    desired: Dict[str, Optional[Cell]] = {n: None for n in names}
    # reserved next cells by decided agents
    reserved: Set[Cell] = set()

    dist_cache: Dict[str, Dict[Cell, int]] = {}
    for n in names:
        cur = (int(poses[n][0]), int(poses[n][1]))
        g = goals.get(n, cur)
        dist_cache[n] = bfs_field(g, blocked)

    order = sorted(names, key=lambda a: (-float(priority.get(a, 0.0)), a))
    # Pre-mark force_stay
    for n in force_stay:
        cur = (int(poses[n][0]), int(poses[n][1]))
        desired[n] = cur
        reserved.add(cur)

    visiting: Set[str] = set()

    def _score(n: str, cur: Cell, cand: Cell) -> float:
        field = dist_cache.get(n) or {}
        d = float(field.get(cand, 10_000))
        if cand == cur:
            d += 0.05 + 0.20 * soft.stay_cost(cur)
        else:
            d += 0.45 * soft.edge_cost(cur, cand)
        g = goals.get(n, cur)
        d += 0.01 * (abs(cand[0] - g[0]) + abs(cand[1] - g[1]))
        # Prefer empty cells (tie-break like PIBT paper).
        if cand in occupied and occupied[cand] != n:
            d += 0.02
        return d

    def _pibt(n: str, parent: Optional[str]) -> bool:
        """Return True if n reserved a valid next cell; False = invalid."""
        if desired[n] is not None:
            return True
        if n in visiting:
            return False
        visiting.add(n)

        cur = (int(poses[n][0]), int(poses[n][1]))
        parent_cur: Optional[Cell] = None
        if parent is not None:
            parent_cur = (int(poses[parent][0]), int(poses[parent][1]))

        cands = [cur] + _neighbors(cur, blocked)
        cands = sorted(cands, key=lambda c: (_score(n, cur, c), c))

        for cand in cands:
            # (1) vertex: already reserved by higher / decided
            if cand in reserved and cand != cur:
                continue
            # (2) swap forbid: cannot move into parent's current cell
            if parent_cur is not None and cand == parent_cur:
                continue

            holder = occupied.get(cand)
            if holder is None or holder == n:
                desired[n] = cand
                reserved.add(cand)
                visiting.discard(n)
                return True

            # Holder already decided to leave → ok
            if desired[holder] is not None and desired[holder] != cand:
                desired[n] = cand
                reserved.add(cand)
                visiting.discard(n)
                return True

            # Push undecided holder (priority inheritance)
            if desired[holder] is None:
                # Tentatively reserve so holder cannot pick this for stay-conflict
                # incorrectly — actually PIBT reserves after success.
                ok = _pibt(holder, parent=n)
                if ok and desired[holder] is not None and desired[holder] != cand:
                    desired[n] = cand
                    reserved.add(cand)
                    visiting.discard(n)
                    return True
                # backtrack: holder failed or stayed on cand → try next

        # Stuck → stay, report invalid to parent
        desired[n] = cur
        reserved.add(cur)
        visiting.discard(n)
        return False

    for n in order:
        if desired[n] is None:
            _pibt(n, parent=None)

    out: Dict[str, Cell] = {}
    for n in names:
        cur = (int(poses[n][0]), int(poses[n][1]))
        out[n] = desired[n] if desired[n] is not None else cur

    # Final legalize: drop any residual 2-cycles (should be rare).
    for i, a in enumerate(names):
        pa = (int(poses[a][0]), int(poses[a][1]))
        ca = out[a]
        for b in names[i + 1 :]:
            pb = (int(poses[b][0]), int(poses[b][1]))
            cb = out[b]
            if ca == pb and cb == pa and pa != pb:
                if float(priority.get(a, 0.0)) >= float(priority.get(b, 0.0)):
                    out[b] = pb
                else:
                    out[a] = pa
    return out


def plan_desired_unified(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    priority: Dict[str, float],
    soft: SoftGuidance,
    carrying: Set[str],
    assigned: Set[str],
) -> Dict[str, Cell]:
    busy = set(carrying) | set(assigned)
    soft.update_progress(
        poses=poses, goals=goals, blocked=blocked, busy=busy
    )
    desired = pibt_push_noswap(
        names=names,
        poses=poses,
        goals=goals,
        blocked=blocked,
        force_stay=force_stay,
        priority=priority,
        soft=soft,
    )
    soft.observe(
        poses=poses, goals=goals, desired=desired, blocked=blocked
    )
    return desired
