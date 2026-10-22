"""Per-AGV local AI: stage-greedy steps under priority stack (RRSC / PIBT-style)."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from greed_coop_mcts.motion import pitch_for_step, turn_cost_ticks
from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, Pose, bfs_field, in_arena
from greed_coop_mcts.stage_protocol import (
    IntentSketch,
    RiskVote,
    StageContract,
    StageRole,
)

if TYPE_CHECKING:
    from greed_coop_mcts.rule_system import RealtimeRuleSystem

Cell = Tuple[int, int]


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


def _side_yield_cell(
    cur: Cell,
    goal: Cell,
    blocked: Set[Cell],
    reserved: Set[Cell],
) -> Optional[Cell]:
    """Pick a side step that does not increase manh much (yield pocket)."""
    best = None
    best_score = 1e18
    base = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
    for nxt in _neighbors(cur, blocked):
        if nxt in reserved:
            continue
        m = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
        # Prefer slight detour / same distance
        score = m + (0.0 if m <= base + 1 else 2.0)
        if score < best_score:
            best_score = score
            best = nxt
    return best


def plan_horizon(
    *,
    agent: str,
    pose: Pose,
    contract: StageContract,
    blocked: Set[Cell],
    reserved_next: Set[Cell],
    bodies: Set[Cell],
    rule_dirs: Optional[np.ndarray],
    horizon: int = 6,
    blocked_streak: int = 0,
    alert_yield: bool = False,
) -> Tuple[IntentSketch, RiskVote]:
    """Plan up to ``horizon`` future cells toward active waypoint.

    Uses soft rule bias + BFS toward contract waypoint. On alert_yield or
    prolonged block, inserts a WAIT / side-step and requests restage.
    """
    cur = (int(pose[0]), int(pose[1]))
    pitch = int(pose[2]) % 360
    contract.advance(cur)
    target = contract.active_waypoint(cur)
    vote = RiskVote(agent=agent)
    cells: List[Cell] = []

    if alert_yield or contract.role == StageRole.HOLD:
        vote.head_on = alert_yield
        # Busy agents: wait one step then continue toward goal (no exile).
        if contract.role in (StageRole.TO_DROP, StageRole.TO_PICK):
            cells = [cur]
            # Fill remaining horizon toward target after a 1-tick yield wait.
            field = bfs_field(target, set(blocked))
            pos = cur
            for _ in range(max(0, horizon - 1)):
                if pos == target or pos not in field:
                    cells.append(pos)
                    continue
                best = pos
                best_d = field.get(pos, 10_000)
                for cand in _neighbors(pos, blocked):
                    d = field.get(cand, 10_000)
                    if d < best_d:
                        best_d = d
                        best = cand
                cells.append(best)
                pos = best
        else:
            cells = [cur] * max(1, horizon)
            if alert_yield and blocked_streak >= 2:
                vote.request_restage = True
        return (
            IntentSketch(
                agent=agent,
                cells=cells,
                priority=float(contract.priority),
                blocked_streak=blocked_streak,
            ),
            vote,
        )

    soft_block = set(blocked)
    # Soft: treat static bodies as costly but try not to path through them early
    field = bfs_field(target, soft_block)
    pos = cur
    heading = pitch
    for step_i in range(max(1, horizon)):
        if pos == target or pos == contract.goal:
            # Hold at goal / pad approach
            cells.append(pos)
            continue
        if pos not in field:
            # unreachable under hard block — try greedy manh + turn
            nxt = pos
            best_m = 1e18
            for cand in _neighbors(pos, blocked):
                if cand in reserved_next and cand != pos:
                    continue
                need = pitch_for_step(pos, cand)
                if need < 0:
                    continue
                m = abs(cand[0] - target[0]) + abs(cand[1] - target[1])
                score = float(m) + float(turn_cost_ticks(heading, need))
                if score < best_m:
                    best_m = score
                    nxt = cand
            cells.append(nxt)
            if nxt != pos:
                need = pitch_for_step(pos, nxt)
                if need >= 0:
                    heading = need
                pos = nxt
            continue

        best = pos
        best_score = 1e18
        rule = -1
        if rule_dirs is not None and 0 <= pos[1] < GRID and 0 <= pos[0] < GRID:
            rule = int(rule_dirs[pos[1], pos[0]])
        for di, (dx, dy) in enumerate(DIR_DELTA):
            cand = (pos[0] + dx, pos[1] + dy)
            if cand in blocked:
                continue
            if not in_arena(cand):
                continue
            # First step: hard-avoid reserved next claims
            if step_i == 0 and cand in reserved_next and cand != pos:
                continue
            d = field.get(cand, 10_000)
            if d >= 10_000 and cand != target:
                continue
            need = pitch_for_step(pos, cand)
            if need < 0:
                continue
            score = float(d) + float(turn_cost_ticks(heading, need))
            if di == rule:
                score -= 0.35
            if cand in bodies and cand != target:
                score += 0.8
            if score < best_score:
                best_score = score
                best = cand
        if best == pos:
            # Stuck this step — wait; maybe side yield
            if blocked_streak >= 3:
                side = _side_yield_cell(
                    pos, target, blocked, reserved_next | (bodies - {pos})
                )
                if side is not None:
                    best = side
                    vote.request_restage = True
                else:
                    vote.stuck = True
                    vote.request_restage = blocked_streak >= 6
            cells.append(best)
            if best != pos:
                need = pitch_for_step(pos, best)
                if need >= 0:
                    heading = need
                pos = best
        else:
            cells.append(best)
            need = pitch_for_step(pos, best)
            if need >= 0:
                heading = need
            pos = best

    if blocked_streak >= 8:
        vote.stuck = True
        vote.request_restage = True
    if contract.role == StageRole.TO_DROP and cur == contract.goal:
        vote.waiting_pad = True

    return (
        IntentSketch(
            agent=agent,
            cells=cells,
            priority=float(contract.priority),
            blocked_streak=blocked_streak,
        ),
        vote,
    )


def plan_all_local(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    contracts: Dict[str, StageContract],
    blocked: Set[Cell],
    force_stay: Set[str],
    rule_dirs: Optional[np.ndarray],
    yielders: Set[str],
    blocked_streak: Dict[str, int],
    horizon: int = 6,
) -> Tuple[Dict[str, IntentSketch], Dict[str, RiskVote], Set[str]]:
    """Priority-ordered local planning; reserved_next filled as we go."""
    sketches: Dict[str, IntentSketch] = {}
    votes: Dict[str, RiskVote] = {}
    restage: Set[str] = set()
    reserved: Set[Cell] = {poses[n][:2] for n in force_stay}
    bodies = {poses[n][:2] for n in names}
    order = sorted(
        names,
        key=lambda n: (
            -float(contracts[n].priority) if n in contracts else 0.0,
            n,
        ),
    )
    for n in order:
        if n in force_stay or n not in contracts:
            sketches[n] = IntentSketch(
                agent=n,
                cells=[poses[n][:2]] * horizon,
                priority=float(contracts[n].priority) if n in contracts else 0.0,
                blocked_streak=int(blocked_streak.get(n, 0)),
            )
            votes[n] = RiskVote(agent=n)
            continue
        sk, vote = plan_horizon(
            agent=n,
            pose=poses[n],
            contract=contracts[n],
            blocked=blocked,
            reserved_next=reserved,
            bodies=bodies - {poses[n][:2]},
            rule_dirs=rule_dirs,
            horizon=horizon,
            blocked_streak=int(blocked_streak.get(n, 0)),
            alert_yield=n in yielders,
        )
        sketches[n] = sk
        votes[n] = vote
        if vote.request_restage:
            restage.add(n)
        if sk.cells:
            reserved.add(sk.cells[0])
    return sketches, votes, restage


def pick_yield_pocket(
    cur: Cell,
    goal: Cell,
    blocked: Set[Cell],
    occupied: Set[Cell],
    *,
    min_dist: int = 2,
) -> Optional[Cell]:
    """Find a nearby free cell for YIELD restage (cheap, no per-cell BFS)."""
    best = None
    best_key = None
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            manh = abs(dx) + abs(dy)
            if manh < min_dist:
                continue
            cand = (cur[0] + dx, cur[1] + dy)
            if cand in blocked or cand in occupied or cand == goal:
                continue
            if not (1 <= cand[0] < GRID and 1 <= cand[1] < GRID):
                continue
            # Prefer side-steps that stay roughly toward goal corridor.
            gmanh = abs(cand[0] - goal[0]) + abs(cand[1] - goal[1])
            key = (manh, gmanh, cand)
            if best_key is None or key < best_key:
                best_key = key
                best = cand
    return best


def pibt_plan_desired(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    contracts: Dict[str, StageContract],
    blocked: Set[Cell],
    force_stay: Set[str],
    rule_sys: Optional["RealtimeRuleSystem"],
    blocked_streak: Dict[str, int],
    yielders: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Cell], Dict[str, RiskVote], Set[str]]:
    """Priority-stack greedy one-step (PIBT-style) toward stage waypoints.

    Priority (high→low): legality → do not claim higher-pri next/edge →
    greedily reduce stage distance under guidance cost → final goal via stages.
    """
    yielders = yielders or set()
    desired: Dict[str, Cell] = {}
    votes: Dict[str, RiskVote] = {}
    restage: Set[str] = set()
    reserved_cell: Set[Cell] = {poses[n][:2] for n in force_stay}
    reserved_edge: Set[Tuple[Cell, Cell]] = set()

    order = sorted(
        names,
        key=lambda n: (
            -float(contracts[n].priority) if n in contracts else 0.0,
            n,
        ),
    )

    for n in order:
        cur = poses[n][:2]
        vote = RiskVote(agent=n)
        if n in force_stay or n not in contracts:
            desired[n] = cur
            votes[n] = vote
            reserved_cell.add(cur)
            continue

        contract = contracts[n]
        contract.advance(cur)
        target = contract.active_waypoint(cur)
        streak = int(blocked_streak.get(n, 0))

        if n in yielders or contract.role == StageRole.HOLD:
            vote.head_on = n in yielders
            desired[n] = cur
            votes[n] = vote
            reserved_cell.add(cur)
            if streak >= 4:
                vote.request_restage = True
                restage.add(n)
            continue

        field = bfs_field(target, blocked)
        d0 = field.get(cur, 10_000)
        candidates: List[Tuple[float, Cell]] = []
        # WAIT is always legal.
        wait_score = float(d0) + 0.05
        if rule_sys is not None and rule_sys.stay_penalty is not None:
            wait_score += 0.2 * float(rule_sys.stay_penalty[cur[1], cur[0]])
        candidates.append((wait_score + 0.4, cur))  # slight bias to move if equal

        for cand in _neighbors(cur, blocked):
            # Do not enter a cell reserved by higher priority.
            if cand in reserved_cell and cand != cur:
                continue
            # Do not take a reserved reverse edge (swap).
            if (cand, cur) in reserved_edge:
                continue
            d = field.get(cand, 10_000)
            if d >= 10_000 and cand != target:
                # Allow greedy manh when field missing.
                d = abs(cand[0] - target[0]) + abs(cand[1] - target[1]) + 50
            score = float(d)
            if rule_sys is not None:
                score += 0.55 * float(rule_sys.edge_step_cost(cur, cand))
            # Prefer progress on stage.
            if d < d0:
                score -= 0.35
            if contract.role == StageRole.YIELD and cand == target:
                score -= 0.5
            candidates.append((score, cand))

        candidates.sort(key=lambda x: (x[0], x[1]))
        chosen = cur
        if candidates:
            chosen = candidates[0][1]

        # If stuck waiting with progress available elsewhere, side-step then restage.
        if chosen == cur and streak >= 3:
            side = _side_yield_cell(
                cur, target, blocked, reserved_cell
            )
            if side is not None and (side, cur) not in reserved_edge:
                chosen = side
                vote.request_restage = True
                restage.add(n)
            elif streak >= 6:
                vote.stuck = True
                vote.request_restage = True
                restage.add(n)

        if streak >= 10:
            vote.stuck = True
            vote.request_restage = True
            restage.add(n)

        if contract.role == StageRole.TO_DROP and cur == contract.goal:
            vote.waiting_pad = True

        desired[n] = chosen
        votes[n] = vote
        reserved_cell.add(chosen)
        if chosen != cur:
            reserved_edge.add((cur, chosen))

    return desired, votes, restage
