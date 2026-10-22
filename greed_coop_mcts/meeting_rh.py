"""Meeting-RH Stage Coop (MRSC): periodic collective meetings + local execute.

Meeting every ``MEET_H`` ticks (or on stall/conflict):
  1) Priority teacher / StageNet priorities
  2) StageNet stage contracts (FINAL / APPROACH / CLEAR)
  3) Soft region admit (carriers protected; approachers may CLEAR)
  4) Windowed prioritized spacetime plan (no edge-swap)
  5) Broadcast contracts + commits; execute first step only

Between meetings each AGV follows its stage contract and may side-step
("钻空子") for higher-priority neighbors — map-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from greed_coop_mcts.glocal_rh import (
    PathCommit,
    RHStats,
    commits_to_desired,
    plan_committed,
    window_repair,
)
from greed_coop_mcts.pgcp import (
    ProgressState,
    apply_pad_egress,
    rewrite_contested_destinations,
)
from greed_coop_mcts.region_admit import apply_region_admit, occupancy_from_goals
from greed_coop_mcts.motion import turn_aware_next
from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, Pose, bfs_field, in_arena
from greed_coop_mcts.stage_net import (
    KIND_CLEAR,
    KIND_FINAL,
    StageChoice,
    StageSelector,
)
from greed_coop_mcts.stage_protocol import StageContract, StageRole
from greed_coop_mcts.rule_system import RealtimeRuleSystem

Cell = Tuple[int, int]

MEET_H = 12
MEET_W = 16
MEET_W_ADAPT = 28
MEET_STALL_FORCE = 24
ADAPT_STALL = 25
ADAPT_HORIZON_TH = 2.0
# Same-drop serialization: blunt heuristic (one FINAL per drop cell).
# Default OFF — prefer global yield / clear / sticky-leader altruism instead.
ENABLE_SAME_DROP_SERIAL = False
SAME_DROP_SERIAL_STALL = 80
SAME_DROP_STREAK = 20
SAME_DROP_REMAINING = 8
# Mid-run clear/yield cool also OFF by default (hurt SH02/SH08); endgame clear stays aggressive.
ENABLE_CLEAR_COOL = False
ENABLE_GAP_YIELD_COOL = False
IDLE_CLEAR_COOL = 8
GAP_YIELD_COOL = 8


@dataclass
class MeetingStats:
    n_meetings: int = 0
    n_window_repairs: int = 0
    n_stage_clear: int = 0
    n_stage_approach: int = 0
    n_stage_final: int = 0
    n_gap_yields: int = 0
    n_corridor_clears: int = 0
    n_same_drop_serial: int = 0
    n_adapt_widen: int = 0
    n_dest_gates: int = 0
    n_outward: int = 0
    n_prog_widen: int = 0
    n_pad_egress: int = 0


IDLE_CLEAR_STALL = 25
IDLE_CLEAR_STREAK = 12
BUSY_PATH_PREFIX = 20


def adaptive_meet_w(
    *,
    stall: int,
    avg_committed: float,
    base: int = MEET_W,
    wide: int = MEET_W_ADAPT,
    stall_th: int = 60,
    horizon_th: float = 1.5,
    no_task_progress: bool = False,
) -> int:
    """Widen under horizon collapse or true task-progress stall (RHCR)."""
    if int(stall) >= int(stall_th) and float(avg_committed) < float(horizon_th):
        return int(wide)
    # SH17-class: commit horizon healthy but done/Manh frozen.
    if no_task_progress and int(stall) >= 25:
        return int(wide)
    return int(base)


def same_drop_demote(
    *,
    carrying: Set[str],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    stall: int,
    blocked_streak: Optional[Dict[str, int]] = None,
    remaining: Optional[int] = None,
    stall_th: int = SAME_DROP_SERIAL_STALL,
    streak_th: int = SAME_DROP_STREAK,
    rem_th: int = SAME_DROP_REMAINING,
) -> Set[str]:
    """Keep one FINAL per identical drop; only endgame or severe stall.

    Disabled unless ``ENABLE_SAME_DROP_SERIAL`` — kept as optional fallback.
    """
    if not ENABLE_SAME_DROP_SERIAL:
        return set()
    endish = remaining is not None and int(remaining) <= int(rem_th)
    severe = int(stall) >= int(stall_th)
    if not (endish or severe) or len(carrying) < 2:
        return set()
    streaks = blocked_streak or {}
    if severe and not endish:
        if not any(int(streaks.get(n, 0)) >= int(streak_th) for n in carrying):
            return set()
    by_goal: Dict[Cell, List[str]] = {}
    for n in carrying:
        g = goals.get(n)
        if g is None:
            continue
        by_goal.setdefault(g, []).append(n)
    demote: Set[str] = set()
    for _g, agents in by_goal.items():
        if len(agents) < 2:
            continue
        ranked = sorted(
            agents, key=lambda a: (-float(priority.get(a, 0.0)), a)
        )
        demote.update(ranked[1:])
    return demote


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


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def should_meet(
    *,
    t: int,
    stall: int,
    last_meet_t: int,
    meet_h: int = MEET_H,
    force_stall: int = MEET_STALL_FORCE,
) -> bool:
    if t == 0 or last_meet_t < 0:
        return True
    if t - last_meet_t >= meet_h:
        return True
    # Avoid thrashing: stall-forced meetings need a longer gap than MEET_H/3.
    if stall >= force_stall and (t - last_meet_t) >= max(8, meet_h // 2):
        return True
    return False


def run_meeting(
    *,
    t: int,
    names: List[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    carrying: Set[str],
    assigned: Set[str],
    blocked_streak: Dict[str, int],
    wait_pri: Dict[str, float],
    rule_sys: RealtimeRuleSystem,
    stage_sel: StageSelector,
    region_cap: Dict,
    existing_commits: Dict[str, PathCommit],
    meet_w: int = MEET_W,
    stall: int = 0,
    remaining: Optional[int] = None,
    stats: Optional[MeetingStats] = None,
    progress: Optional[ProgressState] = None,
    home: Optional[Dict[str, Cell]] = None,
    last_unload_pad: Optional[Dict[str, Cell]] = None,
    near_pad_deg2: int = 0,
) -> Tuple[
    Dict[str, StageContract],
    Dict[str, PathCommit],
    Dict[str, float],
    Dict[str, Cell],
]:
    """Collective meeting → contracts, commits, priorities, desired next cells."""
    stats = stats or MeetingStats()
    stats.n_meetings += 1
    positions = {n: poses[n][:2] for n in names}
    progress = progress or ProgressState()
    home = home or {n: positions[n] for n in names}
    last_unload_pad = last_unload_pad or {}
    # Paths for StageNet candidates (cheap BFS via rule_sys).
    path_cache: Dict[str, List[Cell]] = {}
    for n in names:
        cur = positions[n]
        g = goals.get(n, cur)
        path_cache[n] = rule_sys.bfs_path(cur, g, blocked)

    region_occ = occupancy_from_goals(
        {
            n: goals[n]
            for n in names
            if n in goals and (n in carrying or n in assigned)
        }
    )

    stage_choices = stage_sel.choose_batch(
        names=names,
        poses=poses,
        goals=goals,
        paths=path_cache,
        blocked=blocked,
        loaded=carrying,
        assigned=assigned,
        blocked_streak=blocked_streak,
        stay_penalty=rule_sys.stay_penalty,
        conflict_heat=rule_sys.last_conflict_heat,
        region_occ=region_occ,
        edge_cost_fn=rule_sys.edge_step_cost,
    )
    # Soft admit: only carriers (TO_DROP) are hard-protected; approachers may CLEAR.
    stage_choices = apply_region_admit(
        stage_choices,
        caps=region_cap,
        paths=path_cache,
        protect_roles={StageRole.TO_DROP},
    )

    # If any carrier is stuck, bias idle AGVs toward CLEAR (space for gaps).
    carrier_stuck = any(
        int(blocked_streak.get(n, 0)) >= 12 for n in carrying
    )

    # Provisional priorities for same-drop serialization ranking.
    seed_pri: Dict[str, float] = {
        n: (
            1e6 + float(blocked_streak.get(n, 0)) * 0.1
            if n in carrying
            else 1e5
            if n in assigned
            else float(wait_pri.get(n, 0.0))
        )
        for n in names
    }
    for n, ch in stage_choices.items():
        if n in carrying:
            seed_pri[n] = max(float(ch.priority), 1e6) + float(
                blocked_streak.get(n, 0)
            ) * 0.1
    demoted = same_drop_demote(
        carrying=carrying,
        goals=goals,
        priority=seed_pri,
        stall=int(stall),
        blocked_streak=blocked_streak,
        remaining=remaining,
    )
    if demoted:
        stats.n_same_drop_serial += len(demoted)

    # Stage goals for windowed planning.
    stage_goals: Dict[str, Cell] = {}
    pri: Dict[str, float] = {}
    for n in names:
        ch = stage_choices.get(n)
        if ch is None:
            stage_goals[n] = goals.get(n, positions[n])
            pri[n] = (
                1e6
                if n in carrying
                else 1e5
                if n in assigned
                else float(wait_pri.get(n, 0.0))
            )
            continue
        # Carriers chase FINAL; same-drop extras CLEAR aside until leader unloads.
        if n in carrying:
            g = goals.get(n, positions[n])
            cur = positions[n]
            if n in demoted:
                best = cur
                best_s = _manh(cur, g)
                for cand in _neighbors(cur, blocked):
                    s = _manh(cand, g)
                    if s > best_s:
                        best_s = s
                        best = cand
                stage_goals[n] = best
                pri[n] = max(float(ch.priority), 1e5) * 0.5
                stats.n_stage_clear += 1
            else:
                stage_goals[n] = g
                pri[n] = max(float(ch.priority), 1e6)
                stats.n_stage_final += 1
        else:
            use_clear = ch.kind == KIND_CLEAR or (
                carrier_stuck
                and n not in assigned
                and int(blocked_streak.get(n, 0)) < 18
            )
            if use_clear and ch.kind != KIND_CLEAR:
                cur = positions[n]
                cgoals = [goals[c] for c in carrying if c in goals]
                best = ch.cell
                if cgoals:
                    best_s = min(_manh(cur, g) for g in cgoals)
                    for cand in _neighbors(cur, blocked):
                        s = min(_manh(cand, g) for g in cgoals)
                        if s > best_s:
                            best_s = s
                            best = cand
                stage_goals[n] = best
                stats.n_stage_clear += 1
            else:
                stage_goals[n] = ch.cell
                if ch.kind == KIND_CLEAR:
                    stats.n_stage_clear += 1
                elif ch.kind == KIND_FINAL:
                    stats.n_stage_final += 1
                else:
                    stats.n_stage_approach += 1
            pri[n] = float(ch.priority) + float(wait_pri.get(n, 0.0)) * 0.01
        pri[n] += float(blocked_streak.get(n, 0)) * 0.1

    # Contested-destination gate: non-holders wait at deg≥3 (keep task identity).
    gate0 = int(progress.n_dest_gates)
    stage_goals = rewrite_contested_destinations(
        progress,
        stage_goals=stage_goals,
        carrying=carrying,
        goals=goals,
        positions=positions,
        blocked=blocked,
        rule_sys=rule_sys,
        blocked_streak=blocked_streak,
        home=home,
        near_pad_deg2=near_pad_deg2,
    )
    gated = int(progress.n_dest_gates) - gate0
    stats.n_dest_gates += gated
    if progress.lock_holder and progress.lock_holder in pri:
        pri[progress.lock_holder] = float(pri[progress.lock_holder]) + 3e5

    # Pad-egress: idle on deg≤2 throat → max pri leave; same-pad carriers wait deg≥3.
    eg0 = int(progress.n_pad_egress)
    stage_goals, pri = apply_pad_egress(
        progress,
        stage_goals=stage_goals,
        carrying=carrying,
        assigned=assigned,
        goals=goals,
        positions=positions,
        blocked=blocked,
        force_stay=force_stay,
        rule_sys=rule_sys,
        priority=pri,
        last_unload_pad=last_unload_pad,
        near_pad_deg2=near_pad_deg2,
    )
    stats.n_pad_egress += int(progress.n_pad_egress) - eg0

    # Windowed prioritized plan toward *stage* goals.
    commits = plan_committed(
        names=names,
        positions=positions,
        goals=stage_goals,
        blocked=blocked,
        force_stay=force_stay,
        priority=pri,
        existing={},  # fresh meeting — full replan
        horizon=meet_w,
        replan=set(names),
    )
    commits, n_rep = window_repair(
        commits,
        positions=positions,
        goals=stage_goals,
        blocked=blocked,
        force_stay=force_stay,
        window=meet_w,
    )
    stats.n_window_repairs += int(n_rep)

    # Legalize step-0: no edge-swap, no double-claim.
    desired = commits_to_desired(commits, positions, force_stay)
    desired = legalize_desired(
        names=names,
        poses=poses,
        desired=desired,
        priority=pri,
        force_stay=force_stay,
        blocked=blocked,
        goals=stage_goals,
        carrying=carrying,
    )
    # Sync commit heads with legalized desired.
    for n in names:
        pc = commits.get(n)
        if pc is None:
            continue
        want = desired.get(n, positions[n])
        if not pc.cells:
            pc.cells = [want]
        else:
            pc.cells[0] = want

    # Issue stage contracts (TTL ≈ meeting period).
    contracts: Dict[str, StageContract] = {}
    for n in names:
        ch = stage_choices.get(n)
        g = stage_goals[n]
        role = (
            StageRole.TO_DROP
            if n in carrying
            else StageRole.TO_PICK
            if n in assigned
            else (ch.role if ch is not None else StageRole.IDLE)
        )
        if ch is not None and n not in carrying and ch.kind == KIND_CLEAR:
            role = StageRole.CLEAR
        path = path_cache.get(n) or []
        # Waypoints: commit prefix toward stage goal, else short path sample.
        wps: List[Cell] = []
        pc = commits.get(n)
        if pc and pc.cells:
            wps = list(pc.cells[:6])
        elif path:
            step = max(1, len(path) // 4)
            wps = path[step - 1 :: step][:5]
        if not wps or wps[-1] != g:
            wps = list(wps) + [g]
        contracts[n] = StageContract(
            agent=n,
            role=role,
            waypoints=wps,
            goal=g,
            priority=float(pri.get(n, 0.0)),
            ttl=MEET_H + 4,
            exclusive_pad=goals.get(n) if n in carrying else None,
            issued_t=int(t),
        )

    return contracts, commits, pri, desired


def legalize_desired(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    desired: Dict[str, Cell],
    priority: Dict[str, float],
    force_stay: Set[str],
    blocked: Set[Cell],
    goals: Optional[Dict[str, Cell]] = None,
    carrying: Optional[Set[str]] = None,
) -> Dict[str, Cell]:
    """Drop illegal swaps and resolve vertex double-claims by priority.

    Medium fleets (10–14): Dual Lexicographic Order (DLO) — try name-ASC and
    progress-ASC secondary keys, keep the one with lower carrier ΣManh. Removes
    PYTHONHASHSEED attractor jitter without map-id branches.
    """
    positions = {n: poses[n][:2] for n in names}
    goals = goals or {}
    carrying = carrying or set()

    def _run(order: List[str]) -> Dict[str, Cell]:
        out = dict(desired)
        # Strip 2-cycles (edge-swap).
        for i, a in enumerate(names):
            pa = positions[a]
            ca = out.get(a, pa)
            for b in names[i + 1 :]:
                pb = positions[b]
                cb = out.get(b, pb)
                if ca == pb and cb == pa and pa != pb:
                    if float(priority.get(a, 0.0)) >= float(priority.get(b, 0.0)):
                        out[b] = pb
                    else:
                        out[a] = pa
        claim: Dict[Cell, str] = {}
        for n in order:
            if n in force_stay:
                out[n] = positions[n]
                continue
            cur = positions[n]
            nxt = out.get(n, cur)
            if nxt in blocked and nxt != cur:
                out[n] = cur
                continue
            if nxt == cur:
                continue
            if nxt in claim:
                out[n] = cur
            else:
                claim[nxt] = n
        return out

    def _carrier_sum(out: Dict[str, Cell]) -> float:
        s = 0.0
        for n in carrying:
            if n not in positions:
                continue
            g = goals.get(n, positions[n])
            c = out.get(n, positions[n])
            s += float(abs(c[0] - g[0]) + abs(c[1] - g[1]))
        return s

    order_name = sorted(names, key=lambda a: (-float(priority.get(a, 0.0)), a))
    use_dlo = 10 <= len(names) <= 14 and len(carrying) >= 2 and bool(goals)
    if not use_dlo:
        return _run(order_name)

    def _manh_key(a: str) -> Tuple[float, int, str]:
        cur = positions[a]
        g = goals.get(a, cur)
        return (
            -float(priority.get(a, 0.0)),
            abs(cur[0] - g[0]) + abs(cur[1] - g[1]),
            a,
        )

    order_prog = sorted(names, key=_manh_key)
    out_a = _run(order_name)
    out_b = _run(order_prog)
    if _carrier_sum(out_b) + 1e-9 < _carrier_sum(out_a):
        return out_b
    return out_a


def local_gap_yield(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    desired: Dict[str, Cell],
    priority: Dict[str, float],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    carrying: Set[str],
    stats: Optional[MeetingStats] = None,
    yield_cool: Optional[Dict[str, int]] = None,
    cool_period: int = GAP_YIELD_COOL,
    beneficiary: Optional[str] = None,
    yield_gate=None,
) -> Dict[str, Cell]:
    """Between meetings: lower-pri body on a higher-pri next cell side-steps.

    Optional ``yield_gate(mover, blocker) -> bool`` enforces asymmetric
    altruism (focal beneficiary / no carrier-carrier mutual yield).
    """
    stats = stats or MeetingStats()
    out = dict(desired)
    positions = {n: poses[n][:2] for n in names}
    order = sorted(names, key=lambda a: (-float(priority.get(a, 0.0)), a))
    use_cool = bool(ENABLE_GAP_YIELD_COOL and yield_cool is not None)
    cool = yield_cool if use_cool else {}

    for n in order:
        if n not in carrying and n not in force_stay:
            # Also help assigned high-pri movers.
            pass
        cur = positions[n]
        want = out.get(n, cur)
        if want == cur or n in force_stay:
            continue
        blocker = next(
            (m for m in names if m != n and positions[m] == want), None
        )
        if blocker is None or blocker in force_stay:
            continue
        if float(priority.get(blocker, 0.0)) >= float(priority.get(n, 0.0)):
            continue
        if out.get(blocker, positions[blocker]) != positions[blocker]:
            continue  # already leaving
        if yield_gate is not None and not yield_gate(n, blocker):
            out[n] = cur
            continue
        # Cool only non-carriers (avoid starving delivery maneuvers).
        if use_cool and blocker not in carrying and int(cool.get(blocker, 0)) > 0:
            out[n] = cur
            continue
        # Side-step: not into n's current cell (no swap).
        bcur = positions[blocker]
        reserved = {out.get(m, positions[m]) for m in names if m != blocker}
        reserved.add(cur)
        best = None
        best_key = None
        cg = goals.get(n, cur)
        for cand in _neighbors(bcur, blocked):
            if cand in reserved:
                continue
            occ = next(
                (
                    m
                    for m in names
                    if m != blocker
                    and positions[m] == cand
                    and out.get(m, positions[m]) == cand
                ),
                None,
            )
            if occ is not None:
                continue
            key = (-_manh(cand, cg), _manh(cand, goals.get(blocker, bcur)), cand)
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        if best is not None:
            out[blocker] = best
            stats.n_gap_yields += 1
            if use_cool and blocker not in carrying:
                yield_cool[blocker] = int(cool_period)
        else:
            # Cannot vacate — high-pri waits this tick.
            out[n] = cur

    return legalize_desired(
        names=names,
        poses=poses,
        desired=out,
        priority=priority,
        force_stay=force_stay,
        blocked=blocked,
        goals=goals,
        carrying=carrying,
    )


def should_idle_corridor_clear(
    *,
    stall: int,
    carrying: Set[str],
    blocked_streak: Dict[str, int],
    remaining: Optional[int] = None,
    stall_th: int = IDLE_CLEAR_STALL,
    streak_th: int = IDLE_CLEAR_STREAK,
) -> bool:
    """State trigger: congestion or endgame — never map-typed."""
    if remaining is not None and int(remaining) <= 5:
        return True
    if int(stall) >= int(stall_th):
        return True
    return any(int(blocked_streak.get(n, 0)) >= int(streak_th) for n in carrying)


def collect_busy_path(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    busy: Set[str],
    rule_sys: Optional[RealtimeRuleSystem],
    prefix: int = BUSY_PATH_PREFIX,
) -> Set[Cell]:
    """Union of short BFS prefixes for busy AGVs (carriers + approachers)."""
    path: Set[Cell] = set()
    for n in busy:
        if n not in poses:
            continue
        cur = poses[n][:2]
        g = goals.get(n, cur)
        path.add(cur)
        if rule_sys is not None:
            cells = rule_sys.bfs_path(cur, g, blocked)
        else:
            field = bfs_field(g, blocked)
            cells = []
            x = cur
            for _ in range(int(prefix)):
                best = x
                bd = field.get(x, 10_000)
                for nxt in _neighbors(x, blocked):
                    d = field.get(nxt, 10_000)
                    if d < bd:
                        bd = d
                        best = nxt
                if best == x:
                    break
                cells.append(best)
                x = best
        path.update(cells[: int(prefix)])
    return path


def idle_clear_goal(
    *,
    cur: Cell,
    home: Cell,
    busy_path: Set[Cell],
    blocked: Set[Cell],
    occupied: Set[Cell],
) -> Cell:
    """Park off busy corridors; prefer near home when already clear."""
    adj = any(_manh(cur, p) <= 1 for p in busy_path if p != cur)
    on = cur in busy_path
    if not on and not adj:
        return home if home not in busy_path else cur

    best = cur
    best_key = None
    for cand in _neighbors(cur, blocked):
        if cand in occupied and cand != cur:
            continue
        cand_on = cand in busy_path
        cand_adj = any(_manh(cand, p) <= 1 for p in busy_path if p != cand)
        far = min((_manh(cand, p) for p in busy_path), default=0)
        key = (
            0 if not cand_on else 1,
            0 if not cand_adj else 1,
            -far,
            _manh(cand, home),
            cand,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = cand
    if best != cur:
        return best
    if home not in busy_path and home not in occupied:
        return home
    return cur


def apply_idle_corridor_clear(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    desired: Dict[str, Cell],
    goals: Dict[str, Cell],
    home: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    carrying: Set[str],
    assigned: Set[str],
    priority: Dict[str, float],
    rule_sys: Optional[RealtimeRuleSystem],
    stats: Optional[MeetingStats] = None,
    prefix: int = BUSY_PATH_PREFIX,
    clear_cool: Optional[Dict[str, int]] = None,
    cool_period: int = IDLE_CLEAR_COOL,
    remaining: Optional[int] = None,
) -> Dict[str, Cell]:
    """Force finished/idle AGVs off busy BFS corridors (global yield).

    Mid-run: rate-limited + count only effective off-path steps.
    Endgame (remaining<=5): unrestricted clear (SH02-critical).
    """
    stats = stats or MeetingStats()
    busy = set(carrying) | set(assigned)
    if not busy:
        return desired
    busy_path = collect_busy_path(
        names=names,
        poses=poses,
        goals=goals,
        blocked=blocked,
        busy=busy,
        rule_sys=rule_sys,
        prefix=prefix,
    )
    if not busy_path:
        return desired

    endgame = remaining is not None and int(remaining) <= 5
    use_cool = bool(ENABLE_CLEAR_COOL and clear_cool is not None and not endgame)
    out = dict(desired)
    positions = {n: poses[n][:2] for n in names}
    cool = clear_cool if use_cool else {}
    # Process lower-pri idle first so they claim side cells before higher idle.
    idles = [
        n
        for n in names
        if n not in busy and n not in force_stay
    ]
    idles.sort(key=lambda a: (float(priority.get(a, 0.0)), a))
    reserved = {
        out.get(n, positions[n])
        for n in names
        if n in busy or n in force_stay
    }

    for n in idles:
        if use_cool and int(cool.get(n, 0)) > 0:
            continue
        cur = positions[n]
        on_or_near = cur in busy_path or any(
            _manh(cur, p) <= 1 for p in busy_path if p != cur
        )
        # Also vacate if current desired steps onto busy path.
        want = out.get(n, cur)
        if want in busy_path:
            on_or_near = True
        if not on_or_near and home.get(n, cur) not in busy_path:
            # Keep non-blocking idles still; optional drift toward safe home.
            h = home.get(n, cur)
            if h not in busy_path and h not in reserved and want == cur:
                continue
            if want in busy_path or (
                h in busy_path and want == h
            ):
                on_or_near = True
            else:
                continue
        occupied = set(reserved)
        occupied.update(
            out.get(m, positions[m]) for m in idles if m != n
        )
        target = idle_clear_goal(
            cur=cur,
            home=home.get(n, cur),
            busy_path=busy_path,
            blocked=blocked,
            occupied=occupied - {cur},
        )
        # One step toward target (neighbor pick), never onto busy_path if avoidable.
        nxt = cur
        if target != cur:
            best = None
            best_key = None
            for cand in _neighbors(cur, blocked):
                if cand in reserved:
                    continue
                occ = next(
                    (
                        m
                        for m in names
                        if m != n
                        and positions[m] == cand
                        and out.get(m, positions[m]) == cand
                    ),
                    None,
                )
                if occ is not None:
                    continue
                key = (
                    0 if cand not in busy_path else 1,
                    _manh(cand, target),
                    -_manh(cand, cur),  # prefer actual move
                    cand,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = cand
            if best is not None:
                nxt = best
        # Default (cool OFF): original aggressive clear — local detour, global flow.
        if not use_cool:
            if nxt != want or nxt != cur:
                if nxt != cur or cur in busy_path:
                    stats.n_corridor_clears += 1
            out[n] = nxt
            reserved.add(nxt)
            continue
        # Optional cool path: only count/cool effective off-corridor steps.
        effective = nxt != cur and nxt not in busy_path
        if effective:
            stats.n_corridor_clears += 1
            clear_cool[n] = int(cool_period)
            out[n] = nxt
            reserved.add(nxt)
        elif nxt != cur and cur in busy_path and nxt not in reserved:
            out[n] = nxt
            reserved.add(nxt)
        else:
            out[n] = want if want not in busy_path else cur
            reserved.add(out[n])

    return legalize_desired(
        names=names,
        poses=poses,
        desired=out,
        priority=priority,
        force_stay=force_stay,
        blocked=blocked,
        goals=goals,
        carrying=carrying,
    )


def follow_contract_step(
    *,
    names: List[str],
    poses: Dict[str, Pose],
    contracts: Dict[str, StageContract],
    commits: Dict[str, PathCommit],
    blocked: Set[Cell],
    force_stay: Set[str],
    priority: Dict[str, float],
    rule_sys: Optional[RealtimeRuleSystem],
) -> Dict[str, Cell]:
    """Prefer commit head; else one step toward active stage waypoint."""
    desired: Dict[str, Cell] = {}
    positions = {n: poses[n][:2] for n in names}
    for n in names:
        cur = positions[n]
        if n in force_stay:
            desired[n] = cur
            continue
        pc = commits.get(n)
        if pc and pc.cells:
            # Drop head if we already sit on it.
            if pc.cells[0] == cur:
                pc.cells.pop(0)
            if pc.cells:
                desired[n] = pc.cells[0]
                continue
        ct = contracts.get(n)
        if ct is not None:
            ct.advance(cur)
            target = ct.active_waypoint(cur)
        else:
            target = cur
        pitch = int(poses[n][2]) % 360 if len(poses[n]) > 2 else 0
        if rule_sys is not None and rule_sys.dirs is not None:
            desired[n] = rule_sys.directed_bfs_next(
                cur, target, blocked, start_pitch=pitch
            )
        else:
            desired[n] = turn_aware_next(cur, target, blocked, pitch)
    return legalize_desired(
        names=names,
        poses=poses,
        desired=desired,
        priority=priority,
        force_stay=force_stay,
        blocked=blocked,
    )
