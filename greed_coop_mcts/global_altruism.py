"""Progress-Gated Focal Altruism (PGFA).

Why naive altruism deadlocks
---------------------------
Unconditional corridor-clear / gap-yield serve *all* busy agents at once,
with no check that fleet distance-to-goal improved. Agents thrash, mutually
yield, or pin on corridor junctions (SH03/SH11/SH17).

PGFA rules
----------
1. One focal **beneficiary** (usually a jammed high-pri carrier).
2. Altruism only opens *that* agent's BFS prefix.
3. Progress gate: if beneficiary distance does not drop, escalate / switch.
4. Successful off-path park freezes (no per-tick re-target).
5. Role asymmetry: idles sacrifice freely; carriers only yield to the focal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from greed_coop_mcts.meeting_rh import (
    BUSY_PATH_PREFIX,
    MeetingStats,
    collect_busy_path,
    idle_clear_goal,
    legalize_desired,
)
from greed_coop_mcts.pgcp import ProgressState, cell_degree, outward_step
from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, Pose, in_arena
from greed_coop_mcts.rule_system import RealtimeRuleSystem

Cell = Tuple[int, int]

PGFA_STALL = 25
PGFA_STREAK = 12
PGFA_ENDGAME_REM = 5
PGFA_SWITCH_NO_PROG = 50
PGFA_SWITCH_NEAR = 120  # sticky longer when beneficiary is already near pad
PGFA_NEAR_GOAL = 2
PGFA_PAD_EVICT = 3  # idle on BFS prefix within this Manh of unload → hard evict
PGFA_PARK_TTL = 24
PGFA_PATH_PREFIX = 24


@dataclass
class AltruismState:
    beneficiary: Optional[str] = None
    beneficiary_goal: Optional[Cell] = None
    last_dist: float = 1e9
    no_progress: int = 0
    park_freeze: Dict[str, int] = field(default_factory=dict)
    n_focal_applies: int = 0
    n_park_freezes: int = 0
    n_switches: int = 0


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors4(cell: Cell, blocked: Set[Cell]) -> List[Cell]:
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


def should_pgfa(
    *,
    stall: int,
    remaining: int,
    carrying: Set[str],
    blocked_streak: Dict[str, int],
    n_switches: int = 0,
) -> bool:
    del n_switches  # reserved; sticky switch already handled in select_beneficiary
    if int(remaining) <= PGFA_ENDGAME_REM:
        return True
    if int(stall) >= PGFA_STALL:
        return True
    return any(int(blocked_streak.get(n, 0)) >= PGFA_STREAK for n in carrying)


def select_beneficiary(
    *,
    state: AltruismState,
    carrying: Set[str],
    assigned: Set[str],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked_streak: Dict[str, int],
    poses: Dict[str, Pose],
) -> Optional[str]:
    """Sticky focal agent; prefer closest-to-goal; longer sticky when near pad."""
    pool = [n for n in carrying if n in poses]
    if not pool:
        pool = [n for n in assigned if n in poses]
    if not pool:
        state.beneficiary = None
        state.beneficiary_goal = None
        return None

    def _near(a: str) -> int:
        g = goals.get(a)
        if g is None:
            return 10_000
        return _manh(poses[a][:2], g)

    # Closest-first, then priority / streak.
    ranked = sorted(
        pool,
        key=lambda a: (
            _near(a),
            -float(priority.get(a, 0.0)),
            -int(blocked_streak.get(a, 0)),
            a,
        ),
    )
    cur = state.beneficiary
    cur_goal = goals.get(cur) if cur is not None else None
    switch_th = PGFA_SWITCH_NO_PROG
    if cur in pool and _near(cur) <= PGFA_NEAR_GOAL:
        switch_th = PGFA_SWITCH_NEAR
    need_switch = (
        cur is None
        or cur not in pool
        or cur_goal != state.beneficiary_goal
        or int(state.no_progress) >= switch_th
    )
    if need_switch:
        pick = ranked[0]
        if (
            cur in pool
            and int(state.no_progress) >= switch_th
            and len(ranked) > 1
        ):
            # Prefer another near-goal carrier if previous lead stalled at pad.
            near_others = [a for a in ranked if a != cur and _near(a) <= PGFA_NEAR_GOAL]
            pick = near_others[0] if near_others else next(
                (a for a in ranked if a != cur), ranked[0]
            )
        if pick != cur:
            state.n_switches += 1
        state.beneficiary = pick
        state.beneficiary_goal = goals.get(pick)
        state.last_dist = 1e9
        state.no_progress = 0
    return state.beneficiary


def _beneficiary_dist(
    *,
    name: str,
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    rule_sys: Optional[RealtimeRuleSystem],
) -> float:
    cur = poses[name][:2]
    g = goals.get(name, cur)
    if rule_sys is not None:
        path = rule_sys.bfs_path(cur, g, blocked)
        if path:
            return float(len(path))
    return float(_manh(cur, g))


def tick_park_freeze(state: AltruismState) -> None:
    for k in list(state.park_freeze.keys()):
        state.park_freeze[k] = int(state.park_freeze[k]) - 1
        if state.park_freeze[k] <= 0:
            state.park_freeze.pop(k, None)


def apply_focal_altruism(
    *,
    state: AltruismState,
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
    blocked_streak: Dict[str, int],
    rule_sys: Optional[RealtimeRuleSystem],
    stall: int,
    remaining: int,
    stats: Optional[MeetingStats] = None,
    progress: Optional[ProgressState] = None,
) -> Dict[str, Cell]:
    """Single altruism channel: serve one beneficiary's corridor."""
    stats = stats or MeetingStats()
    progress = progress or ProgressState()
    tick_park_freeze(state)
    if not should_pgfa(
        stall=stall,
        remaining=remaining,
        carrying=carrying,
        blocked_streak=blocked_streak,
        n_switches=int(state.n_switches),
    ):
        return desired

    # Pad-egress idles outrank carriers: clear their exit before pad ingress.
    if progress.egress_agents:
        eg_pool = [e for e in progress.egress_agents if e in poses and e not in force_stay]
        if eg_pool:
            bene = max(
                eg_pool,
                key=lambda a: (
                    float(priority.get(a, 0.0)),
                    int(blocked_streak.get(a, 0)),
                    a,
                ),
            )
            state.beneficiary = bene
            state.beneficiary_goal = goals.get(bene)
        else:
            bene = None
    # Progress lock: keep contested-destination holder as focal when set.
    elif (
        progress.lock_holder is not None
        and progress.lock_ttl > 0
        and progress.lock_holder in carrying
        and progress.lock_holder in poses
    ):
        bene = progress.lock_holder
        state.beneficiary = bene
        state.beneficiary_goal = goals.get(bene)
    else:
        bene = select_beneficiary(
            state=state,
            carrying=carrying,
            assigned=assigned,
            goals=goals,
            priority=priority,
            blocked_streak=blocked_streak,
            poses=poses,
        )
    if bene is None:
        return desired

    state.n_focal_applies += 1
    dist = _beneficiary_dist(
        name=bene,
        poses=poses,
        goals=goals,
        blocked=blocked,
        rule_sys=rule_sys,
    )
    if dist < state.last_dist - 0.5:
        state.no_progress = 0
        state.last_dist = dist
    else:
        state.no_progress += 1
        state.last_dist = min(state.last_dist, dist)

    # Focal corridor only (not union of all busy — that starves SH02/SH03).
    focal_path = collect_busy_path(
        names=names,
        poses=poses,
        goals=goals,
        blocked=blocked,
        busy={bene},
        rule_sys=rule_sys,
        prefix=PGFA_PATH_PREFIX,
    )
    if not focal_path:
        return desired

    out = dict(desired)
    positions = {n: poses[n][:2] for n in names}
    pri = dict(priority)
    pri[bene] = float(pri.get(bene, 0.0)) + 1e7
    bg = goals.get(bene, positions[bene])
    keep_clear: Set[Cell] = set(focal_path)
    keep_clear.add(bg)

    for n in list(state.park_freeze.keys()):
        if n != bene and positions.get(n) in keep_clear:
            state.park_freeze.pop(n, None)

    reserved: Set[Cell] = {positions[bene]}
    for n in names:
        if n in force_stay:
            out[n] = positions[n]
            reserved.add(positions[n])
            continue
        if n in state.park_freeze and n != bene:
            cur = positions[n]
            if cur not in keep_clear:
                out[n] = cur
                reserved.add(cur)

    def _occ(cand: Cell, self_n: str) -> bool:
        return any(
            m != self_n
            and positions[m] == cand
            and out.get(m, positions[m]) == cand
            for m in names
        )

    def _pick_evac(
        cur: Cell, self_n: str, *, prefer_pad: bool, outward: bool = False
    ) -> Cell:
        # Outward push for non-carrier blockers on narrow cells / explicit flag.
        # Do NOT push other carriers here — Contested-Destination Gate handles
        # them (carrier↔carrier outward caused SH17 livelock).
        if outward and self_n not in carrying:
            step = outward_step(
                cur=cur,
                pad=bg,
                blocked=blocked,
                reserved=reserved,
                forbid=keep_clear,
                occupied_fn=lambda c, sn=self_n: _occ(c, sn),
            )
            if step != cur:
                progress.n_outward += 1
                stats.n_outward += 1
            return step
        lateral = [
            c
            for c in _neighbors4(cur, blocked)
            if c not in reserved and c not in keep_clear and not _occ(c, self_n)
        ]
        # Hard preference: if pad neighbor free, take it when boxed on approach.
        if prefer_pad and bg in _neighbors4(cur, blocked):
            if bg not in reserved and not _occ(bg, self_n) and bg != positions[bene]:
                return bg
        best = None
        best_key = None
        for cand in _neighbors4(cur, blocked):
            if cand in reserved or cand == positions[bene] or _occ(cand, self_n):
                continue
            if lateral:
                key = (
                    0 if cand not in keep_clear else 1,
                    -_manh(cand, positions[bene]),
                    _manh(cand, home.get(self_n, cur)),
                    cand,
                )
            else:
                toward = _manh(cand, bg) < _manh(cur, bg)
                key = (
                    0 if cand != positions[bene] else 2,
                    0 if (cand == bg or toward) else 1,
                    0 if cand not in keep_clear else 1,
                    -_manh(cand, positions[bene]),
                    cand,
                )
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        return best if best is not None else cur

    # Idle first; no same-drop serialization (ENABLE_SAME_DROP_SERIAL stays OFF).
    order = sorted(
        names,
        key=lambda a: (
            0 if a == bene else 1,
            0
            if a not in carrying and a not in assigned
            else 1
            if a in assigned
            else 2,
            float(pri.get(a, 0.0)),
            a,
        ),
    )
    for n in order:
        if n == bene or n in force_stay:
            continue
        cur = positions[n]
        if n in state.park_freeze and cur not in keep_clear:
            continue
        want = out.get(n, cur)
        pad_block = n not in carrying and cur in keep_clear and (
            _manh(cur, bg) <= PGFA_PAD_EVICT
        )
        on_keep = cur in keep_clear or want in keep_clear
        must = pad_block or (
            on_keep
            and (
                n not in carrying
                or cur in focal_path
                or _manh(cur, positions[bene]) <= 2
            )
        )
        if not must:
            if want in keep_clear and n not in carrying:
                out[n] = _pick_evac(cur, n, prefer_pad=False)
                if out[n] in keep_clear and cur not in keep_clear:
                    out[n] = cur
                reserved.add(out[n])
            else:
                reserved.add(want)
            continue

        # Non-holder carriers on focal path: step toward their stage wait
        # (already rewritten off-pad) or lateral clear — never toward pad.
        if n in carrying and n != bene and cur in keep_clear:
            nxt = _pick_evac(cur, n, prefer_pad=False, outward=False)
            # Prefer increasing Manh to pad among legal laterals.
            best = nxt
            best_d = _manh(nxt, bg)
            for cand in _neighbors4(cur, blocked):
                if cand in reserved or cand == positions[bene] or _occ(cand, n):
                    continue
                if cand in keep_clear and _manh(cand, bg) <= _manh(cur, bg):
                    continue
                d = _manh(cand, bg)
                if d > best_d:
                    best_d = d
                    best = cand
            out[n] = best
            reserved.add(best)
            if best != cur:
                stats.n_corridor_clears += 1
            continue
        nxt = _pick_evac(
            cur, n, prefer_pad=pad_block and n not in carrying, outward=False
        )
        out[n] = nxt
        reserved.add(nxt)
        if nxt != cur and nxt not in keep_clear:
            stats.n_corridor_clears += 1
            if n not in carrying and n not in assigned:
                state.park_freeze[n] = PGFA_PARK_TTL
                state.n_park_freezes += 1
        elif nxt != cur and pad_block:
            stats.n_corridor_clears += 1

    # Mild whack-a-mole guard: idles must not retarget beneficiary pad cell.
    for n in names:
        if n == bene or n in force_stay or n in carrying:
            continue
        if out.get(n) == bg:
            alt = _pick_evac(positions[n], n, prefer_pad=False)
            out[n] = alt if alt != bg else positions[n]

    bcur = positions[bene]
    if rule_sys is not None:
        path = rule_sys.bfs_path(bcur, bg, blocked)
        bnxt = path[0] if path else bcur
    else:
        bnxt = bcur
        best_d = _manh(bcur, bg)
        for cand in _neighbors4(bcur, blocked):
            if cand in reserved and cand != bcur:
                continue
            d = _manh(cand, bg)
            if d < best_d:
                best_d = d
                bnxt = cand
    # Push whoever occupies bene's next cell (idle: outward; carrier: away).
    depth = 0
    while depth < 3:
        holder = next(
            (
                m
                for m in names
                if m != bene and out.get(m, positions[m]) == bnxt
            ),
            None,
        )
        if holder is None or bnxt == bcur:
            break
        hcur = positions[holder]
        if holder not in carrying:
            alt = _pick_evac(
                hcur,
                holder,
                prefer_pad=False,
                outward=cell_degree(hcur, blocked) <= 2,
            )
        else:
            alt = _pick_evac(hcur, holder, prefer_pad=False, outward=False)
            best = alt
            best_d = _manh(alt, bg)
            for cand in _neighbors4(hcur, blocked):
                if cand in reserved or cand == bcur or _occ(cand, holder):
                    continue
                d = _manh(cand, bg)
                if d > best_d:
                    best_d = d
                    best = cand
            alt = best
        if alt != bnxt and alt != hcur:
            out[holder] = alt
            reserved.add(alt)
            depth += 1
            if out.get(holder) != bnxt:
                break
        else:
            bnxt = bcur
            break
    out[bene] = bnxt

    return legalize_desired(
        names=names,
        poses=poses,
        desired=out,
        priority=pri,
        force_stay=force_stay,
        blocked=blocked,
        goals=goals,
        carrying=carrying,
    )


def focal_gap_ok(
    *,
    mover: str,
    blocker: str,
    beneficiary: Optional[str],
    carrying: Set[str],
    priority: Dict[str, float],
) -> bool:
    """Asymmetric yield gate for local_gap_yield."""
    if beneficiary is None:
        return True
    # Always allow yielding for the focal mover (incl. idle on pad approach).
    if mover == beneficiary:
        return True
    # Idle blockers always side-step for any higher-pri mover under PGFA.
    if blocker not in carrying:
        return True
    # Carriers do not yield to non-focal carriers (kills mutual livelock).
    if blocker in carrying and mover in carrying and mover != beneficiary:
        return False
    # Large priority gap still OK (legacy).
    if float(priority.get(mover, 0.0)) - float(priority.get(blocker, 0.0)) >= 1e5:
        return True
    return False
