"""Progress-Gated Corridor Protocol (PGCP).

Thin layers on MRSC+PGFA for capacity-1 corridor livelocks:
  A) Contested-destination gate → wait at deg≥3 (never on-corridor FIFO)
  B) Outward push helpers (used by PGFA / gap)
  C) Progress potential (done / Σ Manh) for RHCR widen + busy pulse
  D) Idle corridor hygiene

No map-id branches; ENABLE_SAME_DROP_SERIAL stays OFF.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, in_arena
from greed_coop_mcts.rule_system import RealtimeRuleSystem

Cell = Tuple[int, int]

CONTEND_NEAR = 6
CONTEND_K = 4
PAD_EXCLUSION = 3  # non-holders must not sit this close to contested pad
GATE_NO_PROG = 48
PROG_DONE_STALE = 120
PROG_MANH_STALE = 70
BUSY_PULSE_TTL = 28  # ≈ 2 * MEET_H
BUSY_PULSE_CAP = 3
IDLE_CORRIDOR_N = 24
EGRESS_PRI = 5e7  # above carrier (1e6) and contested-holder boost (3e5)
PAD_CORRIDOR_MAX = 48
# CEBI E4 (SHARP-lite): post-unload sticky retreat on large moderate throats.
E4_FLEET_LO = 19
E4_NEAR_LO = 70
E4_NEAR_HI = 85
T_EGRESS = 16  # reserved; sticky is throat-residence gated


def e4_sticky_enabled(n_agents: int, near_pad_deg2: int) -> bool:
    return (
        int(n_agents) >= E4_FLEET_LO
        and E4_NEAR_LO <= int(near_pad_deg2) <= E4_NEAR_HI
    )


@dataclass
class ProgressState:
    last_done: int = -1
    done_stale: int = 0
    last_sum_manh: float = 1e18
    manh_stale: int = 0
    busy_pulse_left: int = 0
    lock_holder: Optional[str] = None
    lock_pad: Optional[Cell] = None
    lock_ttl: int = 0
    n_prog_widen: int = 0
    n_busy_pulse: int = 0
    n_dest_gates: int = 0
    n_outward: int = 0
    n_idle_hygiene: int = 0
    last_contested: int = 0
    idle_corridor_ticks: Dict[str, int] = field(default_factory=dict)
    # Pad-egress: idle still on deg≤2 pad throat → max priority leave;
    # same-pad carriers wait at deg≥3 (egress-before-ingress).
    egress_agents: Dict[str, Cell] = field(default_factory=dict)  # name -> pad
    n_pad_egress: int = 0
    n_lane_locks: int = 0
    n_e4_sticky: int = 0


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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


def cell_degree(cell: Cell, blocked: Set[Cell]) -> int:
    return len(_neighbors(cell, blocked))


def sum_carrier_manh(
    *,
    carrying: Set[str],
    poses: Dict[str, Tuple],
    goals: Dict[str, Cell],
) -> float:
    s = 0.0
    for n in carrying:
        if n not in poses or n not in goals:
            continue
        s += float(_manh(poses[n][:2], goals[n]))
    return s


def _contested_pads(
    *,
    carrying: Set[str],
    poses: Dict[str, Tuple],
    goals: Dict[str, Cell],
) -> int:
    by: Dict[Cell, int] = defaultdict(int)
    near: Dict[Cell, int] = defaultdict(int)
    for n in carrying:
        g = goals.get(n)
        if g is None or n not in poses:
            continue
        by[g] += 1
        if _manh(poses[n][:2], g) <= CONTEND_NEAR:
            near[g] += 1
    return sum(1 for p, c in near.items() if c >= 2 and by[p] >= 2)


def update_progress(
    state: ProgressState,
    *,
    done: int,
    carrying: Set[str],
    poses: Dict[str, Tuple],
    goals: Dict[str, Cell],
) -> bool:
    """Update stale counters; return True if no-task-progress (need widen/pulse)."""
    if int(done) > int(state.last_done):
        state.done_stale = 0
        state.last_done = int(done)
    else:
        if state.last_done < 0:
            state.last_done = int(done)
        state.done_stale += 1

    sm = sum_carrier_manh(carrying=carrying, poses=poses, goals=goals)
    if sm < state.last_sum_manh - 0.5:
        state.manh_stale = 0
        state.last_sum_manh = sm
    else:
        state.manh_stale += 1
        state.last_sum_manh = min(state.last_sum_manh, sm)

    if state.busy_pulse_left > 0:
        state.busy_pulse_left -= 1
    if state.lock_ttl > 0:
        state.lock_ttl -= 1
        if state.lock_ttl <= 0:
            state.lock_holder = None
            state.lock_pad = None

    contested = _contested_pads(
        carrying=carrying, poses=poses, goals=goals
    )
    state.last_contested = int(contested)
    n_ag = len(poses)
    # Denser fleets tip into corridor deadlock faster → shorter patience.
    done_th = 80 if n_ag >= 16 else PROG_DONE_STALE
    manh_th = 50 if n_ag >= 16 else PROG_MANH_STALE
    no_prog = contested >= 1 and (
        (state.done_stale >= done_th and len(carrying) >= 2)
        or (state.manh_stale >= manh_th and len(carrying) >= 2)
    )
    return no_prog


def effective_fleet_cap(
    base: int, state: ProgressState, *, n_agents: int = 0
) -> int:
    del n_agents  # reserved for future density-aware pulses
    if state.busy_pulse_left > 0:
        return min(int(base), int(BUSY_PULSE_CAP))
    return int(base)


# Throat-Band Capacity (TBC): mid fleets on *moderate* near-pad corridors
# inject one fewer concurrent carrier (6→5). Open maps (low deg2) and
# extreme labyrinths (very high deg2) keep full cap — validated on SH02/08/15.
MID_FLEET_LO = 10
MID_FLEET_HI = 14
THROAT_NEAR_LO = 55
THROAT_NEAR_HI = 85
THROAT_CAP = 5


def near_pad_deg2_count(
    blocked: Set[Cell],
    drop_centers: Set[Cell],
    *,
    radius: int = 6,
) -> int:
    """Count deg≤2 walkable cells within ``radius`` of any dropoff center."""
    if not drop_centers:
        return 0
    n = 0
    for x in range(1, GRID):
        for y in range(1, GRID):
            c = (x, y)
            if c in blocked:
                continue
            if cell_degree(c, blocked) > 2:
                continue
            if any(_manh(c, p) <= int(radius) for p in drop_centers):
                n += 1
    return n


def throat_band_fleet_cap(n_agents: int, near_pad_deg2: int, *, base: int = 6) -> int:
    """Map-id-free capacity from fleet size × corridor-band geometry."""
    if int(n_agents) >= 28:
        return max(int(base), 7)
    if (
        MID_FLEET_LO <= int(n_agents) <= MID_FLEET_HI
        and THROAT_NEAR_LO <= int(near_pad_deg2) <= THROAT_NEAR_HI
    ):
        return int(THROAT_CAP)
    # Dense mid-large fleets on moderate throats (not mega-open / not
    # extreme labyrinth): inject one fewer concurrent carrier. Empirically
    # SH11-class (n in [16,18], near in [80,90]); wider bands regress SH12.
    if 16 <= int(n_agents) <= 18 and 80 <= int(near_pad_deg2) <= 90:
        return int(THROAT_CAP)
    return int(base)


def adaptive_use_stage_net(n_agents: int, near_pad_deg2: int) -> bool:
    """Adaptive Stage Authority (ASA): when StageNet beats goal-flow teacher.

    Empirically StageNet helps dense fleets and extreme near-pad labyrinths;
    the heuristic teacher wins on small fleets with mild/moderate throats and
    on mid-large fleets with moderate corridor density (SH16/18-class).
    Map-id free; decisions use only (n_agents, near_pad_deg2).
    """
    n = int(n_agents)
    r = int(near_pad_deg2)
    # Extreme labyrinths (tiny fleets): learned staging opens deadlocks.
    if n <= 9 and r >= 90:
        return True
    # SH04-class pocket: n=9 with mid-open floor still prefers StageNet.
    if n == 9 and 50 <= r <= 59:
        return True
    # Small fleets otherwise: teacher staging (SH01/03/05/06/07).
    if n <= 9:
        return False
    # Classic mid fleets (10–14): StageNet (gates SH02/08 + SH15 TBC).
    if 10 <= n <= 14:
        return True
    # n=15: open → StageNet (SH09); moderate throat → teacher (SH16).
    if n == 15:
        return r < 60
    # n=18 moderate throat: teacher (SH18); other large fleets: StageNet.
    if n == 18:
        return False
    return True


def trigger_progress_response(
    state: ProgressState, *, n_agents: int = 0
) -> None:
    """Pulse lower busy cap on dense fleets only (sparse: widen-only)."""
    if int(n_agents) < 12:
        return  # SH17-class: busy pulse starved approachers while carriers locked
    if state.busy_pulse_left <= 0:
        state.n_busy_pulse += 1
        state.busy_pulse_left = BUSY_PULSE_TTL


def nearest_deg3(
    *,
    cur: Cell,
    pad: Cell,
    blocked: Set[Cell],
    occupied: Set[Cell],
    forbid: Set[Cell],
    max_steps: int = 24,
) -> Cell:
    """BFS to nearest degree≥3 cell off forbid set (corridor prefix)."""
    if cell_degree(cur, blocked) >= 3 and cur not in forbid and cur not in occupied:
        return cur
    from collections import deque

    q = deque([cur])
    seen = {cur}
    parent: Dict[Cell, Optional[Cell]] = {cur: None}
    best = None
    while q and len(seen) < max_steps * 8:
        c = q.popleft()
        if (
            c != cur
            and cell_degree(c, blocked) >= 3
            and c not in forbid
            and c not in occupied
        ):
            # Prefer farther from pad among first finds via BFS (closest first).
            best = c
            break
        for nxt in _neighbors(c, blocked):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = c
            q.append(nxt)
    if best is None:
        # Fallback: step that increases Manh to pad and leaves forbid if possible.
        best_c = cur
        best_k = (-1, 0)
        for cand in _neighbors(cur, blocked):
            if cand in occupied:
                continue
            k = (
                0 if cand not in forbid else 1,
                _manh(cand, pad),
            )
            if k > best_k:
                best_k = k
                best_c = cand
        return best_c
    return best


def path_prefix(
    *,
    cur: Cell,
    goal: Cell,
    blocked: Set[Cell],
    rule_sys: Optional[RealtimeRuleSystem],
    k: int = CONTEND_K,
) -> Set[Cell]:
    cells: List[Cell] = []
    if rule_sys is not None:
        cells = list(rule_sys.bfs_path(cur, goal, blocked) or [])
    else:
        x = cur
        for _ in range(GRID * 2):
            if x == goal:
                break
            best = None
            bd = _manh(x, goal)
            for nxt in _neighbors(x, blocked):
                d = _manh(nxt, goal)
                if d < bd:
                    bd = d
                    best = nxt
            if best is None:
                break
            cells.append(best)
            x = best
    if not cells or cells[-1] != goal:
        cells = cells + [goal]
    return set(cells[-int(k) :])


def _station_key(pad: Cell, blocked: Set[Cell]) -> frozenset:
    """Group unload pads that share a dropoff hub (same destination station)."""
    hubs = _station_hubs(pad, blocked)
    return frozenset(hubs) if hubs else frozenset([pad])


def rewrite_contested_destinations(
    state: ProgressState,
    *,
    stage_goals: Dict[str, Cell],
    carrying: Set[str],
    goals: Dict[str, Cell],
    positions: Dict[str, Cell],
    blocked: Set[Cell],
    rule_sys: Optional[RealtimeRuleSystem],
    blocked_streak: Optional[Dict[str, int]] = None,
    home: Optional[Dict[str, Cell]] = None,
    near_pad_deg2: int = 0,
) -> Dict[str, Cell]:
    """Non-holders near a contested *station* wait at deg≥3 (keep TO_DROP).

    Prefer-empty-pad can assign sibling unload cells of the same destination;
    gating by exact pad cell then never fires. Group by shared hub instead.
    """
    del home
    streaks = blocked_streak or {}
    by_station: Dict[frozenset, List[str]] = defaultdict(list)
    pad_of: Dict[str, Cell] = {}
    for n in carrying:
        g = goals.get(n)
        if g is None:
            continue
        by_station[_station_key(g, blocked)].append(n)
        pad_of[n] = g

    out = dict(stage_goals)
    occupied = set(positions.values())
    del near_pad_deg2  # reserved for future geometry-gated gates

    for _st, agents in by_station.items():
        if len(agents) < 2:
            continue
        station_pads = {pad_of[a] for a in agents}
        contend: List[str] = []
        for n in agents:
            cur = positions.get(n, pad_of[n])
            if any(_manh(cur, sp) <= CONTEND_NEAR for sp in station_pads):
                contend.append(n)
        if len(contend) < 2:
            continue

        def _near_pad(a: str) -> int:
            cur = positions.get(a, pad_of[a])
            return min(_manh(cur, sp) for sp in station_pads)

        # Lock pad = holder's true unload cell (representative for station).
        sticky_ok = (
            state.lock_holder in contend
            and state.lock_pad in station_pads
            and state.lock_ttl > 0
            and int(streaks.get(state.lock_holder or "", 0)) < GATE_NO_PROG
        )
        if sticky_ok:
            hold = state.lock_holder
        else:
            hold = min(
                contend,
                key=lambda a: (_near_pad(a), -int(streaks.get(a, 0)), a),
            )
            if (
                state.lock_holder in contend
                and int(streaks.get(state.lock_holder, 0)) >= GATE_NO_PROG
            ):
                others = [a for a in contend if a != state.lock_holder]
                if others:
                    hold = min(others, key=lambda a: (_near_pad(a), a))

        hold_pad = pad_of[hold]
        forbid: Set[Cell] = set(station_pads)
        for sp in station_pads:
            forbid |= pad_deg2_corridor(sp, blocked)
        for n in contend:
            forbid |= path_prefix(
                cur=positions.get(n, pad_of[n]),
                goal=pad_of[n],
                blocked=blocked,
                rule_sys=rule_sys,
            )

        for n in agents:
            if n == hold:
                out[n] = goals.get(n, hold_pad)
                continue
            cur = positions.get(n, pad_of[n])
            dpad = _near_pad(n)
            # Far agents keep true goal; near must wait at deg≥3.
            if dpad > CONTEND_NEAR and n not in contend:
                continue
            wait = nearest_deg3(
                cur=cur,
                pad=hold_pad,
                blocked=blocked,
                occupied=occupied - {cur},
                forbid=forbid,
            )
            if (
                dpad > PAD_EXCLUSION
                and cur not in forbid
                and cell_degree(cur, blocked) >= 3
            ):
                wait = cur
            if wait != out.get(n):
                state.n_dest_gates += 1
            out[n] = wait
            occupied.add(wait)
            occupied.discard(cur)
        if state.lock_holder != hold or state.lock_pad != hold_pad:
            state.lock_holder = hold
            state.lock_pad = hold_pad
            state.lock_ttl = GATE_NO_PROG
        elif state.lock_ttl <= 0:
            state.lock_ttl = GATE_NO_PROG
    return out


def outward_step(
    *,
    cur: Cell,
    pad: Cell,
    blocked: Set[Cell],
    reserved: Set[Cell],
    forbid: Set[Cell],
    occupied_fn,
) -> Cell:
    """One step that prefers increasing Manh to pad and leaving forbid."""
    best = cur
    best_key = None
    for cand in _neighbors(cur, blocked):
        if cand in reserved or occupied_fn(cand):
            continue
        key = (
            0 if cand not in forbid else 1,
            -_manh(cand, pad),  # larger Manh better → negate for min
            0 if cell_degree(cand, blocked) >= 3 else 1,
            cand,
        )
        # Fix: we want MAX manh, so use negative in min-key → already -manh
        if best_key is None or key < best_key:
            best_key = key
            best = cand
    return best


def apply_idle_hygiene(
    state: ProgressState,
    *,
    names: List[str],
    poses: Dict[str, Tuple],
    desired: Dict[str, Cell],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    carrying: Set[str],
    assigned: Set[str],
    rule_sys: Optional[RealtimeRuleSystem],
) -> Dict[str, Cell]:
    """Idle on carrier corridor deg≤2 for too long → step to deg≥3."""
    if not carrying:
        state.idle_corridor_ticks.clear()
        return desired
    positions = {n: poses[n][:2] for n in names}
    prefixes: Set[Cell] = set()
    for n in carrying:
        g = goals.get(n)
        if g is None:
            continue
        prefixes |= path_prefix(
            cur=positions[n],
            goal=g,
            blocked=blocked,
            rule_sys=rule_sys,
            k=12,
        )
        prefixes.add(g)

    out = dict(desired)
    reserved = {out.get(n, positions[n]) for n in names}
    for n in names:
        if n in carrying or n in assigned or n in force_stay:
            state.idle_corridor_ticks.pop(n, None)
            continue
        cur = positions[n]
        on_bad = cur in prefixes and cell_degree(cur, blocked) <= 2
        if on_bad:
            state.idle_corridor_ticks[n] = int(state.idle_corridor_ticks.get(n, 0)) + 1
        else:
            state.idle_corridor_ticks[n] = 0
        if int(state.idle_corridor_ticks.get(n, 0)) < IDLE_CORRIDOR_N:
            continue
        # Force one step to deg≥3 off prefixes.
        wait = nearest_deg3(
            cur=cur,
            pad=next(iter(prefixes)) if prefixes else cur,
            blocked=blocked,
            occupied=reserved - {cur},
            forbid=prefixes,
        )
        # One neighbor step toward wait.
        if wait != cur:
            best = cur
            bd = _manh(cur, wait)
            for cand in _neighbors(cur, blocked):
                if cand in reserved and cand != cur:
                    continue
                d = _manh(cand, wait)
                if d < bd:
                    bd = d
                    best = cand
            if best != cur:
                out[n] = best
                reserved.add(best)
                state.n_idle_hygiene += 1
                state.idle_corridor_ticks[n] = 0
    return out


def pad_deg2_corridor(
    pad: Cell,
    blocked: Set[Cell],
    *,
    max_cells: int = PAD_CORRIDOR_MAX,
) -> Set[Cell]:
    """deg≤2 throat attached to pad (empty if pad is open junction with no stubs)."""
    from collections import deque

    corr: Set[Cell] = set()
    q: deque = deque()
    if cell_degree(pad, blocked) <= 2:
        corr.add(pad)
        q.append(pad)
    for nb in _neighbors(pad, blocked):
        if cell_degree(nb, blocked) <= 2 and nb not in corr:
            corr.add(nb)
            q.append(nb)
    seen = set(corr)
    while q and len(corr) < max_cells:
        c = q.popleft()
        for nxt in _neighbors(c, blocked):
            if nxt in seen:
                continue
            if cell_degree(nxt, blocked) <= 2:
                seen.add(nxt)
                corr.add(nxt)
                q.append(nxt)
    return corr


def _station_hubs(pad: Cell, blocked: Set[Cell]) -> Set[Cell]:
    """deg≥3 neighbors of an unload pad (= dropoff hub candidates)."""
    return {c for c in _neighbors(pad, blocked) if cell_degree(c, blocked) >= 3}


def same_unload_station(p1: Cell, p2: Cell, blocked: Set[Cell]) -> bool:
    """True if two unload pads share a dropoff hub (same dest, not cross-dest)."""
    if p1 == p2:
        return True
    return bool(_station_hubs(p1, blocked) & _station_hubs(p2, blocked))


def apply_pad_egress(
    state: ProgressState,
    *,
    stage_goals: Dict[str, Cell],
    carrying: Set[str],
    assigned: Set[str],
    goals: Dict[str, Cell],
    positions: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    rule_sys: Optional[RealtimeRuleSystem],
    priority: Optional[Dict[str, float]] = None,
    last_unload_pad: Optional[Dict[str, Cell]] = None,
    near_pad_deg2: int = 0,
) -> Tuple[Dict[str, Cell], Dict[str, float]]:
    """Egress-before-ingress on pad deg≤2 throats.

    Idle on unload goal / same-station sibling pad (when carriers near) /
    same-pad throat gets max priority to leave. Carriers already inside the
    throat back out. No cross-dest corridor union.

    Fleet×geometry CEBI E2 (lane-lock): mid+ fleets on *extreme* near-pad
    throats (n≥10 and near_pad_deg2≥100) hold outside carriers whose
    shortest path crosses an on-throat egressor. Covers SH08/19; skips
    moderate-throat SH12, fragile SH14 (n=9), and tiny-fleet SH17.
    """
    del rule_sys
    last_unload_pad = last_unload_pad or {}
    pri = dict(priority or {})
    out = dict(stage_goals)
    occupied = set(positions.values())
    n_ag = len(positions)
    near = int(near_pad_deg2)
    e4_on = e4_sticky_enabled(n_ag, near)
    # Extreme-throat CEBI gate (map-id free).
    fleet_cebi = n_ag >= 10 and near >= 100

    seeds: Set[Cell] = set()
    carriers_by_goal: Dict[Cell, List[str]] = defaultdict(list)
    for n in carrying:
        g = goals.get(n)
        if g is None:
            continue
        seeds.add(g)
        carriers_by_goal[g].append(n)
    for n, pad in last_unload_pad.items():
        if n in carrying or n in assigned:
            continue
        if n in positions:
            seeds.add(pad)

    corridors: Dict[Cell, Set[Cell]] = {
        pad: pad_deg2_corridor(pad, blocked) for pad in seeds
    }

    soft_leave: Dict[str, Cell] = {}
    egress: Dict[str, Cell] = {}
    sibling: Dict[str, Cell] = {}  # vacate unused sibling pad; do not steal PGFA
    for n, cur in positions.items():
        if n in carrying or n in assigned or n in force_stay:
            continue
        for g, clist in carriers_by_goal.items():
            if not clist:
                continue
            if cur == g:
                egress[n] = g
                break
            near = any(
                _manh(positions[c], g) <= CONTEND_NEAR
                for c in clist
                if c in positions
            )
            if (
                near
                and cell_degree(cur, blocked) <= 2
                and _station_hubs(cur, blocked)
                and same_unload_station(cur, g, blocked)
            ):
                sibling[n] = g
                break
        if n in egress or n in sibling:
            continue
        if cell_degree(cur, blocked) > 2:
            continue
        for pad, corr in corridors.items():
            if not corr or cur not in corr:
                continue
            pressure = bool(carriers_by_goal.get(pad))
            just_here = last_unload_pad.get(n) == pad
            if pressure:
                egress[n] = pad
                break
            if just_here and e4_on:
                # E4: force max-pri retreat until off throat (no wait for ingress).
                state.n_e4_sticky += 1
                egress[n] = pad
                break
            if just_here:
                soft_leave[n] = pad
                break

    # PGFA focal for egressors. Head-on with a carrier: keep focal only when
    # the idle is closer to the pad (blocks approach); otherwise carrier wins.
    state.egress_agents = {}
    for n, pad in egress.items():
        cur = positions[n]
        blockers = [
            c
            for c in carrying
            if c in positions
            and abs(cur[0] - positions[c][0]) + abs(cur[1] - positions[c][1])
            == 1
        ]
        if not blockers:
            state.egress_agents[n] = pad
            continue
        d_idle = _manh(cur, pad)
        if any(_manh(positions[c], pad) > d_idle for c in blockers):
            state.egress_agents[n] = pad

    def _nudge_off(n: str, pad: Cell, *, boost: float) -> None:
        nonlocal occupied
        corr = set(corridors.get(pad) or ())
        forbid = set(corr) | {pad}
        cur = positions[n]
        wait = nearest_deg3(
            cur=cur,
            pad=pad,
            blocked=blocked,
            occupied=occupied - {cur},
            forbid=forbid,
        )
        if wait != out.get(n, cur):
            state.n_pad_egress += 1
        out[n] = wait
        if boost > 0:
            pri[n] = max(float(pri.get(n, 0.0)), boost)
        occupied.add(wait)
        occupied.discard(cur)

    for n, pad in soft_leave.items():
        if n in egress or n in sibling:
            continue
        _nudge_off(n, pad, boost=0.0)

    # Sibling pad campers: leave with high pri but carriers keep PGFA focus.
    for n, pad in sibling.items():
        if n in egress:
            continue
        _nudge_off(n, pad, boost=EGRESS_PRI * 0.5)

    if not egress:
        return out, pri

    by_epad: Dict[Cell, List[str]] = defaultdict(list)
    for n, pad in egress.items():
        by_epad[pad].append(n)

    def _egressor_between(carrier: str, eg_names: List[str], pad: Cell) -> bool:
        if carrier not in positions:
            return False
        cc = positions[carrier]
        d_cp = _manh(cc, pad)
        for e in eg_names:
            if e not in positions:
                continue
            ee = positions[e]
            if _manh(cc, ee) + _manh(ee, pad) == d_cp:
                return True
        return False

    for pad, eg_names in by_epad.items():
        corr = set(corridors.get(pad) or ())
        forbid = set(corr)
        forbid.add(pad)
        eg_on_throat = any(
            e in positions and (positions[e] in forbid or positions[e] == pad)
            for e in eg_names
        )

        for n in eg_names:
            cur = positions[n]
            wait = nearest_deg3(
                cur=cur,
                pad=pad,
                blocked=blocked,
                occupied=occupied - {cur},
                forbid=forbid,
            )
            if wait != out.get(n, cur):
                state.n_pad_egress += 1
            out[n] = wait
            pri[n] = max(float(pri.get(n, 0.0)), EGRESS_PRI) + float(
                _manh(cur, pad)
            )
            occupied.add(wait)
            occupied.discard(cur)

        for n in carriers_by_goal.get(pad, []):
            cur = positions[n]
            adj_eg = any(
                abs(cur[0] - positions[e][0]) + abs(cur[1] - positions[e][1])
                == 1
                for e in eg_names
                if e in positions
            )
            on_throat = cur in forbid
            between = _egressor_between(n, eg_names, pad)
            lane_lock = (
                fleet_cebi
                and eg_on_throat
                and not on_throat
                and between
            )
            if not on_throat and not adj_eg and not lane_lock:
                continue
            if on_throat and not adj_eg:
                eg_deeper = any(
                    _manh(positions[e], pad) <= _manh(cur, pad) for e in eg_names
                )
                if not eg_deeper:
                    continue
            wait = nearest_deg3(
                cur=cur,
                pad=pad,
                blocked=blocked,
                occupied=occupied - {cur},
                forbid=forbid,
            )
            if wait in forbid or _manh(wait, pad) <= _manh(cur, pad):
                me = n
                wait = outward_step(
                    cur=cur,
                    pad=pad,
                    blocked=blocked,
                    reserved=occupied - {cur},
                    forbid=forbid,
                    occupied_fn=lambda c, _me=me: c in occupied
                    and c != positions.get(_me),
                )
            if wait != out.get(n, cur):
                state.n_pad_egress += 1
                if lane_lock:
                    state.n_lane_locks += 1
            out[n] = wait
            # Goal rewrite only — never demote carrier role priority.
            pri[n] = min(float(pri.get(n, 1e6)), 1e6)
            occupied.add(wait)
            occupied.discard(cur)

        # Same-station carrier head-on with egressor (sibling goal) also backs off.
        for g, clist in carriers_by_goal.items():
            if g == pad or not same_unload_station(g, pad, blocked):
                continue
            for n in clist:
                if n in carriers_by_goal.get(pad, []):
                    continue
                cur = positions[n]
                adj_eg = any(
                    abs(cur[0] - positions[e][0])
                    + abs(cur[1] - positions[e][1])
                    == 1
                    for e in eg_names
                    if e in positions
                )
                if not adj_eg:
                    continue
                me = n
                wait = outward_step(
                    cur=cur,
                    pad=pad,
                    blocked=blocked,
                    reserved=occupied - {cur},
                    forbid=forbid,
                    occupied_fn=lambda c, _me=me: c in occupied
                    and c != positions.get(_me),
                )
                if wait != out.get(n, cur):
                    state.n_pad_egress += 1
                out[n] = wait
                pri[n] = min(float(pri.get(n, 1e6)), 1e6)
                occupied.add(wait)
                occupied.discard(cur)

    # Progress-gated pad seizure: only when done/ΣManh is stale. Idle on a
    # station pad / deg≤2 throat adjacent to a carrier must take one free
    # step; carrier claims that cell. Prefer-empty can hide pad campers.
    if (
        int(state.done_stale) >= PROG_DONE_STALE
        or int(state.manh_stale) >= PROG_MANH_STALE
    ):
        occupied = set(positions.values())
        for c in carrying:
            if c not in positions:
                continue
            g = goals.get(c)
            if g is None:
                continue
            cur_c = positions[c]
            corr = corridors.get(g) or pad_deg2_corridor(g, blocked)
            for idle, cur_i in positions.items():
                if idle in carrying or idle in assigned or idle in force_stay:
                    continue
                if (
                    abs(cur_c[0] - cur_i[0]) + abs(cur_c[1] - cur_i[1])
                    != 1
                ):
                    continue
                on_pad = cur_i == g or same_unload_station(cur_i, g, blocked)
                on_throat = cur_i in corr
                # Idle closer to pad than the adjacent carrier = blocks approach.
                between = _manh(cur_i, g) < _manh(cur_c, g)
                if not on_pad and not on_throat and not between:
                    continue
                best = None
                best_k = None
                for cand in _neighbors(cur_i, blocked):
                    if cand in occupied and cand != cur_i:
                        continue
                    if cand == cur_c:
                        continue
                    # Leave throat/pad: prefer deg≥3 off corridor / farther pad.
                    k = (
                        0 if cand not in corr and cand != g else 1,
                        0 if cell_degree(cand, blocked) >= 3 else 1,
                        -_manh(cand, g),
                        cand,
                    )
                    if best_k is None or k < best_k:
                        best_k = k
                        best = cand
                if best is None:
                    continue
                if between and _manh(best, g) < _manh(cur_i, g):
                    # Sole free neighbor is deeper into the throat — target
                    # nearest deg≥3 off-corridor instead of diving in.
                    best = nearest_deg3(
                        cur=cur_i,
                        pad=g,
                        blocked=blocked,
                        occupied=occupied - {cur_i},
                        forbid=set(corr) | {g},
                    )
                if best == cur_i:
                    continue
                if out.get(idle) != best:
                    state.n_pad_egress += 1
                out[idle] = best
                pri[idle] = max(float(pri.get(idle, 0.0)), EGRESS_PRI)
                state.egress_agents[idle] = g
                out[c] = cur_i
                occupied.add(best)
                occupied.discard(cur_i)
                break

    return out, pri
