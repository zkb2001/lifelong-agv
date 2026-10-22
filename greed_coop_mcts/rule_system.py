"""Realtime global traffic-rule system (RRSC road-rights layer).

- Each AGV is an agent (pose + deliver/pickup goal).
- AI outputs a one-way direction for every walkable cell.
- Directions + conflict heat form a guidance graph (edge costs / stay penalties)
  that constrains stage contracts and local greedy steps.
- Rules are recomputed when a short rollout predicts conflicts.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from greed_coop_mcts.motion import (
    pitch_for_step,
    turn_aware_next,
    turn_aware_search,
    turn_cost_ticks,
)
from greed_coop_mcts.rule_net import GlobalRuleNet, load_rule_net
from greed_coop_mcts.rule_obs import (
    DIR_DELTA,
    DIR_TO_PITCH,
    GRID,
    Pose,
    bfs_field,
    build_rule_state,
    goal_flow_teacher,
    in_arena,
    pitch_to_dir,
)
from greed_coop_mcts.stage_protocol import StageContract, StageRole

Cell = Tuple[int, int]


class RealtimeRuleSystem:
    def __init__(
        self,
        net: Optional[GlobalRuleNet] = None,
        *,
        device: str = "cpu",
        horizon: int = 12,
        min_replan_gap: int = 80,
        use_net: bool = True,
    ):
        self.device = device
        self.use_net = bool(use_net)
        if net is not None:
            self.net = net
        elif self.use_net:
            self.net = load_rule_net(device=device)
        else:
            self.net = None  # default prod: goal-flow teacher (RuleNet opt-in)
        self.horizon = int(horizon)
        # Large gap: thrashing the one-way field every ~15 ticks causes livelock.
        self.min_replan_gap = max(60, int(min_replan_gap))
        self.dirs: Optional[np.ndarray] = None  # (H,W) int
        self.stay_penalty: Optional[np.ndarray] = None  # (H,W) float
        self.last_conflict_heat: Optional[np.ndarray] = None
        self.last_plan_t = -10**9
        self.n_replans = 0
        self.n_conflict_triggers = 0
        self.n_yield_stages = 0
        self.stall = 0
        self._stable_goals: Dict[str, Cell] = {}
        self.contracts: Dict[str, StageContract] = {}
        self.last_contract_t = -10**9
        self.contract_gap = 12
        self.n_restage = 0
        # TERN / SoftGuidance overlays (set by solve; optional).
        self.soft_guide = None
        self.tern = None
        self.use_tern: bool = False
        self.n_agents_hint: int = 0
        # Residual only when solve sets lambda_tf > 0 (opt-in --tern).
        self.lambda_soft: float = 0.0
        self.lambda_tf: float = 0.0

    def refresh_guidance(
        self,
        *,
        blocked: Set[Cell],
        conflict_heat: Optional[np.ndarray] = None,
    ) -> None:
        """Derive stay_penalty (and cache heat) from dirs + conflict heat."""
        heat = conflict_heat
        if heat is None:
            heat = self.last_conflict_heat
        if heat is None:
            heat = np.zeros((GRID, GRID), dtype=np.float32)
        self.last_conflict_heat = np.asarray(heat, dtype=np.float32)
        stay = np.zeros((GRID, GRID), dtype=np.float32)
        for y in range(GRID):
            for x in range(GRID):
                if (x, y) in blocked:
                    stay[y, x] = 10.0
                    continue
                h = float(self.last_conflict_heat[y, x])
                # Occupied-looking / hot cells: discourage staging there.
                stay[y, x] = 0.15 + 1.5 * h
        self.stay_penalty = stay

    def edge_step_cost(self, cur: Cell, nxt: Cell) -> float:
        """Guidance edge cost: rule-align (+ optional TERN residual when enabled)."""
        x, y = cur
        nx, ny = nxt
        if self.dirs is None:
            base = 1.0
        else:
            rule = int(self.dirs[y, x]) if 0 <= y < GRID and 0 <= x < GRID else -1
            di = -1
            for i, (dx, dy) in enumerate(DIR_DELTA):
                if (x + dx, y + dy) == (nx, ny):
                    di = i
                    break
            if self.stall >= 80:
                off = 1.05
            elif self.stall >= 40:
                off = 1.25
            else:
                off = 1.6
            base = 1.0 if di == rule else off
            if self.stay_penalty is not None and 0 <= ny < GRID and 0 <= nx < GRID:
                base += 0.12 * float(self.stay_penalty[ny, nx])

        # SoftGuidance residual (open-floor fleets; λ set by solve geometry gate).
        if (
            self.soft_guide is not None
            and float(self.lambda_soft) > 0.0
            and self.stall < 40
        ):
            try:
                # edge_cost ∈ [1, 3.5]; use excess over 1 as residual.
                soft = float(self.soft_guide.edge_cost(cur, nxt)) - 1.0
                if soft > 0.0:
                    base += float(self.lambda_soft) * soft
            except Exception:  # noqa: BLE001
                pass

        # Opt-in only: residual under high heat. Defaults keep baseline costs.
        if (
            self.use_tern
            and self.tern is not None
            and self.stall < 20
            and self.lambda_tf > 0
        ):
            heat = self.last_conflict_heat
            mean_heat = float(heat.mean()) if heat is not None else 0.0
            if mean_heat >= 0.25:
                try:
                    base += 0.25 * float(self.tern.residual(cur, nxt))
                except Exception:  # noqa: BLE001
                    pass

        return float(base)

    def _walkable(self, blocked: Set[Cell], grid: int = GRID) -> np.ndarray:
        w = np.ones((grid, grid), dtype=np.float32)
        w[0, :] = 0.0
        w[:, 0] = 0.0
        for x, y in blocked:
            if 0 <= x < grid and 0 <= y < grid:
                w[y, x] = 0.0
        return w

    def infer(
        self,
        *,
        blocked: Set[Cell],
        poses: Dict[str, Pose],
        deliver_goals: Dict[str, Cell],
        pickup_goals: Dict[str, Cell],
        loaded: Set[str],
        conflict_heat: Optional[np.ndarray] = None,
        t: int = 0,
    ) -> np.ndarray:
        walk = self._walkable(blocked)
        # Completion-first: goal-flow is the primary field; net only mild prior.
        # Load/drop channels are in the obs (ch8/10/11) for RuleNet retrain;
        # do not override the merged field at runtime until a full v2 ckpt
        # beats the 20/20 baseline (naive load-aware thrash stuck SH17).
        goals = {**deliver_goals, **pickup_goals}
        self._stable_goals = dict(goals)
        flow = goal_flow_teacher(walk, blocked, goals, blend_circulation=0.10)
        blend = flow.copy()
        if self.use_net and self.net is not None and self.stall < 40:
            state = build_rule_state(
                blocked=blocked,
                poses=poses,
                deliver_goals=deliver_goals,
                pickup_goals=pickup_goals,
                loaded=loaded,
                conflict_heat=conflict_heat,
            )
            net_dirs = self.net.predict_dirs(state, walk, device=self.device)
            # Keep net vote only where flow is weak / ties (off-corridor cells).
            for y in range(GRID):
                for x in range(GRID):
                    if walk[y, x] < 0.5:
                        continue
                    if int(net_dirs[y, x]) >= 0 and int(flow[y, x]) < 0:
                        blend[y, x] = int(net_dirs[y, x])
        # Prolonged stall: pure goal-flow (ignore net entirely).
        if self.stall >= 40:
            blend = np.where(walk > 0.5, flow, -1).astype(np.int64)
        self.dirs = blend.astype(np.int64)
        self.refresh_guidance(blocked=blocked, conflict_heat=conflict_heat)
        self.last_plan_t = int(t)
        self.n_replans += 1
        return self.dirs

    def predict_conflict_heat(
        self,
        *,
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        blocked: Set[Cell],
        dirs: np.ndarray,
    ) -> np.ndarray:
        """Roll out agents along current one-way rules; mark overlapping cells."""
        heat = np.zeros((GRID, GRID), dtype=np.float32)
        # Simulate positions without turning cost (coarse conflict probe)
        sim = {n: (int(poses[n][0]), int(poses[n][1])) for n in poses}
        for _ in range(self.horizon):
            nxt: Dict[str, Cell] = {}
            claimed: Dict[Cell, str] = {}
            conflict_cells: Set[Cell] = set()
            for n, (x, y) in sim.items():
                d = int(dirs[y, x]) if 0 <= y < GRID and 0 <= x < GRID else -1
                step = (x, y)
                if d >= 0:
                    dx, dy = DIR_DELTA[d]
                    cand = (x + dx, y + dy)
                    if in_arena(cand) and cand not in blocked:
                        # only advance if not increasing manh to goal too much
                        g = goals.get(n)
                        if g is None or (
                            abs(cand[0] - g[0]) + abs(cand[1] - g[1])
                            <= abs(x - g[0]) + abs(y - g[1]) + 1
                        ):
                            step = cand
                if step in claimed and claimed[step] != n:
                    conflict_cells.add(step)
                else:
                    claimed[step] = n
                nxt[n] = step
            for c in conflict_cells:
                heat[c[1], c[0]] += 1.0
            # also mark swap conflicts
            for a, pa in sim.items():
                for b, pb in sim.items():
                    if a >= b:
                        continue
                    if nxt[a] == pb and nxt[b] == pa and pa != pb:
                        heat[pa[1], pa[0]] += 1.0
                        heat[pb[1], pb[0]] += 1.0
            sim = nxt
        if heat.max() > 0:
            heat = heat / float(heat.max())
        return heat

    def update_if_needed(
        self,
        *,
        t: int,
        blocked: Set[Cell],
        poses: Dict[str, Pose],
        deliver_goals: Dict[str, Cell],
        pickup_goals: Dict[str, Cell],
        loaded: Set[str],
        force: bool = False,
        stall: int = 0,
    ) -> np.ndarray:
        self.stall = int(stall)
        goals = {**pickup_goals, **deliver_goals}
        # Replan on goal set change, long period, or stall milestones — not every conflict.
        goals_changed = goals != self._stable_goals
        if stall > 0 and stall % 120 == 0:
            force = True
        if self.dirs is None or force or goals_changed or (t - self.last_plan_t) >= 100:
            return self.infer(
                blocked=blocked,
                poses=poses,
                deliver_goals=deliver_goals,
                pickup_goals=pickup_goals,
                loaded=loaded,
                t=t,
            )

        heat = self.predict_conflict_heat(
            poses=poses, goals=goals, blocked=blocked, dirs=self.dirs
        )
        self.refresh_guidance(blocked=blocked, conflict_heat=heat)
        # Only replan on sustained high heat + large gap (avoid field thrash).
        if float(heat.max()) >= 0.95 and (t - self.last_plan_t) >= self.min_replan_gap:
            self.n_conflict_triggers += 1
            return self.infer(
                blocked=blocked,
                poses=poses,
                deliver_goals=deliver_goals,
                pickup_goals=pickup_goals,
                loaded=loaded,
                conflict_heat=heat,
                t=t,
            )
        return self.dirs

    def bfs_path(
        self,
        start: Cell,
        goal: Cell,
        blocked: Set[Cell],
        *,
        max_len: int = 64,
        start_pitch: Optional[int] = None,
    ) -> List[Cell]:
        """Shortest path start→goal (cells after start); turn ticks if pitch given."""
        if start == goal:
            return []
        if start in blocked or goal in blocked:
            return []
        if start_pitch is not None:
            _c, _f, path = turn_aware_search(
                start, goal, blocked, int(start_pitch), max_expand=4000
            )
            return path[:max_len]
        field = bfs_field(goal, blocked)
        if start not in field:
            return []
        path: List[Cell] = []
        cur = start
        guard = 0
        while cur != goal and guard < max_len:
            guard += 1
            best = None
            best_d = field.get(cur, 10_000)
            for dx, dy in DIR_DELTA:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in blocked:
                    continue
                d = field.get(nxt)
                if d is None:
                    continue
                if d < best_d:
                    best_d = d
                    best = nxt
            if best is None:
                break
            path.append(best)
            cur = best
        return path

    def weighted_path(
        self,
        start: Cell,
        goal: Cell,
        blocked: Set[Cell],
        *,
        max_expand: int = 800,
        start_pitch: Optional[int] = None,
    ) -> List[Cell]:
        """Guidance-weighted shortest path; includes turn ticks when pitched."""
        if start == goal:
            return []
        if start in blocked or goal in blocked:
            return []
        if self.dirs is None:
            return self.bfs_path(
                start, goal, blocked, start_pitch=start_pitch
            )
        if start_pitch is not None:
            _c, _f, path = turn_aware_search(
                start,
                goal,
                blocked,
                int(start_pitch),
                move_cost=self.edge_step_cost,
                max_expand=max(max_expand * 4, 4000),
            )
            if path:
                return path
            return self.bfs_path(
                start, goal, blocked, start_pitch=start_pitch
            )

        dist: Dict[Cell, float] = {start: 0.0}
        prev: Dict[Cell, Optional[Cell]] = {start: None}
        pq: List[Tuple[float, Cell]] = [(0.0, start)]
        found = False
        expands = 0
        while pq and expands < max_expand:
            cost, cur = heapq.heappop(pq)
            expands += 1
            if cost > dist.get(cur, 1e18):
                continue
            if cur == goal:
                found = True
                break
            x, y = cur
            for dx, dy in DIR_DELTA:
                nxt = (x + dx, y + dy)
                if not in_arena(nxt):
                    continue
                if nxt in blocked:
                    continue
                # Skip cells marked non-walkable in dir field (except goal).
                if (
                    self.dirs[nxt[1], nxt[0]] < 0
                    and nxt != goal
                ):
                    continue
                nd = cost + self.edge_step_cost(cur, nxt)
                if nd < dist.get(nxt, 1e18):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(pq, (nd, nxt))
        if not found:
            return self.bfs_path(start, goal, blocked)
        cur: Optional[Cell] = goal
        path: List[Cell] = []
        while cur is not None and cur != start:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

    def _waypoints_from_path(
        self, path: List[Cell], goal: Cell, *, n_wp: int = 6
    ) -> List[Cell]:
        """Subsample path into stage waypoints ending at goal."""
        if not path:
            return [goal] if goal else []
        full = list(path)
        if not full or full[-1] != goal:
            full.append(goal)
        if len(full) <= n_wp:
            return full
        # Evenly spaced indices including last
        idxs = [
            int(round(i * (len(full) - 1) / max(1, n_wp - 1)))
            for i in range(n_wp)
        ]
        out: List[Cell] = []
        seen: Set[Cell] = set()
        for i in idxs:
            c = full[min(i, len(full) - 1)]
            if c not in seen:
                out.append(c)
                seen.add(c)
        if goal not in seen:
            out.append(goal)
        return out

    def _buffer_cell(
        self,
        cur: Cell,
        goal: Cell,
        blocked: Set[Cell],
        occupied: Set[Cell],
    ) -> Cell:
        """Low stay-penalty neighbor for YIELD/CLEAR staging (not final goal)."""
        best = cur
        best_score = 1e18
        for dx, dy in DIR_DELTA:
            cand = (cur[0] + dx, cur[1] + dy)
            if cand in blocked or cand in occupied:
                continue
            if not in_arena(cand):
                continue
            stay = (
                float(self.stay_penalty[cand[1], cand[0]])
                if self.stay_penalty is not None
                else 0.0
            )
            # Prefer not increasing manh too much, low stay cost.
            m = abs(cand[0] - goal[0]) + abs(cand[1] - goal[1])
            score = stay + 0.05 * m
            if score < best_score:
                best_score = score
                best = cand
        if best == cur:
            # Expand search ring
            for rad in (2, 3):
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        if abs(dx) + abs(dy) != rad:
                            continue
                        cand = (cur[0] + dx, cur[1] + dy)
                        if cand in blocked or cand in occupied or cand == goal:
                            continue
                        if not (1 <= cand[0] < GRID and 1 <= cand[1] < GRID):
                            continue
                        stay = (
                            float(self.stay_penalty[cand[1], cand[0]])
                            if self.stay_penalty is not None
                            else 0.0
                        )
                        score = stay + 0.1 * (abs(dx) + abs(dy))
                        if score < best_score:
                            best_score = score
                            best = cand
                if best != cur:
                    break
        return best

    def _path_headon(
        self,
        path_a: List[Cell],
        start_a: Cell,
        path_b: List[Cell],
        start_b: Cell,
        *,
        horizon: int = 6,
    ) -> bool:
        """True if A and B swap / meet head-on in the first horizon steps."""
        seq_a = [start_a] + list(path_a[:horizon])
        seq_b = [start_b] + list(path_b[:horizon])
        for t in range(min(len(seq_a), len(seq_b)) - 1):
            a0, a1 = seq_a[t], seq_a[t + 1]
            b0, b1 = seq_b[t], seq_b[t + 1]
            if a1 == b0 and b1 == a0 and a0 != b0:
                return True
            if a1 == b1 and a0 != b0:
                return True
        return False

    def issue_contracts(
        self,
        *,
        t: int,
        names: List[str],
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        blocked: Set[Cell],
        loaded: Set[str],
        assigned: Set[str],
        priority: Optional[Dict[str, float]] = None,
        force: bool = False,
        restage: Optional[Set[str]] = None,
        yield_cells: Optional[Dict[str, Cell]] = None,
        use_guidance: bool = True,
        stage_choices: Optional[Dict[str, object]] = None,
    ) -> Dict[str, StageContract]:
        """Global downlink: stage contracts constrained by guidance + StageNet.

        If ``stage_choices`` is provided (RRSC-v2), each agent's stage goal/role
        comes from the selector (busy agents included). Otherwise legacy rules.
        """
        pri = priority or {}
        need = force or (t - self.last_contract_t) >= self.contract_gap
        restage = set(restage or ())
        if not need and not restage and self.contracts:
            for n, c in list(self.contracts.items()):
                if n in poses:
                    c.advance(poses[n][:2])
                if c.expired(t) or not c.waypoints:
                    restage.add(n)
            if not restage:
                return self.contracts
        else:
            restage = set(names) if need else restage

        if self.dirs is not None and self.stay_penalty is None:
            self.refresh_guidance(blocked=blocked)

        yield_cells = dict(yield_cells or {})
        choices = stage_choices or {}
        order = sorted(
            restage,
            key=lambda n: (-float(pri.get(n, 0.0)), n),
        )
        planned_paths: Dict[str, List[Cell]] = {}
        occupied = {poses[n][:2] for n in names}

        for n in order:
            cur = poses[n][:2]
            final = goals.get(n, cur)
            prio = float(pri.get(n, 0.0))
            ch = choices.get(n)

            if ch is not None:
                # AI-selected stage target (may be FINAL / APPROACH / CLEAR).
                g = getattr(ch, "cell", final)
                role = getattr(ch, "role", None) or StageRole.IDLE
                if hasattr(ch, "priority"):
                    prio = float(ch.priority)
                path = (
                    self.weighted_path(cur, g, blocked)
                    if use_guidance
                    else self.bfs_path(cur, g, blocked)
                )
            elif n in yield_cells:
                role = StageRole.YIELD
                g = yield_cells[n]
                path = (
                    self.weighted_path(cur, g, blocked)
                    if use_guidance
                    else self.bfs_path(cur, g, blocked)
                )
            elif n in loaded:
                role = StageRole.TO_DROP
                g = final
                path = (
                    self.weighted_path(cur, g, blocked)
                    if use_guidance
                    else self.bfs_path(cur, g, blocked)
                )
            elif n in assigned:
                role = StageRole.TO_PICK
                g = final
                path = (
                    self.weighted_path(cur, g, blocked)
                    if use_guidance
                    else self.bfs_path(cur, g, blocked)
                )
            else:
                role = StageRole.IDLE
                g = final
                path = (
                    self.weighted_path(cur, g, blocked)
                    if use_guidance
                    else self.bfs_path(cur, g, blocked)
                )

            # Only IDLE without StageNet may be restaged to YIELD on head-on.
            if ch is None and role == StageRole.IDLE:
                for hi in planned_paths:
                    if float(pri.get(hi, 0.0)) < prio:
                        continue
                    if self._path_headon(
                        path, cur, planned_paths[hi], poses[hi][:2]
                    ):
                        buf = self._buffer_cell(
                            cur, g, blocked, occupied - {cur}
                        )
                        role = StageRole.YIELD
                        g = buf
                        path = (
                            self.weighted_path(cur, g, blocked)
                            if use_guidance
                            else self.bfs_path(cur, g, blocked)
                        )
                        self.n_yield_stages += 1
                        break

            planned_paths[n] = path
            wps = self._waypoints_from_path(path, g, n_wp=6)
            if role in (StageRole.YIELD, StageRole.CLEAR, StageRole.HOLD):
                wps = [g]
            ttl = 20 if role in (StageRole.TO_DROP, StageRole.TO_PICK) else 12
            self.contracts[n] = StageContract(
                agent=n,
                role=role,
                waypoints=wps,
                goal=g,
                priority=prio,
                ttl=ttl,
                exclusive_pad=final if role == StageRole.TO_DROP else None,
                issued_t=int(t),
            )
            if n in restage and not need:
                self.n_restage += 1

        for n in names:
            if n not in self.contracts:
                cur = poses[n][:2]
                g = goals.get(n, cur)
                self.contracts[n] = StageContract(
                    agent=n,
                    role=StageRole.IDLE,
                    waypoints=[g],
                    goal=g,
                    priority=float(pri.get(n, 0.0)),
                    ttl=12,
                    issued_t=int(t),
                )

        if need:
            self.last_contract_t = int(t)
        return self.contracts

    def directed_bfs_next(
        self,
        start: Cell,
        goal: Cell,
        blocked: Set[Cell],
        *,
        dirs: Optional[np.ndarray] = None,
        start_pitch: Optional[int] = None,
    ) -> Cell:
        """First step: one-way soft costs + turn ticks (when pitch known)."""
        field = dirs if dirs is not None else self.dirs
        if start == goal:
            return start
        if start in blocked:
            return start

        if self.stall >= 80:
            off = 1.05
        elif self.stall >= 40:
            off = 1.25
        else:
            off = 1.6

        def _move_cost(cur: Cell, nxt: Cell) -> float:
            if field is None:
                return 1.0
            x, y = cur
            rule = int(field[y, x]) if 0 <= y < GRID and 0 <= x < GRID else -1
            di = -1
            for i, (dx, dy) in enumerate(DIR_DELTA):
                if (x + dx, y + dy) == nxt:
                    di = i
                    break
            return 1.0 if di == rule else off

        if start_pitch is not None:
            # Soft-block rule-forbidden cells (except goal) like the old search.
            soft = set(blocked)
            if field is not None:
                for y in range(GRID):
                    for x in range(GRID):
                        if field[y, x] < 0 and (x, y) != goal:
                            soft.add((x, y))
            return turn_aware_next(
                start,
                goal,
                soft,
                int(start_pitch),
                move_cost=_move_cost,
            )

        if field is None:
            return turn_aware_next(start, goal, blocked, 0)

        dist: Dict[Cell, float] = {start: 0.0}
        prev: Dict[Cell, Optional[Cell]] = {start: None}
        pq: List[Tuple[float, Cell]] = [(0.0, start)]
        found = False
        while pq:
            cost, cur = heapq.heappop(pq)
            if cost > dist.get(cur, 1e18):
                continue
            if cur == goal:
                found = True
                break
            x, y = cur
            rule = int(field[y, x]) if 0 <= y < GRID and 0 <= x < GRID else -1
            for di, (dx, dy) in enumerate(DIR_DELTA):
                nxt = (x + dx, y + dy)
                if not in_arena(nxt):
                    continue
                if nxt in blocked:
                    continue
                if field[nxt[1], nxt[0]] < 0 and nxt != goal:
                    continue
                step = 1.0 if di == rule else off
                nd = cost + step
                if nd < dist.get(nxt, 1e18):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(pq, (nd, nxt))
        if not found:
            best = start
            best_m = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            for dx, dy in DIR_DELTA:
                nxt = (start[0] + dx, start[1] + dy)
                if nxt in blocked or not in_arena(nxt):
                    continue
                m = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                if m < best_m:
                    best_m = m
                    best = nxt
            return best
        cur: Optional[Cell] = goal
        path: List[Cell] = []
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path[1] if len(path) >= 2 else start

    def desired_cells(
        self,
        *,
        names: List[str],
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        blocked: Set[Cell],
        force_stay: Set[str],
        priority: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Cell]:
        assert self.dirs is not None
        out: Dict[str, Cell] = {}
        # Only reserve claimed next cells + force_stay. Do NOT treat all
        # current bodies as hard walls in BFS (that freezes corridors).
        reserved: Set[Cell] = {poses[n][:2] for n in force_stay}
        pri = priority or {}
        order = sorted(
            names,
            key=lambda n: (-float(pri.get(n, 0.0)), 0 if n in goals else 1, n),
        )
        for n in order:
            if n in force_stay:
                out[n] = poses[n][:2]
                continue
            cur = poses[n][:2]
            pitch = int(poses[n][2]) % 360
            g = goals.get(n, cur)
            soft_block = blocked | (reserved - {cur})
            nxt = self.directed_bfs_next(
                cur, g, soft_block, start_pitch=pitch
            )
            # Avoid stepping onto another reserved next-cell; body cells OK if
            # that agent is also moving away (resolved later by swap logic).
            if nxt in reserved and nxt != cur:
                # try alternate neighbor: manh + turn + rule soft bias
                alt = cur
                rule = int(self.dirs[cur[1], cur[0]]) if self.dirs is not None else -1
                best_score = 1e18
                for di, (dx, dy) in enumerate(DIR_DELTA):
                    cand = (cur[0] + dx, cur[1] + dy)
                    if cand in blocked or cand in reserved:
                        continue
                    if not in_arena(cand):
                        continue
                    need = pitch_for_step(cur, cand)
                    if need < 0:
                        continue
                    m = abs(cand[0] - g[0]) + abs(cand[1] - g[1])
                    score = (
                        float(m)
                        + float(turn_cost_ticks(pitch, need))
                        - (0.25 if di == rule else 0.0)
                    )
                    if score < best_score:
                        best_score = score
                        alt = cand
                nxt = alt
            out[n] = nxt
            reserved.add(nxt)
        return out
