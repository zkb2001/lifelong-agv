"""Fast livelock detection + AI-guided corridor unblock for lifelong MAPD.

Detection (does not wait for stall=300):
  Tracks per-AGV cell motion over a short window. A frozen fleet with
  in-flight carriers/approachers scores as livelock within ~40–80 ticks.

Unblock (CostTracer/BFS guided, not random exile):
  Builds each carrier's shortest path, ranks blockers by path occupancy,
  releases blocking approachers, and assigns escape goals that clear the
  throat while staying reachable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from greed_coop_mcts.motion import bfs_dist_field

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]  # x, y, pitch


def _manh(a: Cell, b: Cell) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _bfs_path(start: Cell, goal: Cell, blocked: Set[Cell]) -> List[Cell]:
    if start == goal:
        return []
    if start in blocked or goal in blocked:
        return []
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    q: deque = deque([start])
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if not found:
        return []
    out: List[Cell] = []
    cur: Optional[Cell] = goal
    while cur is not None and cur != start:
        out.append(cur)
        cur = parent.get(cur)
    out.reverse()
    return out


@dataclass
class LivelockFeatures:
    freeze_frac: float = 0.0
    n_frozen: int = 0
    n_agents: int = 0
    n_carry: int = 0
    n_approach: int = 0
    pending: int = 0
    stall: int = 0
    pose_repeats: int = 0
    carrier_blocked: float = 0.0  # 0..1 fraction of carriers with neighbor occupied
    score: float = 0.0


@dataclass
class UnblockPlan:
    release: List[str] = field(default_factory=list)
    escape: Dict[str, Cell] = field(default_factory=dict)
    boost_carriers: List[str] = field(default_factory=list)
    critical_cells: Set[Cell] = field(default_factory=set)
    reason: str = ""


class LivelockDetector:
    """Online livelock scorer. Fires much earlier than stall=300."""

    def __init__(
        self,
        *,
        window: int = 48,
        score_th: float = 0.72,
        min_freeze_ticks: int = 36,
        cooldown: int = 100,
    ):
        self.window = int(window)
        self.score_th = float(score_th)
        self.min_freeze_ticks = int(min_freeze_ticks)
        self.cooldown = int(cooldown)
        self._pos_hist: Dict[str, deque] = {}
        self._pose_sig_hist: deque = deque(maxlen=max(8, window // 2))
        self._last_fire_t = -10_000
        self.last_features = LivelockFeatures()

    def reset(self) -> None:
        self._pos_hist.clear()
        self._pose_sig_hist.clear()
        self.last_features = LivelockFeatures()

    def observe(
        self,
        *,
        t: int,
        poses: Dict[str, Pose],
        carrying: Set[str],
        assigned: Set[str],
        pending: int,
        stall: int,
    ) -> LivelockFeatures:
        # Update per-agent position histories.
        for n, pose in poses.items():
            cell = (int(pose[0]), int(pose[1]))
            hist = self._pos_hist.setdefault(n, deque(maxlen=self.window))
            hist.append(cell)

        names = list(poses.keys())
        n_agents = max(1, len(names))
        frozen = 0
        for n in names:
            hist = self._pos_hist.get(n)
            if not hist or len(hist) < self.min_freeze_ticks:
                continue
            recent = list(hist)[-self.min_freeze_ticks :]
            if len(set(recent)) <= 1:
                frozen += 1

        freeze_frac = frozen / float(n_agents)
        sig = tuple(sorted((n, int(poses[n][0]), int(poses[n][1])) for n in names))
        self._pose_sig_hist.append(sig)
        repeats = 0
        if len(self._pose_sig_hist) >= 8:
            last = self._pose_sig_hist[-1]
            for s in list(self._pose_sig_hist)[-16:]:
                if s == last:
                    repeats += 1

        # Carriers with all 4-neighbors occupied (geometric jam).
        occ = {(int(poses[n][0]), int(poses[n][1])) for n in names}
        blocked_carriers = 0
        for n in carrying:
            if n not in poses:
                continue
            x, y = int(poses[n][0]), int(poses[n][1])
            nbrs = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            free_n = sum(
                1
                for c in nbrs
                if 1 <= c[0] <= 20
                and 1 <= c[1] <= 20
                and c not in occ
            )
            if free_n == 0:
                blocked_carriers += 1
        carrier_blocked = (
            blocked_carriers / float(max(1, len(carrying))) if carrying else 0.0
        )

        # Linear score (hand-tuned prior; acts as a tiny "AI" classifier).
        # High freeze + in-flight work + trapped carriers → livelock.
        score = 0.0
        score += 0.45 * freeze_frac
        score += 0.15 * min(1.0, repeats / 12.0)
        score += 0.20 * carrier_blocked
        if carrying:
            score += 0.10
        if assigned:
            score += 0.08
        if pending > 0 and (carrying or assigned):
            score += 0.07
        # Mild stall contribution (not dominant — we want early fire).
        score += 0.05 * min(1.0, float(stall) / 120.0)

        feat = LivelockFeatures(
            freeze_frac=freeze_frac,
            n_frozen=frozen,
            n_agents=n_agents,
            n_carry=len(carrying),
            n_approach=len(assigned),
            pending=int(pending),
            stall=int(stall),
            pose_repeats=repeats,
            carrier_blocked=carrier_blocked,
            score=score,
        )
        self.last_features = feat
        return feat

    def should_unlock(self, *, t: int, feat: Optional[LivelockFeatures] = None) -> bool:
        f = feat or self.last_features
        if int(t) - int(self._last_fire_t) < int(self.cooldown):
            return False
        if f.n_carry <= 0:
            return False
        # Need enough history.
        if f.n_frozen < max(2, f.n_agents // 3) and f.freeze_frac < 0.45:
            return False
        if f.score < self.score_th:
            return False
        # Require either trapped carrier or heavy freeze with approachers.
        if f.carrier_blocked < 0.34 and f.freeze_frac < 0.55 and f.n_approach == 0:
            return False
        return True

    def mark_fired(self, t: int) -> None:
        self._last_fire_t = int(t)


def plan_unblock(
    *,
    names: Sequence[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    carrying: Set[str],
    assigned: Set[str],
    blocked: Set[Cell],
    free: Set[Cell],
    escape_ttl: int = 64,
) -> UnblockPlan:
    """Rank blockers on carrier shortest paths; evacuate them off the throat."""
    plan = UnblockPlan(boost_carriers=list(carrying))
    if not carrying:
        plan.reason = "no_carrier"
        return plan

    critical: Set[Cell] = set()
    for n in carrying:
        if n not in poses:
            continue
        cur = (int(poses[n][0]), int(poses[n][1]))
        g = goals.get(n, cur)
        critical.add(cur)
        path = _bfs_path(cur, g, blocked)
        critical.update(path[:14])  # near-term throat

    plan.critical_cells = set(critical)
    occ = {(int(poses[n][0]), int(poses[n][1])) for n in names if n in poses}

    def block_score(n: str) -> Tuple[float, str]:
        if n not in poses or n in carrying:
            return (-1.0, n)
        cell = (int(poses[n][0]), int(poses[n][1]))
        on = 1.0 if cell in critical else 0.0
        near = 0.0
        if critical:
            dmin = min(_manh(cell, c) for c in critical)
            near = 1.0 / float(1 + dmin)
        role = 0.35 if n in assigned else 0.15  # approachers more disposable
        # Prefer freeing cells immediately in front of carriers.
        front = 0.0
        for cname in carrying:
            if cname not in poses:
                continue
            cc = (int(poses[cname][0]), int(poses[cname][1]))
            if _manh(cell, cc) == 1:
                front = 0.5
                break
        return (on * 1.2 + near * 0.8 + role + front, n)

    ranked = sorted((block_score(n) for n in names), reverse=True)
    # Release approachers that score as blockers.
    for sc, n in ranked:
        if sc < 0.55:
            break
        if n in assigned:
            plan.release.append(n)

    # Also release any approacher sitting on critical cells.
    for n in list(assigned):
        if n in plan.release or n not in poses:
            continue
        cell = (int(poses[n][0]), int(poses[n][1]))
        if cell in critical:
            plan.release.append(n)

    # Escape goals: maximize clearance from critical path (reachable).
    # Build a repulsion field = min dist to any critical cell (inverted).
    free_pool = [c for c in free if c not in occ]
    # Allow current cell of evacuees into pool after they leave conceptually.
    for n in names:
        if n in carrying or n not in poses:
            continue
        cell = (int(poses[n][0]), int(poses[n][1]))
        sc, _ = block_score(n)
        if sc < 0.45 and n not in plan.release:
            continue
        if not free_pool:
            break
        # Prefer cells far from critical, not on critical columns of carriers.
        carrier_cols = {int(poses[c][0]) for c in carrying if c in poses}

        def key(c: Cell, _cell: Cell = cell) -> Tuple:
            dcrit = min((_manh(c, k) for k in critical), default=0) if critical else 0
            return (
                0 if c in critical else 1,
                0 if c[0] in carrier_cols else 1,
                dcrit,
                -_manh(c, _cell),
                c,
            )

        dest = max(free_pool, key=key)
        # Reachability check via BFS field from dest.
        field = bfs_dist_field(dest, blocked)
        if cell not in field and dest != cell:
            # Fallback: any reachable free cell maximizing clearance.
            cands = [c for c in free_pool if c in bfs_dist_field(cell, blocked)]
            if not cands:
                continue
            dest = max(cands, key=key)
        plan.escape[n] = dest
        free_pool = [c for c in free_pool if c != dest]

    plan.reason = (
        f"crit={len(critical)} release={len(plan.release)} "
        f"escape={len(plan.escape)}"
    )
    return plan
