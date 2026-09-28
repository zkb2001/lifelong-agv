"""Rule-based failure recovery: freq artery/parking + rule semaphore."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import (
    GRID,
    PARKING_THRESH,
    PLAY_HI,
    PLAY_LO,
    SEMAPHORE_THRESH,
)
from .features import bfs_dist_field, build_park_artery_channels, project_active_paths
from .rule_policy import RuleParkArteryPolicy

Cell = Tuple[int, int]
PolicyLike = Any


@dataclass
class RecoverAgentState:
    name: str
    unload: Optional[Cell] = None  # primary walkable unload pad
    unload_pads: List[Cell] = field(default_factory=list)
    unload_station: Optional[Cell] = None  # raw station (may be blocked)
    pickup: Optional[Cell] = None
    parked_at: Optional[Cell] = None
    phase: str = "need_pickup"  # need_pickup | go_park | waiting | go_unload | done
    task: Optional[dict] = None


@dataclass
class RecoveryController:
    """Attach to Simulation: on path/assign fail → park → semaphore → unload."""

    policy: PolicyLike
    agents: Dict[str, RecoverAgentState] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "triggers": 0,
        "parks": 0,
        "activations": 0,
        "holds": 0,
    })
    # One activation per unload pad per tick (FIFO by register order).
    _unload_token: Dict[Cell, str] = field(default_factory=dict)

    def observe_fields(self, sim, failed_names: Sequence[str]) -> dict:
        failed_pos: List[Cell] = []
        failed_unloads: List[Cell] = []
        occupied_parking: List[Cell] = []
        for n in failed_names:
            st = self.agents.get(n)
            agv = _agv_by_name(sim, n)
            if agv is not None and agv.state is not None:
                failed_pos.append((int(agv.state[0]), int(agv.state[1])))
            if st and st.unload:
                failed_unloads.append(st.unload)
            if st and st.parked_at:
                occupied_parking.append(st.parked_at)
        ch = build_park_artery_channels(
            sim,
            failed_positions=failed_pos,
            failed_unloads=failed_unloads,
            occupied_parking=occupied_parking,
        )
        return self.policy.infer(channels=ch, sim=sim)

    def register_failure(
        self,
        sim,
        agv_name: str,
        task: Optional[dict] = None,
    ) -> None:
        task = dict(task or {})
        primary, pads, station = resolve_unload_pads(sim, task)
        pickup = task.get("pickup_point")
        pk = (int(pickup[0]), int(pickup[1])) if pickup is not None else None
        if pk is not None:
            walk = sim_walkable(sim)
            pk = snap_walkable(walk, pk) or pk
        loaded = _is_loaded(sim, agv_name)
        phase = "go_park" if loaded or pk is None else "need_pickup"
        self.agents[str(agv_name)] = RecoverAgentState(
            name=str(agv_name),
            unload=primary,
            unload_pads=list(pads),
            unload_station=station,
            pickup=pk,
            phase=phase,
            task=task,
        )
        self.stats["triggers"] = int(self.stats.get("triggers") or 0) + 1

    def _pick_unload_winner(self, sim, unload: Cell) -> Optional[str]:
        """Among waiting agents targeting the same unload, pick closest / earliest."""
        cands: List[Tuple[int, int, str]] = []
        for i, (name, st) in enumerate(self.agents.items()):
            if st.phase != "waiting" or st.unload != unload:
                continue
            agv = _agv_by_name(sim, name)
            if agv is None:
                continue
            x, y = _cell(agv)
            manh = abs(x - unload[0]) + abs(y - unload[1])
            cands.append((manh, i, name))
        if not cands:
            return None
        cands.sort()
        return cands[0][2]

    def tick(self, sim) -> bool:
        """Run one recovery step for registered agents. Returns True if any action."""
        if not self.agents:
            return False
        names = list(self.agents.keys())
        fields = self.observe_fields(sim, names)
        parking = fields["parking"]
        active = project_active_paths(sim)
        acted = False
        self._unload_token.clear()
        for name, st in list(self.agents.items()):
            agv = _agv_by_name(sim, name)
            if agv is None:
                self.agents.pop(name, None)
                continue
            if st.phase == "done":
                self.agents.pop(name, None)
                continue
            # Skip if AGV already has a non-recovery path in progress
            tid = getattr(agv, "task_id", None)
            if tid and not str(tid).startswith(("escape_", "park_", "recover_")):
                # normal task progressing
                if st.phase in ("go_unload", "waiting") and _near_unload(agv, st):
                    st.phase = "done"
                    self.agents.pop(name, None)
                continue

            if st.phase == "need_pickup" and st.pickup is not None:
                if _cell(agv) == st.pickup or _is_loaded(sim, name):
                    st.phase = "go_park"
                else:
                    if _try_relocate(sim, agv, st.pickup, tag="recover_pickup"):
                        acted = True
                    continue

            if st.phase in ("go_park", "need_pickup"):
                goal = select_parking_cell(
                    parking=parking,
                    unload=st.unload,
                    walkable=fields.get("walkable"),
                    active_path=active,
                    occupied=_occupied(sim),
                    prefer_closer_than=_cell(agv),
                )
                if goal is None:
                    continue
                if _cell(agv) == goal:
                    st.parked_at = goal
                    st.phase = "waiting"
                    _mark_parked(sim, name, goal)
                    self.stats["parks"] = int(self.stats.get("parks") or 0) + 1
                    acted = True
                elif _try_relocate(sim, agv, goal, tag="park_hold"):
                    st.parked_at = goal
                    st.phase = "waiting"
                    _mark_parked(sim, name, goal)
                    self.stats["parks"] = int(self.stats.get("parks") or 0) + 1
                    acted = True
                continue

            if st.phase == "waiting":
                # Same unload: only the closest waiter may try an unload pad this tick.
                if st.unload is not None:
                    winner = self._pick_unload_winner(sim, st.unload)
                    if winner == name:
                        self._unload_token[st.unload] = name
                targets: List[Cell] = []
                if (
                    st.unload is not None
                    and self._unload_token.get(st.unload) == name
                ):
                    pad = _pick_free_unload_pad(sim, st, exclude=name)
                    if pad is not None:
                        targets.append(pad)
                nearer = select_parking_cell(
                    parking=parking,
                    unload=st.unload,
                    walkable=fields.get("walkable"),
                    active_path=active,
                    occupied=_occupied(sim),
                    prefer_closer_than=_cell(agv),
                    must_closer_than=_cell(agv),
                )
                if nearer is not None and nearer != st.parked_at:
                    targets.append(nearer)
                advanced = False
                for tgt in targets:
                    is_unload = bool(st.unload_pads) and tgt in st.unload_pads
                    if not is_unload and st.unload is not None and tgt == st.unload:
                        is_unload = True
                    pad_free = 1.0
                    if is_unload:
                        pad_free = 0.0 if _pad_occupied(sim, tgt, exclude=name) else 1.0
                    path_clear = 1.0 if _path_clear_to(sim, agv, tgt, active) else 0.0
                    allow = self.policy.semaphore_allow(
                        fields,
                        pos=_cell(agv),
                        goal=tgt,
                        pad_free=pad_free,
                        path_clear=path_clear,
                    )
                    if not allow:
                        continue
                    if is_unload:
                        if _try_relocate(sim, agv, tgt, tag="recover_unload"):
                            st.phase = "go_unload"
                            st.unload = tgt  # actual pad headed to
                            _unmark_parked(sim, name)
                            self.stats["activations"] = (
                                int(self.stats.get("activations") or 0) + 1
                            )
                            acted = True
                            advanced = True
                            break
                    else:
                        if _try_relocate(sim, agv, tgt, tag="park_advance"):
                            st.parked_at = tgt
                            _mark_parked(sim, name, tgt)
                            self.stats["activations"] = (
                                int(self.stats.get("activations") or 0) + 1
                            )
                            acted = True
                            advanced = True
                            break
                if not advanced:
                    self.stats["holds"] = int(self.stats.get("holds") or 0) + 1
                continue

            if st.phase == "go_unload":
                if st.unload and _cell(agv) == st.unload:
                    st.phase = "done"
                    self.agents.pop(name, None)
                    acted = True
                elif st.unload_pads and _cell(agv) in st.unload_pads:
                    st.phase = "done"
                    self.agents.pop(name, None)
                    acted = True
        return acted


def select_parking_cell(
    *,
    parking: np.ndarray,
    unload: Optional[Cell],
    walkable: Optional[np.ndarray] = None,
    active_path: Optional[np.ndarray] = None,
    occupied: Optional[Set[Cell]] = None,
    prefer_closer_than: Optional[Cell] = None,
    must_closer_than: Optional[Cell] = None,
    thresh: float = PARKING_THRESH,
) -> Optional[Cell]:
    """Pick parking argmin A* (BFS) distance to unload among high parking scores."""
    H, W = parking.shape
    if walkable is None:
        walkable = np.ones((H, W), dtype=np.float32)
        walkable[:PLAY_LO, :] = 0
        walkable[:, :PLAY_LO] = 0
        if PLAY_HI + 1 < H:
            walkable[PLAY_HI + 1 :, :] = 0
        if PLAY_HI + 1 < W:
            walkable[:, PLAY_HI + 1 :] = 0
    occupied = occupied or set()
    if unload is not None and walkable is not None:
        snapped = snap_walkable(walkable, unload)
        unload = snapped if snapped is not None else unload
    goals = [unload] if unload is not None else []
    if not goals:
        # fall back: any parking peak
        ys, xs = np.where(parking >= thresh)
        cands = [(int(x), int(y)) for y, x in zip(ys, xs)]
    else:
        dist = bfs_dist_field(walkable, goals, blocked=active_path)
        ys, xs = np.where(parking >= thresh)
        cands = []
        for y, x in zip(ys, xs):
            x, y = int(x), int(y)
            if (x, y) in occupied:
                continue
            if active_path is not None and float(active_path[y, x]) > 0.5:
                continue
            if not (PLAY_LO <= x <= PLAY_HI and PLAY_LO <= y <= PLAY_HI):
                continue
            if dist[y, x] < 0:
                continue
            cands.append((x, y))
        if not cands:
            return None
        if must_closer_than is not None and unload is not None:
            cur_d = dist[must_closer_than[1], must_closer_than[0]]
            cands = [
                c
                for c in cands
                if dist[c[1], c[0]] >= 0 and (cur_d < 0 or dist[c[1], c[0]] < cur_d)
            ]
            if not cands:
                return None

        def key(c: Cell):
            d = float(dist[c[1], c[0]])
            score = float(parking[c[1], c[0]])
            return (d, -score)

        cands.sort(key=key)
        return cands[0]

    # no unload: highest parking score farthest from prefer cell noise
    best = None
    best_s = -1.0
    for c in cands:
        if c in occupied:
            continue
        if active_path is not None and float(active_path[c[1], c[0]]) > 0.5:
            continue
        s = float(parking[c[1], c[0]])
        if s > best_s:
            best_s = s
            best = c
    return best


def attach_traffic_recovery(
    sim,
    policy: Optional[PolicyLike] = None,
    *,
    meta: Optional[dict] = None,
) -> RecoveryController:
    """Attach rule-based recovery (freq artery/parking + rule semaphore)."""
    if policy is None:
        if meta is not None:
            policy = RuleParkArteryPolicy.from_meta(meta)
        else:
            policy = RuleParkArteryPolicy.from_sim(sim, meta=meta)
    if meta is not None:
        sim._scene_meta = meta
        if meta.get("id"):
            sim._map_id = str(meta["id"])
    ctrl = RecoveryController(policy=policy)
    sim._traffic_recovery = ctrl
    sim._traffic_parked_cells = set()
    _install_engine_hooks(sim, ctrl)
    return ctrl


def maybe_trigger_from_sim(sim) -> None:
    """Call from engine when assign fails / path empty."""
    ctrl = getattr(sim, "_traffic_recovery", None)
    if ctrl is None:
        return
    fail_streak = int(getattr(sim, "_assign_fail_streak", 0) or 0)
    if fail_streak < 1:
        return
    # Register idle AGVs that have surface tasks but cannot proceed
    surface = getattr(sim, "surface_tasks", None) or {}
    for agv in getattr(sim, "agvs", []) or []:
        tid = getattr(agv, "task_id", None)
        if tid and not str(tid).startswith(("escape_", "park_", "recover_")):
            continue
        name = str(agv.name)
        if name in ctrl.agents:
            continue
        # pick any surface task as recovery target
        task = None
        if surface:
            task = next(iter(surface.values()))
        if task is None:
            continue
        ctrl.register_failure(sim, name, task)
    ctrl.tick(sim)


# ---- helpers ----

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def sim_walkable(sim) -> np.ndarray:
    walk = np.ones((GRID, GRID), dtype=np.float32)
    walk[:PLAY_LO, :] = 0
    walk[:, :PLAY_LO] = 0
    if PLAY_HI + 1 < GRID:
        walk[PLAY_HI + 1 :, :] = 0
        walk[:, PLAY_HI + 1 :] = 0
    env = getattr(sim, "env", None)
    if env is not None:
        for p in env.get_static_obstacles() or []:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < GRID and 0 <= y < GRID:
                walk[y, x] = 0.0
    return walk


def snap_walkable(walk: np.ndarray, cell: Cell) -> Optional[Cell]:
    x0, y0 = int(cell[0]), int(cell[1])
    H, W = walk.shape
    if 0 <= x0 < W and 0 <= y0 < H and walk[y0, x0] > 0.5:
        return (x0, y0)
    for r in range(1, 6):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) != r:
                    continue
                x, y = x0 + dx, y0 + dy
                if 0 <= x < W and 0 <= y < H and walk[y, x] > 0.5:
                    return (x, y)
    return None


def resolve_unload_pads(
    sim, task: dict
) -> Tuple[Optional[Cell], List[Cell], Optional[Cell]]:
    """Walkable unload pads for any map (station cell itself is often blocked)."""
    walk = sim_walkable(sim)
    pads: List[Cell] = []
    station: Optional[Cell] = None
    raw_station = task.get("unload_point")
    if raw_station is not None:
        station = (int(raw_station[0]), int(raw_station[1]))

    for e in task.get("end_points") or []:
        c = (int(e[0]), int(e[1]))
        s = snap_walkable(walk, c)
        if s is not None and s not in pads:
            pads.append(s)

    if station is not None:
        for dx, dy in DIR4:
            c = (station[0] + dx, station[1] + dy)
            s = snap_walkable(walk, c)
            if s is not None and s not in pads:
                pads.append(s)
        if not pads:
            s = snap_walkable(walk, station)
            if s is not None:
                pads.append(s)

    primary = pads[0] if pads else None
    return primary, pads, station


def _pick_free_unload_pad(
    sim, st: RecoverAgentState, *, exclude: str
) -> Optional[Cell]:
    pads = list(st.unload_pads or [])
    if st.unload is not None and st.unload not in pads:
        pads.insert(0, st.unload)
    if not pads:
        return None
    free = [p for p in pads if not _pad_occupied(sim, p, exclude=exclude)]
    pool = free or pads
    agv = _agv_by_name(sim, st.name)
    if agv is None:
        return pool[0]
    x, y = _cell(agv)
    pool.sort(key=lambda c: abs(c[0] - x) + abs(c[1] - y))
    return pool[0]


def _agv_by_name(sim, name: str):
    for a in getattr(sim, "agvs", []) or []:
        if str(a.name) == str(name):
            return a
    return None


def _cell(agv) -> Cell:
    st = agv.state
    return (int(st[0]), int(st[1]))


def _is_loaded(sim, name: str) -> bool:
    agv = _agv_by_name(sim, name)
    if agv is None:
        return False
    # crude: loaded flag on current path step
    t = int(getattr(sim, "time", 0) or 0)
    path = getattr(agv, "path", None) or []
    if path and t < len(path):
        step = path[t]
        if isinstance(step, dict):
            return str(step.get("loaded", "")).lower() in ("true", "1", "yes")
        if len(step) > 4:
            return bool(step[4])
    return False


def _occupied(sim) -> Set[Cell]:
    cells: Set[Cell] = set()
    for agv in getattr(sim, "agvs", []) or []:
        if agv.state is not None:
            cells.add((int(agv.state[0]), int(agv.state[1])))
    return cells


def _mark_parked(sim, name: str, cell: Cell) -> None:
    parked = getattr(sim, "_traffic_parked_cells", None)
    if parked is None:
        sim._traffic_parked_cells = set()
        parked = sim._traffic_parked_cells
    parked.add(cell)
    mapping = getattr(sim, "_traffic_parked_by_agv", None)
    if mapping is None:
        sim._traffic_parked_by_agv = {}
        mapping = sim._traffic_parked_by_agv
    mapping[str(name)] = cell


def _unmark_parked(sim, name: str) -> None:
    mapping = getattr(sim, "_traffic_parked_by_agv", None) or {}
    cell = mapping.pop(str(name), None)
    parked = getattr(sim, "_traffic_parked_cells", None)
    if cell is not None and parked is not None:
        parked.discard(cell)


def _pad_occupied(sim, pad: Cell, exclude: str = "") -> bool:
    for agv in getattr(sim, "agvs", []) or []:
        if str(agv.name) == str(exclude):
            continue
        if agv.state is not None and (int(agv.state[0]), int(agv.state[1])) == pad:
            return True
    return False


def _near_unload(agv, st: RecoverAgentState) -> bool:
    c = _cell(agv)
    if st.unload_pads and c in st.unload_pads:
        return True
    if st.unload is not None and abs(c[0] - st.unload[0]) + abs(c[1] - st.unload[1]) <= 1:
        return True
    if st.unload_station is not None:
        return abs(c[0] - st.unload_station[0]) + abs(c[1] - st.unload_station[1]) <= 1
    return False


def _path_clear_to(sim, agv, goal: Cell, active: np.ndarray) -> bool:
    """Cheap check: BFS avoiding active-path + other AGVs (except self)."""
    walk = sim_walkable(sim)
    start = _cell(agv)
    g = snap_walkable(walk, goal)
    if g is None:
        return False
    blocked = (active > 0.5).astype(np.float32)
    if 0 <= start[0] < GRID and 0 <= start[1] < GRID:
        blocked[start[1], start[0]] = 0.0
    for other in getattr(sim, "agvs", []) or []:
        if other is agv or other.state is None:
            continue
        ox, oy = int(other.state[0]), int(other.state[1])
        if 0 <= ox < GRID and 0 <= oy < GRID:
            blocked[oy, ox] = 1.0
    dist = bfs_dist_field(walk, [g], blocked=blocked)
    return float(dist[start[1], start[0]]) >= 0


def _try_relocate(sim, agv, goal: Cell, *, tag: str) -> bool:
    if not hasattr(sim, "_relocate_idle_agv"):
        return False
    walk = sim_walkable(sim)
    g = snap_walkable(walk, goal)
    if g is None:
        return False
    occupied, _ = sim._occupied_cells_now() if hasattr(sim, "_occupied_cells_now") else (set(), set())
    ok = bool(sim._relocate_idle_agv(agv, g, occupied))
    if ok:
        if agv.task_id and str(agv.task_id).startswith("escape_"):
            agv.task_id = f"{tag}_{agv.name}"
    return ok


def _install_engine_hooks(sim, ctrl: RecoveryController) -> None:
    """Override get_safe_parking_points + wrap time_forward recovery tick."""

    def _safe_parking():
        try:
            fields = ctrl.observe_fields(sim, list(ctrl.agents.keys()) or [])
            park = fields["parking"]
            ys, xs = np.where(park >= PARKING_THRESH)
            return [(int(x), int(y)) for y, x in zip(ys, xs)]
        except Exception:  # noqa: BLE001
            return []

    sim.get_safe_parking_points = _safe_parking  # type: ignore[method-assign]

    orig_pick = getattr(sim, "_pick_holding_cell", None)

    def _pick_holding(pos, forbidden, occupied, static_obs, min_r=2, max_r=4):
        try:
            fields = ctrl.observe_fields(sim, list(ctrl.agents.keys()) or [])
            unload = None
            # nearest unload pad if any
            if hasattr(sim, "_unload_pad_set"):
                pads = list(sim._unload_pad_set() or [])
                if pads:
                    unload = min(
                        pads,
                        key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]),
                    )
            goal = select_parking_cell(
                parking=fields["parking"],
                unload=unload,
                active_path=project_active_paths(sim),
                occupied=set(occupied) | set(forbidden),
                prefer_closer_than=tuple(pos),
            )
            if goal is not None:
                return goal
        except Exception:  # noqa: BLE001
            pass
        if orig_pick is not None:
            return orig_pick(pos, forbidden, occupied, static_obs, min_r=min_r, max_r=max_r)
        return None

    sim._pick_holding_cell = _pick_holding  # type: ignore[method-assign]
