"""Independent lifelong MAPD: CostTracer prior + greed–coop neural MCTS.

Lifelong MAPD shell (same as PIBT-MAPD):
  - FIFO surface heads (never reorder queues)
  - one approacher per pickup station
  - pickup on approach door + 1s no-turn dwell
  - unload on legal dropoff neighbor + 1s dwell
  - any turn (90°/180°) completes in 1s; no same-tick move+turn
  - path search edge cost = turn(0|1) + move; conflict waits not in search

Motion: each AGV is an agent. Crowded steps use NeuralLocalMCTS with an
explicit greed–cooperation leaf reward (α progress − β conflict + γ yield);
distant agents follow CostTracer/teacher greedily (MATS-LP style).
Independent package: no ECBS / Hier / A* solvers.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

try:
    import torch as _torch

    _torch.set_num_threads(1)
except Exception:  # noqa: BLE001
    pass

from greed_coop_mcts.mcts import (
    LocalAgent,
    MCTSConfig,
    NeuralLocalMCTS,
)
from greed_coop_mcts.motion import (
    bfs_dist_field,
    pitch_for_step,
    turn_aware_next,
    turn_aware_search,
    turn_pitch_chain,
)
from greed_coop_mcts.network import COSTTRACER_CKPT, CostTracerNet, load_cost_tracer
from greed_coop_mcts.obs import apply_action
from greed_coop_mcts.paths import RESULTS, TRAJ as TRAJ_DIR
from greed_coop_mcts.livelock_ai import LivelockDetector, plan_unblock
from greed_coop_mcts.runtime import pick_ego_crowded
from greed_coop_mcts.scenario import (
    TRAJ_HEADER,
    load_scenario,
    load_slot_meta,
    patch_extra_obstacles,
)
from greed_coop_mcts.glocal_rh import (
    PathCommit,
    RHStats,
    advance_commits_after_move,
    commits_to_desired,
    plan_committed,
    sketches_from_commits,
    staging_goals,
    window_repair,
)
from greed_coop_mcts.local_agent import pick_yield_pocket, plan_all_local
from greed_coop_mcts.policy_xfmr import get_transformer, plan_desired as xfmr_plan
from greed_coop_mcts.region_admit import (
    apply_region_admit,
    claim_region_ok,
    count_free_by_region,
    fleet_hard_cap_default,
    occupancy_from_goals,
    region_caps,
)
from greed_coop_mcts.global_altruism import (
    AltruismState,
    apply_focal_altruism,
    focal_gap_ok,
    should_pgfa,
)
from greed_coop_mcts.meeting_rh import (
    MEET_H,
    MEET_W,
    MeetingStats,
    adaptive_meet_w,
    apply_idle_corridor_clear,
    collect_busy_path,
    follow_contract_step,
    idle_clear_goal,
    legalize_desired,
    local_gap_yield,
    run_meeting,
    should_idle_corridor_clear,
    should_meet,
)
from greed_coop_mcts.pgcp import (
    ProgressState,
    apply_idle_hygiene,
    apply_pad_egress,
    cell_degree,
    e4_sticky_enabled,
    effective_fleet_cap,
    rewrite_contested_destinations,
    trigger_progress_response,
    update_progress,
)
from greed_coop_mcts.rrsc_unified import SoftGuidance
from greed_coop_mcts.rule_net import RULE_CKPT
from greed_coop_mcts.rule_obs import seal_ghost_border
from greed_coop_mcts.rule_system import RealtimeRuleSystem
from greed_coop_mcts.stage_net import (
    KIND_CLEAR,
    KIND_FINAL,
    StageChoice,
    StageSelector,
    region_id,
)
from greed_coop_mcts.stage_protocol import (
    CommStats,
    IntentSketch,
    StageContract,
    StageRole,
    arbitrate_intents,
)
from greed_coop_mcts.teacher import greedy_action
from greed_coop_mcts.traffic_rules import (
    TempTask,
    assign_temp_tasks,
    learn_traffic_maps,
    prefer_dir_action,
)
from greed_coop_mcts.transformer_net import MapdTransformer
from greed_coop_mcts.validate import format_validation_summary, validate_trajectory

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

OUT = RESULTS
TRAJ = TRAJ_DIR

_HUB13 = "Beijing"
_HUB13_PADS = {(7, 4), (6, 5)}

_NET_CACHE: Dict[str, Optional[CostTracerNet]] = {}


def _pickup_approach(cell: Cell) -> Cell:
    x, y = int(cell[0]), int(cell[1])
    if x <= 1:
        return (x + 1, y)
    if x >= 20:
        return (x - 1, y)
    return (x, y)


def _neighbors4(cell: Cell) -> List[Cell]:
    x, y = int(cell[0]), int(cell[1])
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _restrict_hub13(dest: str, pads: List[Cell]) -> List[Cell]:
    """SH13-only hub clamp. Never invent pads on other maps.

    Previously, any destination named ``Beijing`` fell back to hardcoded
    ``(7,4)/(6,5)`` when geometric pads missed the hub — breaking synth maps
    that reuse city names (mass premature_unload / INVALID).
    """
    if str(dest) != _HUB13:
        return pads
    kept = [p for p in pads if p in _HUB13_PADS]
    # Only clamp when this map actually exposes hub pad geometry (SH13).
    return kept if kept else pads


def _get_net(device: str) -> Optional[CostTracerNet]:
    key = str(device)
    if key not in _NET_CACHE:
        try:
            _NET_CACHE[key] = (
                load_cost_tracer(device=device) if COSTTRACER_CKPT.exists() else None
            )
        except Exception:  # noqa: BLE001
            _NET_CACHE[key] = None
    return _NET_CACHE[key]


def _need_mcts(pos: Dict[str, Cell], active: List[str], radius: int = 3) -> bool:
    cells = [pos[n] for n in active]
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if _manh(a, b) <= radius:
                return True
    return False


def _adaptive_greed_coop(
    stall: int,
    greed_w: float,
    coop_w: float,
    yield_w: float,
) -> Tuple[float, float, float]:
    """Shift toward greed under prolonged stall to break yield livelock."""
    if stall < 80:
        return greed_w, coop_w, yield_w
    if stall < 200:
        # Mild: less yielding, slightly more greed.
        return (
            min(0.85, greed_w + 0.15),
            max(0.10, coop_w - 0.10),
            max(0.0, yield_w - 0.05),
        )
    # Hard stall: almost pure personal progress + conflict hard-stop only.
    return (
        min(0.90, greed_w + 0.30),
        max(0.08, coop_w - 0.20),
        0.0,
    )


def _mats_desired(
    *,
    names: List[str],
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    net: Optional[CostTracerNet],
    num_simulations: int,
    k_proximal: int,
    device: str,
    seed: int,
    greed_w: float = 0.55,
    coop_w: float = 0.35,
    yield_w: float = 0.10,
    force_mcts: bool = False,
) -> Dict[str, Cell]:
    """One-step desired cells via greed–coop neural MCTS (crowded) + greedy."""
    pos = {n: pose[n][:2] for n in names}
    desired: Dict[str, Cell] = {}
    active = [n for n in names if n not in force_stay]
    agents = [
        LocalAgent(name=n, pos=pos[n], goal=goals.get(n)) for n in active
    ]

    acts: Dict[str, str] = {}
    if agents and (force_mcts or _need_mcts(pos, active)):
        ordered = pick_ego_crowded(agents)
        planner = NeuralLocalMCTS(
            net=net,
            cfg=MCTSConfig(
                num_simulations=int(num_simulations),
                horizon=4,
                k_proximal=int(k_proximal),
                greed_w=float(greed_w),
                coop_w=float(coop_w),
                yield_w=float(yield_w),
                seed=int(seed) % 997,
            ),
            device=device,
        )
        acts = planner.plan(ordered, list(blocked))

    for n in names:
        if n in force_stay:
            desired[n] = pos[n]
            continue
        others = [pos[m] for m in names if m != n]
        if n in acts:
            desired[n] = apply_action(pos[n], acts[n], blocked)
        else:
            ga = greedy_action(pos[n], goals.get(n), others, blocked)
            desired[n] = apply_action(pos[n], ga, blocked)
    return desired


def solve(
    slot: int,
    *,
    meta: Optional[dict] = None,
    task_csv: Optional[Path] = None,
    max_sim_time: int = 200000,
    wall_timeout: float = 1800.0,
    progress_every: int = 500,
    num_simulations: int = 24,
    k_proximal: int = 3,
    greed_w: float = 0.55,
    coop_w: float = 0.35,
    yield_w: float = 0.10,
    use_traffic_rules: bool = True,
    policy: str = "auto",  # auto | rrsc | glocal | rules | xfmr | mcts
    use_stage_net: bool = True,
    force_stage_net: bool = False,
    use_rule_net: bool = False,
    use_cgpn: bool = False,
    use_facn: bool = True,
    force_fleet_cap: Optional[int] = None,
    device: str = "cpu",
    lifelong_random: bool = False,
    lifelong_seed: int = 0,
    lifelong_batch: int = 24,
    lifelong_low_water: int = 8,
    lifelong_max_total: Optional[int] = None,
) -> dict:
    t0 = time.perf_counter()
    if meta is None:
        meta = load_slot_meta(
            int(slot),
            max_sim_time=int(max_sim_time),
            wall_timeout=float(wall_timeout),
            n_tasks=100,
        )
        assert meta, f"missing meta for slot {slot}"
    meta = dict(meta)
    sid = str(meta.get("id") or f"SH{int(slot):02d}_gcm")
    meta["id"] = sid

    tcsv = Path(task_csv) if task_csv else Path(meta["task_csv"])
    meta["task_csv"] = str(tcsv.resolve())
    mod, env, agv_states, task_states, _ = load_scenario(
        tcsv, Path(meta["position_csv"]), force_reload=True
    )
    del mod, task_states
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))

    static_list = list(env.get_static_obstacles() or [])
    static_all = {(int(p[0]), int(p[1])) for p in static_list}

    pickups: Dict[str, Cell] = {}
    dropoffs: Dict[str, Cell] = {}
    jpath = Path(meta.get("map_json") or "")
    if jpath.is_file():
        j = json.loads(jpath.read_text(encoding="utf-8"))
        for p in j.get("pickups") or []:
            pickups[str(p["name"])] = (int(p["x"]), int(p["y"]))
        for p in j.get("dropoffs") or []:
            dropoffs[str(p["name"])] = (int(p["x"]), int(p["y"]))

    stations = set(pickups.values()) | set(dropoffs.values())
    walls = set(static_all) - set(stations)
    # Seal ghost border x=0/y=0: planner tensors use GRID=21 (coords 0..20) but
    # the competition validator only accepts [1,20]. Training already seals this;
    # runtime must too or RRSC/local will path along the illegal rim.
    blocked = seal_ghost_border(set(walls) | set(stations))
    free = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in blocked
    }
    dist_cache: Dict[Cell, Dict[Cell, int]] = {}

    def _field(goal: Cell) -> Dict[Cell, int]:
        if goal not in dist_cache:
            dist_cache[goal] = bfs_dist_field(goal, blocked)
        return dist_cache[goal]

    def _spath(a: Cell, b: Cell, *, pitch: Optional[int] = None) -> int:
        """Path length in ticks; with pitch, includes turn costs."""
        if pitch is not None:
            cost, _, _ = turn_aware_search(a, b, blocked, int(pitch))
            if cost < 1e17:
                return int(round(cost))
        field = _field(b)
        if a in field:
            return int(field[a])
        return 10_000 + _manh(a, b)

    def _bfs_next(
        cur: Cell,
        goal: Cell,
        *,
        reserved: Set[Cell],
        pitch: int,
    ) -> Cell:
        """Obstacle-aware one-step; edge cost = turn_ticks + 1 move."""
        if cur == goal:
            return cur
        return turn_aware_next(
            cur,
            goal,
            blocked,
            int(pitch),
            reserved=reserved,
        )

    def unload_pads(dest_name: str) -> List[Cell]:
        center = dropoffs.get(str(dest_name))
        if not center:
            return []
        pads = [
            c
            for c in _neighbors4(center)
            if 1 <= c[0] <= 20 and 1 <= c[1] <= 20 and c not in blocked
        ]
        return _restrict_hub13(dest_name, pads)

    queues: Dict[str, List[dict]] = defaultdict(list)
    with tcsv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = str(row.get("start_point") or "").strip()
            tid_s = str(row.get("task_id") or "").strip()
            dest_s = str(row.get("end_point") or "").strip()
            if not st or not tid_s:
                continue
            queues[st].append(
                {
                    "task_id": tid_s,
                    "start_point": st,
                    "destination": dest_s,
                    "end_point": dest_s,
                    "priority": row.get("priority") or "Normal",
                }
            )
    total = sum(len(v) for v in queues.values())
    initial_total = int(total)
    station_names = sorted(pickups.keys()) or sorted(queues.keys())
    dest_names = sorted(dropoffs.keys())
    if not dest_names:
        dest_names = sorted(
            {
                str(t.get("destination") or t.get("end_point") or "")
                for qq in queues.values()
                for t in qq
                if str(t.get("destination") or t.get("end_point") or "")
            }
        )
    life_rng = random.Random(int(lifelong_seed) if lifelong_seed else int(slot) * 10007 + 3)
    life_counters: Dict[str, int] = {}
    for st, qq in queues.items():
        for t in qq:
            tid0 = str(t.get("task_id") or "")
            if "-" in tid0 and tid0.rsplit("-", 1)[-1].isdigit():
                pref, num = tid0.rsplit("-", 1)
                life_counters[pref] = max(int(life_counters.get(pref, 0)), int(num))
    n_streamed = 0
    stream_on = False

    def _queue_pending() -> int:
        return sum(len(v) for v in queues.values())

    def _inflight() -> int:
        return len(assigned) + len(carrying)

    def _spawn_random_task() -> dict:
        st = life_rng.choice(station_names)
        dest_s = life_rng.choice(dest_names)
        life_counters[st] = int(life_counters.get(st, 0)) + 1
        return {
            "task_id": f"{st}-{life_counters[st]}",
            "start_point": st,
            "destination": dest_s,
            "end_point": dest_s,
            "priority": "Normal",
        }

    def _maybe_refill_lifelong() -> None:
        nonlocal total, n_streamed, stream_on
        if not lifelong_random:
            return
        if not station_names or not dest_names:
            return
        if done >= initial_total and not stream_on:
            stream_on = True
        if not stream_on:
            return
        if lifelong_max_total is not None and total >= int(lifelong_max_total):
            return
        if _queue_pending() + _inflight() >= int(lifelong_low_water):
            return
        batch = max(1, int(lifelong_batch))
        if lifelong_max_total is not None:
            batch = min(batch, max(0, int(lifelong_max_total) - int(total)))
        for _ in range(batch):
            task = _spawn_random_task()
            queues[str(task["start_point"])].append(task)
            total += 1
            n_streamed += 1

    pose: Dict[str, Pose] = {}
    home: Dict[str, Cell] = {}
    for name, agv in agv_states.items():
        st = agv["state"] if isinstance(agv, dict) else agv.state
        pose[name] = (int(st[0]), int(st[1]), int(st[3]) if len(st) > 3 else 90)
        home[name] = (int(st[0]), int(st[1]))
    names = sorted(pose)

    assigned: Dict[str, dict] = {}
    carrying: Dict[str, dict] = {}
    loaded: Dict[str, bool] = {n: False for n in names}
    dest: Dict[str, str] = {n: "" for n in names}
    tid: Dict[str, str] = {n: "" for n in names}
    wait_pri: Dict[str, float] = {n: 0.0 for n in names}
    approach_station: Dict[str, str] = {}
    force_stay: Set[str] = set()
    escape_goal: Dict[str, Cell] = {}
    escape_ttl: Dict[str, int] = {}
    face_commit: Dict[str, Tuple[int, int]] = {}

    steps: Dict[str, List[dict]] = {n: [] for n in names}
    now = 0
    done = 0
    done_seen: Set[str] = set()
    net = _get_net(device)
    rule_sys: Optional[RealtimeRuleSystem] = None
    xfmr: Optional[MapdTransformer] = None
    # RRSC (road-rights → stage → priority greedy) is default for auto.
    # RuleNet is opt-in: default uses goal-flow teacher only (ablation: same
    # done/sim, lower wall). StageNet remains the only default learned module.
    use_rrsc = policy in ("rrsc", "auto")
    use_glocal = policy == "glocal"
    if policy in ("auto", "rrsc", "rules", "glocal"):
        want_rulesys = policy in ("rrsc", "rules", "glocal") or policy == "auto"
        if want_rulesys:
            if use_rule_net and not RULE_CKPT.exists():
                if policy in ("rrsc", "rules", "glocal"):
                    raise RuntimeError(
                        f"use_rule_net but checkpoint missing; "
                        "run: python -m greed_coop_mcts.train_rule_system"
                    )
                rule_sys = RealtimeRuleSystem(device=device, use_net=False)
            else:
                rule_sys = RealtimeRuleSystem(
                    device=device, use_net=bool(use_rule_net)
                )
    if policy == "rules":
        use_glocal = False
        use_rrsc = False
    if policy == "glocal":
        use_rrsc = False
    if policy == "rrsc":
        use_glocal = False
        # Deterministic StageNet / numpy paths for reproducible meetings.
        np.random.seed(0)
        try:
            import torch

            torch.manual_seed(0)
            torch.use_deterministic_algorithms(False)
        except Exception:  # noqa: BLE001
            pass
    if use_rrsc and rule_sys is None:
        use_rrsc = False
    if policy in ("auto", "xfmr") and rule_sys is None:
        xfmr = get_transformer(device)
    if policy == "xfmr" and xfmr is None:
        raise RuntimeError("policy=xfmr but Transformer checkpoint missing; train first")
    use_rules = (
        (not use_glocal)
        and (not use_rrsc)
        and rule_sys is not None
        and policy != "mcts"
        and policy != "xfmr"
    )
    use_xfmr = (
        (not use_rules)
        and (not use_glocal)
        and (not use_rrsc)
        and xfmr is not None
        and policy != "mcts"
    )
    n_mcts_calls = 0
    n_xfmr_calls = 0
    n_rule_ticks = 0
    n_glocal_ticks = 0
    n_rrsc_ticks = 0
    glocal_stats = CommStats()
    blocked_streak: Dict[str, int] = defaultdict(int)
    restage_cool: Dict[str, int] = defaultdict(int)
    last_yielders: Set[str] = set()
    traffic = learn_traffic_maps(meta) if use_traffic_rules else None
    n_temp_overrides = 0
    temp_counts: Dict[str, int] = defaultdict(int)
    sticky_winners: Dict[str, str] = {}
    # Capacity-anchored ceiling + Throat-Band + Adaptive Stage Authority.
    asa_stage_net = bool(use_stage_net)
    near_pad_deg2 = 0
    if use_rrsc or use_glocal or use_rules:
        from greed_coop_mcts.pgcp import (
            adaptive_use_stage_net,
            near_pad_deg2_count,
            throat_band_fleet_cap,
        )

        near_pad_deg2 = int(
            near_pad_deg2_count(blocked, set(dropoffs.values()))
        )
        max_fleet_busy = throat_band_fleet_cap(
            len(names), near_pad_deg2, base=6
        )
        if use_rrsc and use_stage_net and not force_stage_net:
            asa_stage_net = bool(
                adaptive_use_stage_net(len(names), near_pad_deg2)
            )
    else:
        max_fleet_busy = 3 if use_traffic_rules else 8
    path_commits: Dict[str, PathCommit] = {}
    rh_stats = RHStats()
    RH_HORIZON = 8
    stage_sel: Optional[StageSelector] = None
    soft_guide: Optional[SoftGuidance] = None
    meet_stats = MeetingStats()
    meet_contracts: Dict[str, StageContract] = {}
    meet_pri: Dict[str, float] = {}
    last_meet_t = -1
    claim_freeze_until = -1  # pulsed clear window after stall-drop
    last_stall_drop_t = -10_000
    # GREED_LIVELOCK_MODE: full | detect | plan | off
    #   full   = early AI detect + path-guided unblock (default)
    #   detect = early AI detect, but legacy evacuate
    #   plan   = stall=300 soft timing, but path-guided unblock
    #   off    = legacy soft unlock only (ablation baseline)
    livelock_mode = str(os.environ.get("GREED_LIVELOCK_MODE", "full")).strip().lower()
    if livelock_mode not in ("full", "detect", "plan", "off"):
        livelock_mode = "full"
    livelock_det = LivelockDetector(
        window=48, score_th=0.70, min_freeze_ticks=36, cooldown=90
    )
    print(f"[greed_coop] livelock_mode={livelock_mode}", flush=True)
    clear_cool: Dict[str, int] = defaultdict(int)
    yield_cool: Dict[str, int] = defaultdict(int)
    endgame_lead: Optional[str] = None
    endgame_lead_goal: Optional[Cell] = None
    altruism = AltruismState()
    progress = ProgressState()
    last_unload_pad: Dict[str, Cell] = {}
    base_fleet_busy = int(max_fleet_busy)
    if force_fleet_cap is not None:
        base_fleet_busy = int(force_fleet_cap)
        max_fleet_busy = int(force_fleet_cap)
    recent_commit_h = float(MEET_W)
    if use_rrsc:
        stage_sel = StageSelector(
            device=device,
            use_net=bool(asa_stage_net),
            use_cgpn=bool(use_cgpn),
        )
        soft_guide = None  # SoftGuidance stays opt-in; motion BFS excludes it
    region_free = count_free_by_region(blocked)
    region_cap = region_caps(region_free)
    sum_parallel_busy = 0
    n_parallel_samples = 0
    sum_region_occ = 0.0
    n_stage_admit_demote = 0
    # FACN: learned concurrency Δ on top of TBC (oracle override disables).
    # Adaptive FACN Authority: raise-eligible open fleets, or mild n≥16
    # fleets (near-pad corridors ≤ THROTTLE_NEAR_HI). Flood maps stay TBC.
    facn = None
    n_free_cells = len(free)
    if (
        force_fleet_cap is None
        and bool(use_facn)
        and (use_rrsc or use_glocal or use_rules)
    ):
        from greed_coop_mcts.facn_net import FacnController, afa_allow_attach

        facn_eligible = bool(
            afa_allow_attach(
                n_agents=len(names),
                free=n_free_cells,
                near_pad_deg2=int(near_pad_deg2),
                k_tbc=int(max_fleet_busy),
            )
        )
        if facn_eligible:
            facn = FacnController.maybe_load(
                device=device,
                enabled=True,
                k_tbc=int(max_fleet_busy),
                n_agents=len(names),
                free=n_free_cells,
                near_pad_deg2=int(near_pad_deg2),
                n_pickups=len(pickups),
                n_dropoffs=len(dropoffs),
                region_free=list(region_free.values()),
                slot=int(slot) if slot is not None else None,
            )
            base_fleet_busy = int(facn.current_k(remaining=int(total)))
    done_at_facn = 0
    mode = (
        "rrsc"
        if use_rrsc
        else (
            "glocal"
            if use_glocal
            else ("rules" if use_rules else ("xfmr" if use_xfmr else "mcts/greedy"))
        )
    )
    print(
        f"[greed_coop] policy={mode} traffic_rules={traffic is not None}"
        + (f" meet_h={MEET_H}" if use_rrsc else "")
        + f" rule_net={bool(use_rule_net and rule_sys is not None and getattr(rule_sys, 'use_net', False))}"
        + f" cgpn={bool(use_cgpn and stage_sel is not None and getattr(stage_sel, 'cgpn', None) is not None)}"
        + f" facn={bool(facn is not None)} k0={int(base_fleet_busy)}"
        + (
            f" d0={int(facn.delta)} neural={bool(facn.net is not None)}"
            if facn is not None
            else ""
        ),
        flush=True,
    )

    def _row(n: str, t: int) -> dict:
        x, y, p = pose[n]
        return {
            "timestamp": t,
            "name": n,
            "X": x,
            "Y": y,
            "pitch": int(p) % 360,
            "loaded": "TRUE" if loaded[n] else "FALSE",
            "destination": dest[n] if loaded[n] else "",
            "Emergency": "FALSE",
            "task-id": tid[n] if loaded[n] else "",
        }

    skip_traj = str(os.environ.get("GREED_SKIP_TRAJ", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    # Append-only viz CSV for pygame (contiguous timestamps — no rolling rewrite jumps).
    viz_on = str(os.environ.get("GREED_VIZ_TRAJ", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        viz_max_mb = max(8, int(os.environ.get("GREED_VIZ_MAX_MB", "80")))
    except ValueError:
        viz_max_mb = 80
    viz_path = TRAJ_DIR / f"{sid}_viz.csv"
    viz_fh = None
    viz_writer: Optional[csv.DictWriter] = None
    viz_last_t = -1
    viz_rows_since_check = 0

    def _close_viz() -> None:
        nonlocal viz_fh, viz_writer
        if viz_fh is not None:
            try:
                viz_fh.flush()
                viz_fh.close()
            except Exception:  # noqa: BLE001
                pass
        viz_fh = None
        viz_writer = None

    def _open_viz(*, fresh: bool) -> None:
        nonlocal viz_fh, viz_writer
        _close_viz()
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        if fresh or not viz_path.is_file():
            viz_fh = viz_path.open("w", newline="", encoding="utf-8")
            viz_writer = csv.DictWriter(viz_fh, fieldnames=TRAJ_HEADER)
            viz_writer.writeheader()
        else:
            viz_fh = viz_path.open("a", newline="", encoding="utf-8")
            viz_writer = csv.DictWriter(viz_fh, fieldnames=TRAJ_HEADER)

    def _flush_viz(force: bool = False) -> None:
        if viz_fh is not None:
            viz_fh.flush()

    def _maybe_compact_viz() -> None:
        """If viz grows huge, keep a trailing contiguous window (rare)."""
        nonlocal viz_rows_since_check, viz_last_t
        if not viz_on or not viz_path.is_file():
            return
        viz_rows_since_check += 1
        if viz_rows_since_check < 2000:
            return
        viz_rows_since_check = 0
        try:
            sz = viz_path.stat().st_size
        except OSError:
            return
        if sz < int(viz_max_mb) * 1024 * 1024:
            return
        _close_viz()
        try:
            with viz_path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                _open_viz(fresh=True)
                return
            # Keep last ~viz_max_mb/2 worth by time span (~12k ticks @ 8 AGVs ≈ 80MB/5)
            times = sorted({int(r["timestamp"]) for r in rows})
            keep_from = times[max(0, len(times) - 8000)]
            kept = [r for r in rows if int(r["timestamp"]) >= keep_from]
            with viz_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
                w.writeheader()
                for r in kept:
                    w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})
            viz_last_t = int(kept[-1]["timestamp"]) if kept else -1
            print(
                f"[greed_coop] viz compact keep_from={keep_from} "
                f"rows={len(kept)} path={viz_path.name}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[greed_coop] viz compact failed: {exc}", flush=True)
        _open_viz(fresh=False)

    def _write_tick(t: int) -> None:
        nonlocal viz_last_t
        block = [_row(n, t) for n in names]
        if skip_traj:
            # Keep only the latest pose row (enough for _last_*; no GB traj).
            for n, r in zip(names, block):
                steps[n] = [r]
        else:
            for n, r in zip(names, block):
                steps[n].append(r)
        if viz_on and viz_writer is not None and int(t) > int(viz_last_t):
            for r in block:
                viz_writer.writerow({k: r.get(k, "") for k in TRAJ_HEADER})
            viz_last_t = int(t)
            if int(t) % 10 == 0:
                _flush_viz()
            _maybe_compact_viz()

    def _last_cell(n: str) -> Optional[Cell]:
        seq = steps[n]
        if not seq:
            return None
        r = seq[-1]
        return (int(r["X"]), int(r["Y"]))

    def _last_pitch(n: str) -> int:
        seq = steps[n]
        if not seq:
            return int(pose[n][2]) % 360
        return int(seq[-1].get("pitch") or 0) % 360

    if viz_on:
        _open_viz(fresh=True)
    _write_tick(0)
    _flush_viz(force=True)

    def _claim() -> None:
        free_agvs = [
            n
            for n in names
            if n not in assigned and n not in carrying and not loaded[n]
        ]
        for st, agv in list(approach_station.items()):
            if agv not in assigned:
                approach_station.pop(st, None)
        occupied_stations = set(approach_station.keys())
        busy = len(carrying) + len(assigned)
        # Cap approachers so carriers keep right-of-way (state-based, not map-typed).
        # FACN sets base_fleet_busy; busy_pulse can still temporarily lower it.
        fleet_cap = effective_fleet_cap(
            base_fleet_busy, progress, n_agents=len(names)
        )
        # Flood maps: a stall pulse to cap=3 (or K<=6) deadlocks the labyrinth.
        # Keep concurrency at least 7 and do not admit extra approachers.
        flood = (
            facn is not None
            and int(getattr(facn, "near_pad_deg2", 0) or 0) > 80
        )
        if flood:
            fleet_cap = max(int(fleet_cap), min(7, len(names)))
        # Keep classic approacher cap unless FACN explicitly raised Δ.
        if flood:
            max_assigned = min(3, max(2, int(fleet_cap) - len(carrying)))
        elif facn is not None and int(getattr(facn, "delta", 0) or 0) > 0:
            max_assigned = min(
                max(3, int(fleet_cap) // 2),
                max(2, int(fleet_cap) - len(carrying)),
            )
        else:
            max_assigned = min(3, max(2, int(fleet_cap) - len(carrying)))
        # Pulsed freeze after stall-drop (not permanent — avoids starvation).
        if now < int(claim_freeze_until):
            return
        for st in sorted(queues.keys()):
            if not free_agvs:
                break
            if busy >= int(fleet_cap):
                break
            if len(assigned) >= int(max_assigned):
                break
            if not queues.get(st):
                continue
            if st in occupied_stations:
                continue
            pad = pickups.get(st)
            if not pad:
                continue
            door = _pickup_approach(pad)
            if door in blocked:
                continue
            task = queues[st][0]
            agv = min(free_agvs, key=lambda n: (_spath(pose[n][:2], door), n))
            assigned[agv] = dict(task)
            last_unload_pad.pop(agv, None)
            approach_station[st] = agv
            free_agvs.remove(agv)
            busy += 1

    def _goals_and_priority() -> Dict[str, Cell]:
        goals: Dict[str, Cell] = {}
        hot: Set[Cell] = set()
        for pad in pickups.values():
            hot.add(_pickup_approach(pad))
        for dname in dropoffs:
            hot.update(unload_pads(dname))

        for n in names:
            cell = pose[n][:2]
            if n in escape_goal and escape_ttl.get(n, 0) > 0:
                goals[n] = escape_goal[n]
                continue
            if n in carrying:
                task = carrying[n]
                pads = unload_pads(str(task.get("destination") or ""))
                # Prefer unload cells not occupied by idles (avoids pad camping).
                idle_occ = {
                    pose[a][:2]
                    for a in names
                    if a not in carrying and a not in assigned and a not in force_stay
                }
                goals[n] = (
                    min(
                        pads,
                        key=lambda c: (
                            1 if c in idle_occ else 0,
                            _spath(cell, c),
                            c,
                        ),
                    )
                    if pads
                    else cell
                )
            elif n in assigned:
                task = assigned[n]
                st = str(task.get("start_point") or "")
                pad = pickups.get(st)
                goals[n] = _pickup_approach(pad) if pad else cell
            else:
                park = home.get(n, cell)
                if cell in hot:
                    cands = [
                        c for c in _neighbors4(cell) if c in free and c not in hot
                    ]
                    goals[n] = (
                        min(cands, key=lambda c: (_manh(c, park), c))
                        if cands
                        else park
                    )
                else:
                    goals[n] = park
        # Congestion: rewrite idle goals off busy corridors (state-triggered).
        if use_rrsc and should_idle_corridor_clear(
            stall=stall,
            carrying=set(carrying),
            blocked_streak=blocked_streak,
            remaining=int(total) - int(done),
        ):
            busy_names = set(carrying) | set(assigned)
            if busy_names and rule_sys is not None:
                bpath = collect_busy_path(
                    names=names,
                    poses=pose,
                    goals=goals,
                    blocked=blocked,
                    busy=busy_names,
                    rule_sys=rule_sys,
                )
                occ = {pose[a][:2] for a in names}
                for n in names:
                    if n in busy_names or n in force_stay:
                        continue
                    cur = pose[n][:2]
                    # Do not park idle onto a cell within PAD_EVICT of any unload goal.
                    g = idle_clear_goal(
                        cur=cur,
                        home=home.get(n, cur),
                        busy_path=bpath,
                        blocked=blocked,
                        occupied=occ - {cur},
                    )
                    near_pad = any(
                        _manh(g, goals[c]) <= 3 for c in carrying if c in goals
                    )
                    goals[n] = home.get(n, cur) if near_pad and home.get(n, cur) not in bpath else g
        return goals

    def _assign_escape() -> None:
        occupied = {pose[n][:2] for n in names}
        free_cells = sorted(c for c in free if c not in occupied)
        if not free_cells:
            return
        for n in names:
            if n in force_stay:
                continue
            # Idle / blocked agents escape earlier; carriers only if long wait.
            if n in carrying and wait_pri.get(n, 0) < 80:
                continue
            if n in assigned and wait_pri.get(n, 0) < 40:
                continue
            cell = pose[n][:2]
            # Idle: leave hot zones; busy: short detour then resume goal.
            if n in carrying or n in assigned:
                near = sorted(
                    free_cells,
                    key=lambda c: (_spath(cell, c), hash((n, c, now))),
                )
                # pick a cell 3..8 steps away (not map-far parking exile)
                pick = near[min(len(near) - 1, 6)] if near else cell
                escape_goal[n] = pick
                escape_ttl[n] = 12
            else:
                far = max(
                    free_cells, key=lambda c: (_spath(cell, c), hash((n, c, now)))
                )
                escape_goal[n] = far
                escape_ttl[n] = 30
            face_commit.pop(n, None)

    def _try_pickup_rise() -> None:
        for agv, task in list(assigned.items()):
            st = str(task.get("start_point") or "")
            pad = pickups.get(st)
            if not pad:
                continue
            door = _pickup_approach(pad)
            cell = pose[agv][:2]
            if cell != door:
                continue
            last = _last_cell(agv)
            if last != door:
                continue
            if _last_pitch(agv) != int(pose[agv][2]) % 360:
                continue
            q = queues.get(st) or []
            if not q or str(q[0].get("task_id") or "") != str(task.get("task_id") or ""):
                continue
            q.pop(0)
            if not q:
                queues.pop(st, None)
            else:
                queues[st] = q
            tid_s = str(task.get("task_id") or "")
            pose[agv] = (door[0], door[1], _last_pitch(agv))
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            tid[agv] = tid_s
            carrying[agv] = dict(task)
            assigned.pop(agv, None)
            last_unload_pad.pop(agv, None)
            approach_station.pop(st, None)
            wait_pri[agv] = 0.0
            force_stay.add(agv)

    def _try_unload() -> None:
        nonlocal done
        for agv, task in list(carrying.items()):
            pads = set(unload_pads(str(task.get("destination") or "")))
            cell = pose[agv][:2]
            if cell not in pads:
                continue
            last = _last_cell(agv)
            if last != cell:
                continue
            if _last_pitch(agv) != int(pose[agv][2]) % 360:
                continue
            tid_s = str(task.get("task_id") or "")
            pose[agv] = (cell[0], cell[1], _last_pitch(agv))
            loaded[agv] = False
            dest[agv] = ""
            tid[agv] = ""
            carrying.pop(agv, None)
            last_unload_pad[agv] = cell  # pad cell just vacated → egress priority
            wait_pri[agv] = 0.0
            force_stay.add(agv)
            if tid_s and tid_s not in done_seen:
                done_seen.add(tid_s)
                done += 1

    def _apply_desired(desired: Dict[str, Cell]) -> bool:
        progressed = False
        new_pose = dict(pose)
        turned: Set[str] = set()

        for n in list(face_commit.keys()):
            pitch_c, ttl = face_commit[n]
            ttl -= 1
            if ttl <= 0:
                face_commit.pop(n, None)
            else:
                face_commit[n] = (pitch_c, ttl)

        for n in names:
            if n in force_stay:
                wait_pri[n] = float(wait_pri[n]) + 1.0
                continue
            cur = pose[n][:2]
            pitch = int(pose[n][2]) % 360
            want = desired.get(n, cur)
            if want == cur:
                wait_pri[n] = float(wait_pri[n]) + 1.0
                continue
            need = pitch_for_step(cur, want)
            if need < 0:
                wait_pri[n] = float(wait_pri[n]) + 1.0
                continue
            if pitch != need:
                # Stall unlock: drop stale face lock so agents can reorient.
                if n in face_commit and face_commit[n][0] != need:
                    if stall < 30 and face_commit[n][1] > 0:
                        wait_pri[n] = float(wait_pri[n]) + 1.0
                        continue
                    face_commit.pop(n, None)
                chain = turn_pitch_chain(pitch, need)
                if chain:
                    new_pose[n] = (cur[0], cur[1], int(chain[0]) % 360)
                    turned.add(n)
                    face_commit[n] = (need, 6 if stall >= 30 else 12)
                    progressed = True
                    wait_pri[n] = 0.0
                continue

        movers: Dict[str, Cell] = {}
        for n in names:
            if n in force_stay or n in turned:
                continue
            cur = pose[n][:2]
            pitch = int(new_pose[n][2]) % 360
            want = desired.get(n, cur)
            if want == cur:
                continue
            need = pitch_for_step(cur, want)
            if need >= 0 and pitch == need:
                movers[n] = want

        occ_owner: Dict[Cell, str] = {pose[n][:2]: n for n in names}
        final_move: Dict[str, Cell] = {}
        order = sorted(
            movers.keys(),
            key=lambda n: (
                -float(1e6 if n in carrying else 1e5 if n in assigned else 0)
                - wait_pri[n],
                n,
            ),
        )
        # Head-on (A→B, B→A) is INVALID under MAPD rules — yield lower priority.
        yield_stay: Set[str] = set()
        for n in order:
            if n in yield_stay:
                continue
            nxt = movers[n]
            cur = pose[n][:2]
            holder = occ_owner.get(nxt)
            if holder is None or holder == n or holder not in movers:
                continue
            if movers.get(holder) != cur:
                continue
            # Higher priority (earlier in order) proceeds; other yields (stay).
            yield_stay.add(holder)

        for n in order:
            if n in yield_stay:
                continue
            nxt = movers[n]
            cur = pose[n][:2]
            if nxt in blocked:
                continue
            holder = occ_owner.get(nxt)
            if holder is not None and holder not in final_move:
                continue
            # Block the illegal swap edge even if holder somehow queued.
            if holder is not None and final_move.get(holder) == cur:
                continue
            if occ_owner.get(cur) == n:
                del occ_owner[cur]
            occ_owner[nxt] = n
            final_move[n] = nxt

        for n, nxt in final_move.items():
            pitch = int(new_pose[n][2]) % 360
            new_pose[n] = (nxt[0], nxt[1], pitch)
            progressed = True
            wait_pri[n] = 0.0
            face_commit.pop(n, None)

        by_cell: Dict[Cell, List[str]] = {}
        for n in names:
            by_cell.setdefault(new_pose[n][:2], []).append(n)
        for cell, ags in by_cell.items():
            if len(ags) <= 1:
                continue
            natives = [n for n in ags if pose[n][:2] == cell]
            if natives:
                keep = sorted(
                    natives,
                    key=lambda n: (
                        -float(1e6 if n in carrying else 1e5 if n in assigned else 0)
                        - wait_pri[n],
                        n,
                    ),
                )[0]
            else:
                keep = sorted(
                    ags,
                    key=lambda n: (
                        -float(1e6 if n in carrying else 1e5 if n in assigned else 0)
                        - wait_pri[n],
                        n,
                    ),
                )[0]
            for n in ags:
                if n != keep:
                    new_pose[n] = pose[n]

        for n in names:
            pose[n] = new_pose[n]
        return progressed

    stall = 0
    last_done = 0
    while now < int(max_sim_time):
        if time.perf_counter() - t0 >= float(wall_timeout):
            break
        _maybe_refill_lifelong()
        idle = _queue_pending() == 0 and _inflight() == 0
        if idle:
            if not lifelong_random:
                if done >= total:
                    break
            else:
                _maybe_refill_lifelong()
                if _queue_pending() == 0 and _inflight() == 0:
                    # finished initial seed and cannot/should not stream more
                    hit_cap = (
                        lifelong_max_total is not None
                        and total >= int(lifelong_max_total)
                    )
                    if done >= initial_total and (
                        hit_cap or not station_names or not dest_names
                    ):
                        break
                    if done >= total and not stream_on and done < initial_total:
                        break

        force_stay.clear()
        for n in list(escape_ttl.keys()):
            escape_ttl[n] = int(escape_ttl[n]) - 1
            if escape_ttl[n] <= 0:
                escape_ttl.pop(n, None)
                escape_goal.pop(n, None)

        _claim()
        _try_pickup_rise()
        _try_unload()
        if (
            not lifelong_random
            and done >= total
            and _queue_pending() == 0
            and _inflight() == 0
        ):
            now += 1
            _write_tick(now)
            break

        for agv, task in list(assigned.items()):
            st = str(task.get("start_point") or "")
            pad = pickups.get(st)
            if not pad:
                continue
            door = _pickup_approach(pad)
            if pose[agv][:2] == door:
                force_stay.add(agv)
        for agv in list(carrying.keys()):
            pads = set(unload_pads(str(carrying[agv].get("destination") or "")))
            if pose[agv][:2] in pads:
                force_stay.add(agv)

        if stall > 0 and stall % 120 == 0:
            _assign_escape()

        def _release_approachers() -> int:
            """Drop approach claims; FIFO heads stay in queues (never popped)."""
            n_rel = 0
            for agv in list(assigned.keys()):
                task = assigned.pop(agv)
                st = str(task.get("start_point") or "")
                if approach_station.get(st) == agv:
                    approach_station.pop(st, None)
                wait_pri[agv] = 0.0
                face_commit.pop(agv, None)
                escape_goal.pop(agv, None)
                escape_ttl.pop(agv, None)
                n_rel += 1
            return n_rel

        def _evacuate_blockers(*, ttl: int = 48) -> int:
            """Push non-carriers off carrier columns / clusters (multi-step escape).

            Neighbor-only idle_clear fails when the corridor is fully packed —
            every adjacent cell is occupied, so agents stay put. Far escape
            goals are required to break SH03-style vertical-throat livelocks.
            """
            if not carrying:
                return 0
            carrier_cells = {pose[n][:2] for n in carrying}
            carrier_cols = {c[0] for c in carrier_cells}
            occupied = {pose[n][:2] for n in names}
            free_pool = [c for c in free if c not in occupied]
            if not free_pool:
                return 0
            n_esc = 0
            for n in names:
                if n in carrying or n in force_stay:
                    continue
                cell = pose[n][:2]
                on_col = cell[0] in carrier_cols
                near = min((_manh(cell, cc) for cc in carrier_cells), default=99)
                if not on_col and near > 2 and n not in assigned:
                    continue
                if not free_pool:
                    break
                dest_c = max(
                    free_pool,
                    key=lambda c: (
                        0 if c[0] in carrier_cols else 1,
                        min((_manh(c, cc) for cc in carrier_cells), default=0),
                        -_manh(c, cell),
                        hash((n, c, now)),
                    ),
                )
                free_pool = [c for c in free_pool if c != dest_c]
                escape_goal[n] = dest_c
                escape_ttl[n] = int(ttl)
                face_commit.pop(n, None)
                n_esc += 1
            return n_esc

        # --- AI livelock detect (pose-freeze, ~40 ticks) + path-guided unblock ---
        _ll_feat = livelock_det.observe(
            t=int(now),
            poses=pose,
            carrying=set(carrying),
            assigned=set(assigned),
            pending=int(_queue_pending()),
            stall=int(stall),
        )
        _ai_detect = (
            livelock_mode in ("full", "detect")
            and livelock_det.should_unlock(t=int(now), feat=_ll_feat)
        )

        # Soft corridor unlock (stall clock) — used by off/plan, or as detect fallback.
        _rem = int(total) - int(done)
        _soft_th = 300
        _soft_cd = 250
        _want_soft = (
            (not _ai_detect)
            and (use_rrsc or use_rules or lifelong_random)
            and bool(carrying)
            and int(stall) >= int(_soft_th)
            and (
                bool(assigned)
                or int(stall) >= 450
                or _rem <= 8
            )
        )
        _do_unlock = (_ai_detect or _want_soft) and (
            now - int(last_stall_drop_t)
        ) >= (60 if _ai_detect else int(_soft_cd))
        if _do_unlock:
            use_plan = livelock_mode in ("full", "plan") and (
                _ai_detect or livelock_mode == "plan"
            )
            tag = (
                "livelock-AI"
                if (_ai_detect and livelock_mode == "full")
                else (
                    "livelock-detect"
                    if (_ai_detect and livelock_mode == "detect")
                    else (
                        "livelock-plan"
                        if (use_plan and livelock_mode == "plan")
                        else "corridor-unlock soft"
                    )
                )
            )
            if use_plan:
                _pre_goals = _goals_and_priority()
                plan = plan_unblock(
                    names=names,
                    poses=pose,
                    goals=_pre_goals,
                    carrying=set(carrying),
                    assigned=set(assigned),
                    blocked=blocked,
                    free=free,
                    escape_ttl=64,
                )
                n_rel = 0
                for agv in list(plan.release):
                    if agv not in assigned:
                        continue
                    task = assigned.pop(agv)
                    st = str(task.get("start_point") or "")
                    if approach_station.get(st) == agv:
                        approach_station.pop(st, None)
                    wait_pri[agv] = 0.0
                    face_commit.pop(agv, None)
                    n_rel += 1
                if _ll_feat.carrier_blocked >= 0.5 and assigned:
                    n_rel += _release_approachers()
                n_esc = 0
                for agv, dest_c in plan.escape.items():
                    if agv in carrying or agv in force_stay:
                        continue
                    escape_goal[agv] = dest_c
                    escape_ttl[agv] = 64
                    face_commit.pop(agv, None)
                    n_esc += 1
                for agv in plan.boost_carriers:
                    wait_pri[agv] = float(wait_pri.get(agv, 0.0)) + 500.0
                n_crit = len(plan.critical_cells)
            else:
                n_rel = _release_approachers()
                n_esc = _evacuate_blockers(ttl=56)
                n_crit = 0
            face_commit.clear()
            sticky_winners.clear()
            last_stall_drop_t = int(now)
            claim_freeze_until = int(now) + (70 if use_plan else 80)
            if _ai_detect:
                livelock_det.mark_fired(int(now))
            if lifelong_random and (_ai_detect or int(stall) >= 400):
                stream_on = True
            print(
                f"[greed_coop] {tag} t={now} stall={stall} "
                f"score={_ll_feat.score:.2f} freeze={_ll_feat.freeze_frac:.2f} "
                f"cblock={_ll_feat.carrier_blocked:.2f} "
                f"done={done}/{total} released={n_rel} escape={n_esc} "
                f"crit={n_crit} freeze->{claim_freeze_until}",
                flush=True,
            )

        true_goals = _goals_and_priority()
        goals = dict(true_goals)
        # Pad-egress (every tick): motion goals + meet_pri; keep true unload goals.
        if use_rrsc:
            eg0 = int(progress.n_pad_egress)
            _pos = {n: pose[n][:2] for n in names}
            goals, meet_pri = apply_pad_egress(
                progress,
                stage_goals=goals,
                carrying=set(carrying),
                assigned=set(assigned),
                goals=true_goals,
                positions=_pos,
                blocked=blocked,
                force_stay=force_stay,
                rule_sys=rule_sys,
                priority=meet_pri
                or {
                    n: (
                        1e6
                        if n in carrying
                        else 1e5
                        if n in assigned
                        else 0.0
                    )
                    for n in names
                },
                last_unload_pad=last_unload_pad,
                near_pad_deg2=near_pad_deg2,
            )
            meet_stats.n_pad_egress += int(progress.n_pad_egress) - eg0
            e4_on = e4_sticky_enabled(len(names), near_pad_deg2)
            for n in list(last_unload_pad.keys()):
                if n in carrying or n in assigned:
                    last_unload_pad.pop(n, None)
                    continue
                # E4: keep pad memory while agent remains on a deg≤2 throat.
                if e4_on and n in _pos and cell_degree(_pos[n], blocked) <= 2:
                    continue
                if n not in progress.egress_agents:
                    last_unload_pad.pop(n, None)
            # Contested station gate every tick (not only on meeting) so
            # non-holders do not rush the throat between meetings.
            g0 = int(progress.n_dest_gates)
            goals = rewrite_contested_destinations(
                progress,
                stage_goals=goals,
                carrying=set(carrying),
                goals=true_goals,
                positions=_pos,
                blocked=blocked,
                rule_sys=rule_sys,
                blocked_streak=blocked_streak,
                home=home,
                near_pad_deg2=near_pad_deg2,
            )
            meet_stats.n_dest_gates += int(progress.n_dest_gates) - g0
            if progress.lock_holder and progress.lock_holder in meet_pri:
                meet_pri[progress.lock_holder] = (
                    float(meet_pri[progress.lock_holder]) + 3e5
                )
        # Pad serialization for MCTS/xfmr only. Under global rule AI, HOLD_PAD
        # parks carriers far from unload pads and causes permanent livelock.
        active_unloaders: Set[str] = set()
        if traffic is not None and not use_rules and not use_glocal and not use_rrsc:
            pads_of: Dict[str, List[Cell]] = {
                n: unload_pads(str(carrying[n].get("destination") or ""))
                for n in carrying
            }
            temps = assign_temp_tasks(
                traffic,
                names=names,
                positions={n: pose[n][:2] for n in names},
                primary_goals=goals,
                carrying=set(carrying),
                assigned=set(assigned),
                force_stay=force_stay,
                blocked=blocked,
                unload_pads_of=pads_of,
                stall=stall,
                wait_pri=wait_pri,
                stall_on=40,
                max_active_unloaders=2,
                sticky_winners=sticky_winners,
            )
            for n, (kind, tgt) in temps.items():
                if kind == TempTask.DELIVER:
                    if n in carrying and tgt is not None:
                        pads = pads_of.get(n) or []
                        goals[n] = (
                            min(pads, key=lambda c: (_manh(pose[n][:2], c), c))
                            if pads
                            else tgt
                        )
                        active_unloaders.add(n)
                    elif tgt is not None:
                        goals[n] = tgt
                    continue
                temp_counts[kind.value] += 1
                n_temp_overrides += 1
                if kind == TempTask.WAIT:
                    force_stay.add(n)
                elif kind in (TempTask.HOLD_PAD, TempTask.PARK, TempTask.CLEAR):
                    cell = pose[n][:2]
                    if tgt is not None and cell == tgt:
                        force_stay.add(n)
                    elif tgt is not None:
                        goals[n] = tgt
                    else:
                        force_stay.add(n)
                elif tgt is not None:
                    goals[n] = tgt
        elif use_rules or use_rrsc:
            active_unloaders = set(carrying)
        elif use_glocal:
            active_unloaders = set(carrying)

        active = [n for n in names if n not in force_stay]
        pos_now = {n: pose[n][:2] for n in names}
        # MRSC: periodic meetings (Pri+StageNet+window PP); between: follow+gap.
        if use_rrsc and rule_sys is not None and stage_sel is not None:
            n_rrsc_ticks += 1
            n_rule_ticks += 1
            deliver_goals = {n: goals[n] for n in carrying if n in goals}
            pickup_goals = {n: goals[n] for n in assigned if n in goals}
            rule_sys.update_if_needed(
                t=now,
                blocked=blocked,
                poses=pose,
                deliver_goals=deliver_goals,
                pickup_goals=pickup_goals,
                loaded=set(carrying),
                force=(now == 0) or (stall > 0 and stall % 120 == 0),
                stall=stall,
            )
            # Tick cooldowns once per sim step.
            for _cool in (clear_cool, yield_cool):
                for k in list(_cool.keys()):
                    if _cool[k] > 0:
                        _cool[k] -= 1
            remaining = int(total) - int(done)
            no_prog = update_progress(
                progress,
                done=int(done),
                carrying=set(carrying),
                poses=pose,
                goals=true_goals,
            )
            if facn is not None:
                # same-dest cluster size + agents on deg≤2 near drops
                from collections import Counter

                _pads = list(dropoffs.values())
                _gcount = Counter(
                    true_goals[n] for n in carrying if n in true_goals
                )
                same_cl = int(max(_gcount.values()) if _gcount else 0)
                throat = 0
                for _n, _p in pose.items():
                    _c = _p[:2]
                    if cell_degree(_c, blocked) > 2:
                        continue
                    if any(_manh(_c, dp) <= 6 for dp in _pads):
                        throat += 1
                # Rate-limit lives inside FacnController (MIN_GAP / period).
                # Do NOT gate on bare no_prog here — that caused per-tick thrash.
                _tot = max(1, int(total))
                _done_frac = float(done) / float(_tot)
                _phase_due = bool(
                    getattr(facn, "phased", False)
                    and type(facn)._phase_of(_done_frac)
                    > int(getattr(facn, "phase", 0))
                )
                due = (
                    facn.ttl <= 1
                    or (int(now) - int(facn.last_update_t)) >= int(facn.period)
                    or _phase_due
                    or (
                        bool(no_prog)
                        and (throat >= 4 or same_cl >= 4)
                        and (int(now) - int(facn.last_update_t))
                        >= int(getattr(facn, "min_gap", 12))
                    )
                )
                if due:
                    dr = float(done - done_at_facn) / float(
                        max(1, int(now) - int(facn.last_update_t))
                    )
                    base_fleet_busy = facn.maybe_update(
                        int(now),
                        busy=len(carrying) + len(assigned),
                        idle=sum(
                            1
                            for n in names
                            if n not in carrying and n not in assigned
                        ),
                        remaining=int(total) - int(done),
                        done_rate=dr,
                        stall=int(stall),
                        avg_h=float(recent_commit_h),
                        n_repair_win=int(
                            getattr(meet_stats, "n_window_repairs", 0)
                        ),
                        conflict_heat=0.0,
                        region_occ=float(sum_region_occ)
                        / float(max(1, n_parallel_samples)),
                        n_dest_gates=int(progress.n_dest_gates),
                        same_dest_cluster=same_cl,
                        throat_occ=throat,
                        no_prog=bool(no_prog),
                        total=int(total),
                        agv_poses={n: pose[n] for n in names if n in pose},
                    )
                    done_at_facn = int(done)
                elif facn.ttl > 0:
                    facn.ttl -= 1
                    base_fleet_busy = int(
                        facn.current_k(remaining=int(total) - int(done))
                    )
            dense = len(names) >= 12
            if no_prog:
                trigger_progress_response(progress, n_agents=len(names))
                if dense or int(stall) >= 40:
                    meet_stats.n_prog_widen += 1
            do_meet = should_meet(
                t=now, stall=stall, last_meet_t=last_meet_t
            )
            # Dense: replan often under no-progress. Sparse: avoid meet thrash (SH17).
            if no_prog and dense and (now - last_meet_t) >= 4:
                do_meet = True
            elif no_prog and (not dense) and stall >= 40 and (now - last_meet_t) >= 8:
                do_meet = True
            cur_meet_w = adaptive_meet_w(
                stall=stall,
                avg_committed=recent_commit_h,
                no_task_progress=bool(no_prog and (dense or stall >= 40)),
            )
            if cur_meet_w > MEET_W:
                meet_stats.n_adapt_widen += 1
            if do_meet:
                meet_contracts, path_commits, meet_pri, desired = run_meeting(
                    t=now,
                    names=names,
                    poses=pose,
                    goals=true_goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    carrying=set(carrying),
                    assigned=set(assigned),
                    blocked_streak=blocked_streak,
                    wait_pri=wait_pri,
                    rule_sys=rule_sys,
                    stage_sel=stage_sel,
                    region_cap=region_cap,
                    existing_commits=path_commits,
                    meet_w=cur_meet_w,
                    stall=int(stall),
                    remaining=remaining,
                    stats=meet_stats,
                    progress=progress,
                    home=home,
                    last_unload_pad=last_unload_pad,
                    near_pad_deg2=near_pad_deg2,
                )
                last_meet_t = int(now)
            else:
                desired = follow_contract_step(
                    names=names,
                    poses=pose,
                    contracts=meet_contracts,
                    commits=path_commits,
                    blocked=blocked,
                    force_stay=force_stay,
                    priority=meet_pri,
                    rule_sys=rule_sys,
                )
                _bene = altruism.beneficiary
                desired = local_gap_yield(
                    names=names,
                    poses=pose,
                    desired=desired,
                    priority=meet_pri,
                    goals=goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    carrying=set(carrying),
                    stats=meet_stats,
                    yield_cool=yield_cool,
                    beneficiary=_bene,
                    yield_gate=(
                        (
                            lambda mover, blocker, _b=_bene, _c=set(carrying), _p=meet_pri: focal_gap_ok(
                                mover=mover,
                                blocker=blocker,
                                beneficiary=_b,
                                carrying=_c,
                                priority=_p,
                            )
                        )
                        if should_pgfa(
                            stall=stall,
                            remaining=remaining,
                            carrying=set(carrying),
                            blocked_streak=blocked_streak,
                            n_switches=int(altruism.n_switches),
                        )
                        else None
                    ),
                )
            _pri_fallback = meet_pri or {
                n: (
                    1e6
                    if n in carrying
                    else 1e5
                    if n in assigned
                    else 0.0
                )
                for n in names
            }
            # PGFA: single altruism channel (focal beneficiary + progress gate).
            if should_pgfa(
                stall=stall,
                remaining=remaining,
                carrying=set(carrying),
                blocked_streak=blocked_streak,
                n_switches=int(altruism.n_switches),
            ):
                desired = apply_focal_altruism(
                    state=altruism,
                    names=names,
                    poses=pose,
                    desired=desired,
                    goals=goals,
                    home=home,
                    blocked=blocked,
                    force_stay=force_stay,
                    carrying=set(carrying),
                    assigned=set(assigned),
                    priority=_pri_fallback,
                    blocked_streak=blocked_streak,
                    rule_sys=rule_sys,
                    stall=int(stall),
                    remaining=int(remaining),
                    stats=meet_stats,
                    progress=progress,
                )
                endgame_lead = altruism.beneficiary
                endgame_lead_goal = altruism.beneficiary_goal
            elif should_idle_corridor_clear(
                stall=stall,
                carrying=set(carrying),
                blocked_streak=blocked_streak,
                remaining=remaining,
            ):
                # Light mid-run clear only when PGFA not active.
                desired = apply_idle_corridor_clear(
                    names=names,
                    poses=pose,
                    desired=desired,
                    goals=goals,
                    home=home,
                    blocked=blocked,
                    force_stay=force_stay,
                    carrying=set(carrying),
                    assigned=set(assigned),
                    priority=_pri_fallback,
                    rule_sys=rule_sys,
                    stats=meet_stats,
                    clear_cool=clear_cool,
                    remaining=remaining,
                )
            else:
                endgame_lead = None
                endgame_lead_goal = None

            # Idle hygiene: don't camp on deg≤2 carrier corridors.
            desired = apply_idle_hygiene(
                progress,
                names=names,
                poses=pose,
                desired=desired,
                goals=goals,
                blocked=blocked,
                force_stay=force_stay,
                carrying=set(carrying),
                assigned=set(assigned),
                rule_sys=rule_sys,
            )

            region_occ = occupancy_from_goals(
                {
                    n: goals[n]
                    for n in names
                    if n in goals and (n in carrying or n in assigned)
                }
            )
            busy_now = len(carrying) + len(assigned)
            sum_parallel_busy += busy_now
            n_parallel_samples += 1
            if region_occ:
                sum_region_occ += float(sum(region_occ.values())) / float(
                    max(1, len(region_occ))
                )
            tick_h_sum = 0
            tick_h_n = 0
            for n in names:
                hlen = len(getattr(path_commits.get(n), "cells", []) or [])
                glocal_stats.n_local_plans += 1
                glocal_stats.sum_path_horizon += max(1, hlen)
                rh_stats.sum_committed_horizon += hlen
                rh_stats.n_horizon_samples += 1
                tick_h_sum += hlen
                tick_h_n += 1
            if tick_h_n > 0:
                # EMA of fleet commit length for adaptive MEET_W.
                recent_commit_h = 0.85 * recent_commit_h + 0.15 * (
                    float(tick_h_sum) / float(tick_h_n)
                )
            last_yielders = set()
            for n in names:
                want = desired.get(n, pose[n][:2])
                if want == pose[n][:2] and n not in force_stay:
                    blocked_streak[n] = int(blocked_streak[n]) + 1
                else:
                    blocked_streak[n] = 0
        # GLocal-RH: stage contracts + committed H-paths + window repair.
        elif use_glocal and rule_sys is not None:
            n_glocal_ticks += 1
            n_rule_ticks += 1
            deliver_goals = {n: goals[n] for n in carrying if n in goals}
            pickup_goals = {n: goals[n] for n in assigned if n in goals}
            rule_sys.update_if_needed(
                t=now,
                blocked=blocked,
                poses=pose,
                deliver_goals=deliver_goals,
                pickup_goals=pickup_goals,
                loaded=set(carrying),
                force=(now == 0) or (stall > 0 and stall % 120 == 0),
                stall=stall,
            )
            pri = {
                n: (
                    1e6
                    if n in carrying
                    else 1e5
                    if n in assigned
                    else float(wait_pri.get(n, 0.0))
                )
                for n in names
            }
            # Traffic shaping while flowing.
            pads_of: Dict[str, List[Cell]] = {
                n: unload_pads(str(carrying[n].get("destination") or ""))
                for n in carrying
            }
            if stall < 25:
                goals, n_stg = staging_goals(
                    carrying=carrying,
                    positions=pos_now,
                    unload_pads_of=pads_of,
                    blocked=blocked,
                    primary_goals=goals,
                    max_per_dest=2,
                )
                rh_stats.n_staging += int(n_stg)

            for n in names:
                if restage_cool[n] > 0:
                    restage_cool[n] -= 1

            # --- Base motion = proven rules path (completion floor) ---
            if stall >= 25:
                desired = {}
                reserved = {pose[n][:2] for n in force_stay}
                order = sorted(
                    names,
                    key=lambda n: (
                        0 if n in carrying else 1 if n in assigned else 2,
                        -float(wait_pri.get(n, 0.0)),
                        n,
                    ),
                )
                for n in order:
                    if n in force_stay:
                        desired[n] = pose[n][:2]
                        continue
                    cur = pose[n][:2]
                    goal = goals.get(n, cur)
                    if n in carrying or n in assigned:
                        others = {
                            pose[m][:2]
                            for m in names
                            if m != n and m not in desired
                        }
                        leave = {
                            pose[m][:2]
                            for m, d in desired.items()
                            if d != pose[m][:2]
                        }
                        nxt = _bfs_next(
                            cur,
                            goal,
                            reserved=reserved | (others - leave),
                            pitch=int(pose[n][2]),
                        )
                    else:
                        nxt = rule_sys.directed_bfs_next(
                            cur,
                            goal,
                            blocked | reserved,
                            start_pitch=int(pose[n][2]),
                        )
                    if nxt in reserved and nxt != cur:
                        nxt = cur
                    desired[n] = nxt
                    reserved.add(nxt)
                path_commits = {}
                last_yielders = set()
            else:
                desired = rule_sys.desired_cells(
                    names=names,
                    poses=pose,
                    goals=goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    priority=pri,
                )
                # RH overlay: commit H-paths; adopt next step only if it does
                # not worsen manh and does not collide with higher-pri next.
                restage_req = {
                    n
                    for n, st in blocked_streak.items()
                    if st >= 20 and n not in force_stay and restage_cool[n] <= 0
                }
                for n in names:
                    pc = path_commits.get(n)
                    if n in force_stay or restage_cool[n] > 0:
                        continue
                    if pc is None or goals.get(n) != getattr(pc, "goal", None):
                        restage_req.add(n)
                for n in restage_req:
                    restage_cool[n] = 16
                if restage_req:
                    rh_stats.n_restage += len(restage_req)
                    glocal_stats.n_restage += len(restage_req)

                rule_sys.issue_contracts(
                    t=now,
                    names=names,
                    poses=pose,
                    goals=goals,
                    blocked=blocked,
                    loaded=set(carrying),
                    assigned=set(assigned),
                    priority=pri,
                    force=(now == 0) or (stall > 0 and stall % 16 == 0),
                    restage=restage_req,
                    yield_cells={},
                )
                path_commits = plan_committed(
                    names=names,
                    positions=pos_now,
                    goals=goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    priority=pri,
                    existing=path_commits,  # type: ignore[arg-type]
                    horizon=RH_HORIZON,
                    replan=restage_req,
                )
                path_commits, n_rep = window_repair(
                    path_commits,  # type: ignore[arg-type]
                    positions=pos_now,
                    goals=goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    window=RH_HORIZON,
                )
                rh_stats.n_window_repairs += int(n_rep)
                rh_stats.n_commits += 1
                claimed = {
                    desired[n]
                    for n in names
                    if n not in force_stay and desired.get(n) != pos_now[n]
                }
                order = sorted(
                    names, key=lambda n: (-float(pri.get(n, 0.0)), n)
                )
                for n in order:
                    pc = path_commits.get(n)
                    if not pc or not getattr(pc, "cells", None):
                        continue
                    if n in force_stay:
                        continue
                    rh_nxt = pc.cells[0]
                    cur = pos_now[n]
                    g = goals.get(n, cur)
                    base = desired.get(n, cur)
                    base_m = abs(base[0] - g[0]) + abs(base[1] - g[1])
                    rh_m = abs(rh_nxt[0] - g[0]) + abs(rh_nxt[1] - g[1])
                    if rh_m > base_m:
                        continue
                    if rh_nxt in claimed and rh_nxt != base:
                        continue
                    # Accept RH step
                    if base in claimed and base != cur:
                        claimed.discard(base)
                    desired[n] = rh_nxt
                    if rh_nxt != cur:
                        claimed.add(rh_nxt)
                for n, pc in path_commits.items():
                    hlen = len(getattr(pc, "cells", []) or [])
                    rh_stats.sum_committed_horizon += hlen
                    rh_stats.n_horizon_samples += 1
                    glocal_stats.n_local_plans += 1
                    glocal_stats.sum_path_horizon += hlen
                sketches = sketches_from_commits(path_commits)  # type: ignore[arg-type]
                _d, residuals, alert = arbitrate_intents(
                    sketches, positions=pos_now, force_stay=force_stay
                )
                glocal_stats.n_residual += len(residuals)
                if alert.cells:
                    glocal_stats.n_conflict_alerts += 1
                last_yielders = set(alert.yielders)

            for n in names:
                want = desired.get(n, pose[n][:2])
                if want == pose[n][:2] and n not in force_stay:
                    blocked_streak[n] = int(blocked_streak[n]) + 1
                else:
                    blocked_streak[n] = 0
            # Match rules fleet pressure on SH03-like density.
            # (max_fleet_busy already set; staging softens pad contention.)
        # Global rule AI: whole map → per-cell one-way; replan on predicted conflict.
        elif use_rules and rule_sys is not None:
            n_rule_ticks += 1
            deliver_goals = {
                n: goals[n] for n in carrying if n in goals
            }
            pickup_goals = {
                n: goals[n] for n in assigned if n in goals
            }
            rule_sys.update_if_needed(
                t=now,
                blocked=blocked,
                poses=pose,
                deliver_goals=deliver_goals,
                pickup_goals=pickup_goals,
                loaded=set(carrying),
                force=(now == 0) or (stall > 0 and stall % 120 == 0),
                stall=stall,
            )
            pri = {
                n: (
                    1e6
                    if n in carrying
                    else 1e5
                    if n in assigned
                    else float(wait_pri.get(n, 0.0))
                )
                for n in names
            }
            # Under stall: carriers/assignees use plain BFS (completion-critical);
            # rule field only guides idle AGVs away from choke points.
            if stall >= 25:
                desired = {}
                reserved = {pose[n][:2] for n in force_stay}
                order = sorted(
                    names,
                    key=lambda n: (
                        0 if n in carrying else 1 if n in assigned else 2,
                        -float(wait_pri.get(n, 0.0)),
                        n,
                    ),
                )
                for n in order:
                    if n in force_stay:
                        desired[n] = pose[n][:2]
                        continue
                    cur = pose[n][:2]
                    goal = goals.get(n, cur)
                    if n in carrying or n in assigned:
                        others = {
                            pose[m][:2]
                            for m in names
                            if m != n and m not in desired
                        }
                        # Allow stepping onto a body if that agent already chose to leave.
                        leave = {
                            pose[m][:2]
                            for m, d in desired.items()
                            if d != pose[m][:2]
                        }
                        nxt = _bfs_next(
                            cur,
                            goal,
                            reserved=reserved | (others - leave),
                            pitch=int(pose[n][2]),
                        )
                    else:
                        nxt = rule_sys.directed_bfs_next(
                            cur,
                            goal,
                            blocked | reserved,
                            start_pitch=int(pose[n][2]),
                        )
                    if nxt in reserved and nxt != cur:
                        nxt = cur
                    desired[n] = nxt
                    reserved.add(nxt)
            else:
                desired = rule_sys.desired_cells(
                    names=names,
                    poses=pose,
                    goals=goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    priority=pri,
                )
        # Prefer trained Transformer joint policy (SCRIMP-style). Fallback: MCTS/BFS.
        elif use_xfmr and xfmr is not None and active:
            n_xfmr_calls += 1
            order = sorted(
                names,
                key=lambda n: (
                    0 if n in active_unloaders else 1 if n in carrying else 2 if n in assigned else 3,
                    n,
                ),
            )
            desired = xfmr_plan(
                names=order,
                pose=pose,
                goals=goals,
                blocked=blocked,
                force_stay=force_stay,
                net=xfmr,
                device=device,
            )
        else:
            crowded = _need_mcts(pos_now, active)
            stride = 1 if stall < 40 else 3
            solo_unload = len(active_unloaders) <= 2 and len(carrying) <= 3
            run_mcts = crowded and (now % stride == 0) and not solo_unload
            gw, cw, yw = _adaptive_greed_coop(stall, greed_w, coop_w, yield_w)
            if run_mcts:
                n_mcts_calls += 1
                desired = _mats_desired(
                    names=names,
                    pose=pose,
                    goals=goals,
                    blocked=blocked,
                    force_stay=force_stay,
                    net=net,
                    num_simulations=num_simulations,
                    k_proximal=k_proximal,
                    device=device,
                    seed=now,
                    greed_w=gw,
                    coop_w=cw,
                    yield_w=yw,
                    force_mcts=True,
                )
            else:
                desired = {}
                reserved = {pose[n][:2] for n in force_stay}
                order = sorted(
                    names,
                    key=lambda n: (
                        0 if n in active_unloaders else 1 if n in carrying else 2 if n in assigned else 3,
                        n,
                    ),
                )
                for n in order:
                    if n in force_stay:
                        desired[n] = pose[n][:2]
                        continue
                    cur = pose[n][:2]
                    goal = goals.get(n, cur)
                    others = [pose[m][:2] for m in names if m != n]
                    if n in carrying or n in assigned or n in active_unloaders:
                        nxt = _bfs_next(
                            cur,
                            goal,
                            reserved=reserved | set(others),
                            pitch=int(pose[n][2]),
                        )
                        desired[n] = nxt
                        if nxt != cur:
                            reserved.add(nxt)
                        continue
                    ga = greedy_action(cur, goal, others, blocked)
                    nxt = apply_action(cur, ga, blocked)
                    if nxt in reserved:
                        nxt = cur
                    desired[n] = nxt
                    if nxt != cur:
                        reserved.add(nxt)
        for n in force_stay:
            desired[n] = pose[n][:2]

        _apply_desired(desired)
        if (use_glocal or use_rrsc) and path_commits:
            pos_after = {n: pose[n][:2] for n in names}
            advance_commits_after_move(path_commits, pos_after, desired)
            for n, pc in list(path_commits.items()):
                cells = getattr(pc, "cells", None)
                if not cells:
                    continue
                if pose[n][:2] == cells[0]:
                    cells.pop(0)
        now += 1
        _write_tick(now)

        if done != last_done:
            last_done = done
            stall = 0
            escape_goal.clear()
            escape_ttl.clear()
            face_commit.clear()
            sticky_winners.clear()
            livelock_det.reset()
        else:
            stall += 1
        # Hard unlock: lifelong recovers early; one-shot still breaks very late.
        _hard_th = 1200 if lifelong_random else 80000
        if stall >= int(_hard_th):
            if lifelong_random:
                n_rel = _release_approachers()
                n_esc = _evacuate_blockers(ttl=72)
                face_commit.clear()
                sticky_winners.clear()
                stream_on = True
                claim_freeze_until = int(now) + 120
                last_stall_drop_t = int(now)
                stall = 0
                _maybe_refill_lifelong()
                print(
                    f"[greed_coop] lifelong stall-recover t={now} "
                    f"done={done}/{total} released={n_rel} evacuate={n_esc} "
                    f"pending={_queue_pending()} streamed={n_streamed}",
                    flush=True,
                )
            else:
                print(
                    f"[greed_coop] stall break t={now} done={done}/{total}",
                    flush=True,
                )
                break

        if progress_every > 0 and (
            now % int(progress_every) == 0 or done == total
        ):
            extra = ""
            if use_rrsc:
                extra = (
                    f" rrsc={n_rrsc_ticks}"
                    f" meet={meet_stats.n_meetings}"
                    f" gap={meet_stats.n_gap_yields}"
                    f" clr={meet_stats.n_corridor_clears}"
                    f" repair={meet_stats.n_window_repairs}"
                    f" stgF={meet_stats.n_stage_final}"
                    f" stgC={meet_stats.n_stage_clear}"
                )
            elif use_glocal:
                extra = (
                    f" glocal={n_glocal_ticks}"
                    f" restage={rh_stats.n_restage}"
                    f" repair={rh_stats.n_window_repairs}"
                    f" stage={rh_stats.n_staging}"
                    f" H={rh_stats.avg_committed_horizon():.1f}"
                )
            print(
                f"  [{sid}/gcm] t={now} done={done}/{total} "
                f"carry={len(carrying)} approach={len(assigned)} "
                f"mcts={n_mcts_calls} xfmr={n_xfmr_calls} rules={n_rule_ticks} "
                f"replan={getattr(rule_sys, 'n_replans', 0)} "
                f"temp={n_temp_overrides}{extra} "
                f"wall={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            # Lightweight live snapshot for dashboard (no trajectory CSV).
            try:
                live_path = OUT / f"{sid}_live.json"
                agv_live = []
                for n in names:
                    x, y, p = pose[n]
                    agv_live.append(
                        {
                            "name": n,
                            "x": int(x),
                            "y": int(y),
                            "pitch": int(p) % 360,
                            "loaded": bool(loaded[n]),
                            "dest": dest[n] if loaded[n] else "",
                            "task_id": tid[n] if loaded[n] else "",
                            "approaching": n in assigned,
                        }
                    )
                live = {
                    "scenario_id": sid,
                    "slot": int(slot),
                    "t": int(now),
                    "done": int(done),
                    "total": int(total),
                    "initial_total": int(initial_total),
                    "stream_on": bool(stream_on),
                    "n_streamed": int(n_streamed),
                    "carry": int(len(carrying)),
                    "approach": int(len(assigned)),
                    "pending": int(_queue_pending()),
                    "k": int(base_fleet_busy),
                    "facn_delta": int(facn.delta) if facn is not None else None,
                    "facn_k_tbc": int(facn.k_tbc) if facn is not None else None,
                    "stall": int(stall),
                    "wall_s": round(time.perf_counter() - t0, 1),
                    "lifelong_random": bool(lifelong_random),
                    "lifelong_max_total": (
                        None
                        if lifelong_max_total is None
                        else int(lifelong_max_total)
                    ),
                    "agvs": agv_live,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                live_path.write_text(
                    json.dumps(live, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:  # noqa: BLE001
                pass

    _try_unload()
    _try_pickup_rise()

    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    _flush_viz(force=True)
    _close_viz()
    traj = TRAJ / f"{sid}_gcm.csv"
    rows: List[dict] = []
    for n in names:
        rows.extend(steps[n])
    def _ts(r: dict) -> int:
        try:
            return int(r.get("timestamp") or 0)
        except (TypeError, ValueError):
            return 0

    rows.sort(key=lambda r: (_ts(r), str(r.get("name") or "")))
    if not skip_traj:
        with traj.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})
        val = validate_trajectory(meta, traj)
    else:
        val = {"ok": True, "issues": {}, "skipped_traj": True}
    iss = val.get("issues") or {}
    wall = round(time.perf_counter() - t0, 2)
    rep = {
        "scenario_id": sid,
        "method": "greed_coop_mcts_v1",
        "slot": int(slot),
        "tasks_total": total,
        "tasks_completed": done,
        "completion_ratio": round(done / max(1, total), 4),
        "sim_time": now,
        "wall_seconds": wall,
        "trajectory": str(traj),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "n_collisions": int(iss.get("n_collisions") or 0),
        "n_swaps": int(iss.get("n_swaps") or 0),
        "n_illegal_motion": int(iss.get("n_illegal_motion") or 0),
        "n_fifo": int(iss.get("n_fifo_violations") or 0),
        "n_display_mismatch": int(iss.get("n_display_mismatch") or 0),
        "n_mcts_calls": int(n_mcts_calls),
        "n_xfmr_calls": int(n_xfmr_calls),
        "n_rule_ticks": int(n_rule_ticks),
        "n_glocal_ticks": int(n_glocal_ticks),
        "n_rrsc_ticks": int(n_rrsc_ticks),
        "n_rule_replans": int(getattr(rule_sys, "n_replans", 0) or 0),
        "n_conflict_triggers": int(
            getattr(rule_sys, "n_conflict_triggers", 0) or 0
        ),
        "n_yield_stages": int(getattr(rule_sys, "n_yield_stages", 0) or 0),
        "n_restage": int(
            rh_stats.n_restage
            if (use_glocal or use_rrsc)
            else glocal_stats.n_restage
        ),
        "n_residual": int(glocal_stats.n_residual),
        "n_conflict_alerts": int(glocal_stats.n_conflict_alerts),
        "avg_path_horizon": round(
            rh_stats.avg_committed_horizon()
            if use_glocal
            else glocal_stats.avg_path_horizon(),
            3,
        ),
        "n_window_repairs": int(
            meet_stats.n_window_repairs if use_rrsc else rh_stats.n_window_repairs
        ),
        "n_staging": int(rh_stats.n_staging),
        "avg_committed_horizon": round(rh_stats.avg_committed_horizon(), 3),
        "n_meetings": int(meet_stats.n_meetings),
        "n_gap_yields": int(meet_stats.n_gap_yields),
        "n_corridor_clears": int(meet_stats.n_corridor_clears),
        "n_same_drop_serial": int(meet_stats.n_same_drop_serial),
        "n_adapt_widen": int(meet_stats.n_adapt_widen),
        "n_prog_widen": int(meet_stats.n_prog_widen),
        "n_dest_gates": int(meet_stats.n_dest_gates),
        "n_outward": int(meet_stats.n_outward + progress.n_outward),
        "n_busy_pulse": int(progress.n_busy_pulse),
        "n_idle_hygiene": int(progress.n_idle_hygiene),
        "n_pad_egress": int(progress.n_pad_egress),
        "n_lane_locks": int(getattr(progress, "n_lane_locks", 0) or 0),
        "n_e4_sticky": int(getattr(progress, "n_e4_sticky", 0) or 0),
        "n_focal_altruism": int(altruism.n_focal_applies),
        "n_park_freezes": int(altruism.n_park_freezes),
        "n_bene_switches": int(altruism.n_switches),
        "n_temp_overrides": int(n_temp_overrides),
        "temp_task_counts": dict(temp_counts),
        "traffic_rules": bool(traffic is not None),
        "traffic_map_id": getattr(traffic, "map_id", "") if traffic else "",
        "planner": (
            "mrsc"
            if use_rrsc
            else (
                "glocal_rh"
                if use_glocal
                else (
                    "global_rule_ai"
                    if use_rules
                    else ("transformer_comm" if use_xfmr else "greed_coop_mcts")
                )
            )
        ),
        "policy": (
            "rrsc"
            if use_rrsc
            else (
                "glocal"
                if use_glocal
                else ("rules" if use_rules else ("xfmr" if use_xfmr else "mcts"))
            )
        ),
        "meet_h": int(MEET_H) if use_rrsc else 0,
        "greed_w": float(greed_w),
        "coop_w": float(coop_w),
        "yield_w": float(yield_w),
        "net": bool(net is not None),
        "transformer": bool(use_xfmr),
        "rule_ai": bool(use_rules or use_glocal or use_rrsc),
        "use_rule_net": bool(
            use_rule_net
            and rule_sys is not None
            and bool(getattr(rule_sys, "use_net", False))
        ),
        "glocal": bool(use_glocal),
        "rrsc": bool(use_rrsc),
        "stage_net": bool(stage_sel is not None),
        "asa_stage_net": bool(asa_stage_net) if use_rrsc else False,
        "near_pad_deg2": int(near_pad_deg2),
        "n_agents": int(len(names)),
        "n_stage_final": int(
            meet_stats.n_stage_final
            if use_rrsc
            else (getattr(stage_sel, "n_final", 0) or 0)
        ),
        "n_stage_clear": int(
            meet_stats.n_stage_clear
            if use_rrsc
            else (getattr(stage_sel, "n_clear", 0) or 0)
        ),
        "n_stage_approach": int(
            meet_stats.n_stage_approach
            if use_rrsc
            else (getattr(stage_sel, "n_approach", 0) or 0)
        ),
        "n_stage_admit_demote": int(n_stage_admit_demote),
        "avg_parallel_busy": round(
            float(sum_parallel_busy) / float(max(1, n_parallel_samples)), 3
        ),
        "avg_region_occ": round(
            float(sum_region_occ) / float(max(1, n_parallel_samples)), 3
        ),
        "max_fleet_busy": int(max_fleet_busy),
        "base_fleet_busy": int(base_fleet_busy),
        "use_facn": bool(facn is not None),
        "facn_neural": bool(facn is not None and facn.net is not None),
        "facn_delta": int(getattr(facn, "delta", 0) or 0),
        "n_facn_updates": int(getattr(facn, "n_updates", 0) or 0),
        "n_facn_phase_updates": int(getattr(facn, "n_phase_updates", 0) or 0),
        "n_facn_neural": int(getattr(facn, "n_neural", 0) or 0),
        "n_facn_recoveries": int(getattr(facn, "n_recoveries", 0) or 0),
        "facn_phased": bool(getattr(facn, "phased", False)),
        "lifelong_random": bool(lifelong_random),
        "initial_tasks": int(initial_total),
        "n_streamed_tasks": int(n_streamed),
        "tasks_offered": int(total),
        "use_cgpn": bool(
            use_cgpn
            and stage_sel is not None
            and getattr(stage_sel, "cgpn", None) is not None
        ),
        "n_cgpn_priority": int(getattr(stage_sel, "n_cgpn_priority", 0) or 0),
    }
    (OUT / f"{sid}_gcm.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[greed_coop] done={done}/{total} sim={now} wall={wall}s "
        f"valid={rep['validate_ok']} mcts={n_mcts_calls} {rep['validate_summary']}",
        flush=True,
    )
    return rep


solve_standalone = solve  # alias


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Standalone CostTracer + greed–coop neural MCTS MAPD"
    )
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument("--max-sim-time", type=int, default=200000)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--greed",
        type=float,
        default=0.55,
        help="weight on personal progress (greed)",
    )
    ap.add_argument(
        "--coop",
        type=float,
        default=0.35,
        help="weight on conflict penalty (cooperation)",
    )
    ap.add_argument(
        "--yield-w",
        type=float,
        default=0.10,
        help="weight on soft yield bonus when blocking a closer agent",
    )
    ap.add_argument(
        "--no-traffic-rules",
        action="store_true",
        help="disable frequency-learned traffic rules / temp tasks",
    )
    ap.add_argument(
        "--policy",
        type=str,
        default="auto",
        choices=["auto", "rrsc", "glocal", "rules", "xfmr", "mcts"],
        help="auto/rrsc: road-rights→StageNet→admit→greedy; glocal/rules/xfmr/mcts",
    )
    ap.add_argument(
        "--no-stage-net",
        action="store_true",
        help="disable StageNet (heuristic teacher only still via StageSelector)",
    )
    ap.add_argument(
        "--force-stage-net",
        action="store_true",
        help="always use StageNet (bypass Adaptive Stage Authority)",
    )
    ap.add_argument(
        "--rule-net",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt-in GlobalRuleNet (default off: goal-flow teacher only)",
    )
    ap.add_argument(
        "--cgpn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASDCR residual (opt-in; floods≥4 skipped; may hurt some maps)",
    )
    ap.add_argument(
        "--facn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FACN fleet concurrency net (default on; teacher if no ckpt)",
    )
    ap.add_argument(
        "--force-fleet-cap",
        type=int,
        default=None,
        help="oracle: hard-set max concurrent busy (disables FACN)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu recommended for deterministic SH01-20 (cuda can inflate sim)",
    )
    ap.add_argument(
        "--lifelong-random",
        action="store_true",
        help="after seed CSV tasks finish, keep spawning random station→dest tasks",
    )
    ap.add_argument("--lifelong-seed", type=int, default=0)
    ap.add_argument("--lifelong-batch", type=int, default=24)
    ap.add_argument("--lifelong-low-water", type=int, default=8)
    ap.add_argument(
        "--lifelong-max-total",
        type=int,
        default=None,
        help="stop offering new tasks once total offered reaches this (seed+stream)",
    )
    ap.add_argument(
        "--task-csv",
        type=str,
        default="",
        help="override task CSV path (e.g. 400-task seed for lifelong)",
    )
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = "cpu"
    use_cgpn = bool(args.cgpn)
    solve(
        int(args.slot),
        task_csv=(Path(args.task_csv) if str(args.task_csv).strip() else None),
        wall_timeout=float(args.wall_timeout),
        max_sim_time=int(args.max_sim_time),
        num_simulations=int(args.sims),
        k_proximal=int(args.k),
        greed_w=float(args.greed),
        coop_w=float(args.coop),
        yield_w=float(args.yield_w),
        use_traffic_rules=not bool(args.no_traffic_rules),
        policy=str(args.policy),
        use_stage_net=not bool(args.no_stage_net),
        force_stage_net=bool(args.force_stage_net),
        use_rule_net=bool(args.rule_net),
        use_cgpn=use_cgpn,
        use_facn=bool(args.facn),
        force_fleet_cap=(
            None
            if args.force_fleet_cap is None
            else int(args.force_fleet_cap)
        ),
        device=device,
        lifelong_random=bool(args.lifelong_random),
        lifelong_seed=int(args.lifelong_seed),
        lifelong_batch=int(args.lifelong_batch),
        lifelong_low_water=int(args.lifelong_low_water),
        lifelong_max_total=(
            None
            if args.lifelong_max_total is None
            else int(args.lifelong_max_total)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
