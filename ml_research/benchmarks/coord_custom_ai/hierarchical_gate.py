"""Map + lifelong runtime gate: ternary scene → planner.

Architecture contract:
  ``ml_research/benchmarks/hier_coord/HIER_EASY_M0_TRANSITION.md``

Bands (committed, with hysteresis):
  easy   → **true** M0 engine + SwapNet (``simulation/engine.py`` / main copy),
           never the wave-batched fake-astar inside ``solve_ecbs``
  medium → hybrid (small joint core) — **only** as easy→hard transition
  hard   → larger ECBS joint core

Hotspot / joint_fail on an easy band must **not** preemptively invent medium;
they may shrink k once already on medium/hard. Medium starts when the selector
actually climbs toward hard (scene/map hard), stepping easy→medium first.

``suggested_k`` is an **ECBS/hybrid-only** wave size. The A* / M0 easy path
fills free AGVs via the continuous engine loop (FIFO surface heads).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]

_GRID_CELLS = 20 * 20

_SCENE_RANK = {"easy": 0, "medium": 1, "hard": 2}


def band_scene_label(
    p_hard: float,
    *,
    t1: float = 0.12,
    t2: float = 0.35,
) -> Tuple[str, str]:
    """Map scalar p_hard → (easy|medium|hard, k_policy)."""
    p = float(p_hard)
    lo = float(t1)
    hi = float(t2)
    if hi < lo:
        lo, hi = hi, lo
    if p < lo:
        return "easy", "shrink_ok"
    if p < hi:
        return "medium", "shrink_ok"
    return "hard", "full_k"


def commit_scene_label(
    committed: str,
    raw: str,
    *,
    calm_streak: int,
    calm_needed: int,
    hotspot_pressure: bool = False,
) -> Tuple[str, int, bool]:
    """Upgrade at most one band/wave; downgrade only after calm.

    Returns (label, calm, down).
    ``hotspot_pressure`` is ignored for scene elevation (kept for API compat).
    Climbing easy→hard lands on **medium** first so hybrid is the transition rung,
    not an easy-band preemption.

    Downgrade is **asymmetric**: once calm, jump straight to ``raw`` (typically
    easy/A*). No hard→medium→easy ladder — ECBS/hybrid may return to A* in one
    step when AllowDeescalate predicates pass (enforced by the caller).
    """
    del hotspot_pressure  # must not preempt easy → medium
    raw_l = str(raw or "easy")
    # Empty committed = start on easy (do NOT collapse to raw)
    cur = str(committed) if str(committed or "").strip() else "easy"
    if cur not in _SCENE_RANK:
        cur = "easy"
    if raw_l not in _SCENE_RANK:
        raw_l = "easy"
    if _SCENE_RANK[raw_l] > _SCENE_RANK[cur]:
        # At most +1 band: easy --(toward hard)--> medium --> hard
        if _SCENE_RANK[raw_l] >= _SCENE_RANK[cur] + 2:
            step = next(
                lab
                for lab, rk in _SCENE_RANK.items()
                if rk == _SCENE_RANK[cur] + 1
            )
            return step, 0, False
        return raw_l, 0, False
    if raw_l == cur:
        return cur, calm_streak + 1, False
    # Downgrade: after calm, land directly on raw (hard/medium → easy/A*).
    nxt_calm = calm_streak + 1
    if nxt_calm >= max(1, int(calm_needed)):
        return raw_l, 0, True
    return cur, nxt_calm, False


@dataclass
class GateState:
    """Escalation decision for one lifelong wave."""

    escalate: bool
    hardness: float  # effective (map ∨ runtime)
    map_hardness: float
    runtime_hardness: float
    dropoff_score: float
    reason: str
    obs_ratio: float
    suggested_k: int
    top_dest: str = ""
    top_dest_count: int = 0
    deescalated: bool = False
    # SceneDifficultyNet → planner branch (ternary scene bands):
    #   easy   → "astar"  (M0-like prioritized spacetime A* + SwapNet)
    #   medium → "hybrid" (small ECBS core + A* bystanders; transition)
    #   hard   → "ecbs"   (larger ECBS core + A* bystanders via split)
    planner: str = "ecbs"
    scene_label: str = ""  # easy | medium | hard (committed, with hysteresis)


@dataclass
class LifelongGateTracker:
    """Per-solve tracker: re-evaluate every wave; recover to fast path when calm."""

    map_hardness: float
    # 1.0 when one free cell is the only link between two regions.
    narrow_cut: float = 0.0
    threshold: float = 0.08
    force: bool = False
    hard_cap: int = 4
    max_active: int = 8
    # Hysteresis: need this many consecutive calm waves before leaving escalate.
    calm_waves_to_release: int = 2
    dropoff_escalate_score: float = 0.45
    escalate_net: object = None  # Optional EscalateNet (scalar) or EscalateNetMap
    scene_diff_net: object = None  # Optional SceneDifficultyNet (map+task hard/easy)
    # Off by default: want_on is ignored on easy band; real upgrades use
    # force_commit / map hardness. Opt-in only for EscalateNet experiments.
    use_escalate_ai: bool = False
    use_map_obs: bool = True
    # Off by default: p_hard collapsed to easy; real upgrades use force_commit /
    # map hardness. Opt-in only for SceneNet experiments.
    use_scene_diff: bool = False
    # Optional live scene for map obs (set each wave by solver)
    scene_static: Optional[Set[Cell]] = None
    scene_stations: Optional[Set[Cell]] = None
    scene_pose: Optional[Dict[str, Tuple[int, int, int]]] = None
    # last scene-diff decision (raw band before hysteresis commit)
    last_scene_label: str = ""
    last_k_policy: str = ""
    last_scene_p_hard: float = 0.0
    scene_decision_log: list = field(default_factory=list)
    # Ternary p_hard bands: easy < t1 ≤ medium < t2 ≤ hard
    scene_t1: float = 0.12
    scene_t2: float = 0.35
    # When False, static-easy episodes short-circuit to M0 and run to completion.
    # When True, easy still starts M0 but may yield mid-run (see HIER_EASY_M0_TRANSITION.md).
    allow_mode_switch: bool = False
    # Internal
    escalated: bool = False
    calm_streak: int = 0
    waves: int = 0
    n_escalate: int = 0
    n_passthrough: int = 0
    n_planner_astar: int = 0
    n_planner_hybrid: int = 0
    n_planner_ecbs: int = 0
    n_runtime_trigger: int = 0
    n_deescalate: int = 0
    n_ai_decisions: int = 0
    # Committed ternary label with slow downgrade
    committed_scene: str = ""
    scene_calm_streak: int = 0
    # Anti-pingpong: block band downgrade until sim_t >= last_escalate + dwell.
    last_escalate_sim_t: int = -10**9
    deescalate_dwell_sim: int = 100
    # After warm M0 handback, temporarily raise re-escalate sensitivity until this t.
    escalate_boost_until_sim_t: int = 0
    escalate_boost_factor: float = 2.0
    last: Optional[GateState] = None
    history: List[dict] = field(default_factory=list)
    last_map_obs: Optional[object] = None

    def set_scene(
        self,
        *,
        static: Set[Cell],
        stations: Set[Cell],
        pose: Dict[str, Tuple[int, int, int]],
    ) -> None:
        self.scene_static = static
        self.scene_stations = stations
        self.scene_pose = pose

    def note_escalate(self, sim_t: int, *, reason: str = "") -> None:
        """Record an upward band change for dwell / anti-pingpong."""
        self.last_escalate_sim_t = int(sim_t)
        if reason:
            self.history.append(
                {"note_escalate_t": int(sim_t), "reason": str(reason)}
            )

    def note_handback(self, sim_t: int, *, dwell: Optional[int] = None) -> None:
        """After warm M0 handback: raise re-escalate bar for ``dwell`` sim ticks."""
        d = int(dwell if dwell is not None else self.deescalate_dwell_sim)
        self.escalate_boost_until_sim_t = int(sim_t) + max(0, d)
        self.history.append(
            {
                "note_handback_t": int(sim_t),
                "boost_until": int(self.escalate_boost_until_sim_t),
            }
        )

    def _allow_deescalate(
        self,
        *,
        sim_t: int,
        recent_joint_fail: int,
        no_progress_waves: int,
        serial_recover_recent: bool,
    ) -> Tuple[bool, str]:
        """AllowDeescalate predicate (ECBS/hybrid → lower band)."""
        if bool(self.force):
            return False, "force_hard"
        dwell = max(0, int(self.deescalate_dwell_sim))
        if int(sim_t) < int(self.last_escalate_sim_t) + dwell:
            return False, (
                f"dwell:{sim_t}<{int(self.last_escalate_sim_t)+dwell}"
            )
        if int(recent_joint_fail) >= 1:
            return False, f"joint_fail:{recent_joint_fail}"
        if int(no_progress_waves) >= 2:
            return False, f"no_progress:{no_progress_waves}"
        if bool(serial_recover_recent):
            return False, "serial_recover"
        return True, "ok"

    def force_commit_at_least(
        self, band: str, *, reason: str = "", sim_t: Optional[int] = None
    ) -> str:
        """Runtime hard escalate: commit at least ``band`` (+1 step max).

        Used when M0 is livelocked (all idle + surface work) so SceneDifficulty
        alone would never leave easy. ``easy`` + want ``medium`` feeds raw
        ``hard`` so ``commit_scene_label`` lands on the medium transition rung.
        """
        want = str(band or "medium").strip().lower()
        if want not in _SCENE_RANK:
            want = "medium"
        cur = str(self.committed_scene or "").strip() or "easy"
        if cur not in _SCENE_RANK:
            cur = "easy"
        if _SCENE_RANK[cur] >= _SCENE_RANK[want]:
            self.last_scene_label = cur
            return cur
        # Aim upward; commit caps at +1 band per call.
        if cur == "easy" and want == "medium":
            raw = "hard"
        elif _SCENE_RANK[want] >= _SCENE_RANK[cur] + 2:
            raw = "hard"
        else:
            raw = want
        prev = str(self.committed_scene or "")
        committed, self.scene_calm_streak, _down = commit_scene_label(
            self.committed_scene,
            raw,
            calm_streak=int(self.scene_calm_streak),
            calm_needed=int(self.calm_waves_to_release),
            hotspot_pressure=False,
        )
        self.committed_scene = committed
        self.last_scene_label = committed
        if committed in ("medium", "hard"):
            self.escalated = True
            self.calm_streak = 0
            self.n_runtime_trigger += 1
            if (
                _SCENE_RANK.get(committed, 0) > _SCENE_RANK.get(prev or "easy", 0)
                and sim_t is not None
            ):
                self.note_escalate(int(sim_t), reason=str(reason or "force_commit"))
        tag = str(reason or f"force_commit>={want}")
        self.history.append(
            {
                "force_commit": committed,
                "want": want,
                "raw": raw,
                "reason": tag,
                "sim_t": sim_t,
            }
        )
        return committed

    def decide(
        self,
        *,
        assigned: Optional[Dict[str, dict]] = None,
        queues: Optional[Dict[str, list]] = None,
        recent_joint_fail: int = 0,
        sim_t: int = 0,
        no_progress_waves: int = 0,
        serial_recover_recent: bool = False,
    ) -> GateState:
        self.waves += 1
        drop_score, drop_meta = dropoff_hotspot_score(
            assigned or {},
            queues or {},
        )
        fail_h = min(1.0, 0.18 * float(max(0, recent_joint_fail)))
        pickup_s = float(drop_meta.get("pickup_score") or 0.0)
        runtime_h = max(float(drop_score), fail_h, pickup_s)
        map_h = float(self.map_hardness)
        effective = max(map_h, runtime_h)
        qdepth = sum(len(v) for v in (queues or {}).values())

        reasons: List[str] = []
        want_on = False
        ai_src = ""
        map_obs = None

        if (
            self.use_map_obs
            and self.scene_static is not None
            and self.scene_stations is not None
            and self.scene_pose is not None
        ):
            from ml_research.benchmarks.hier_coord.obs import build_hier_map_obs
            from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
                truncate_queues_horizon,
            )

            # SceneDifficulty / escalate map nets should see near-horizon queues
            # only (not the entire lifelong backlog).
            n_pose = len(self.scene_pose or {})
            queues_hz = truncate_queues_horizon(
                queues or {}, n_pose, assigned=assigned or {}
            )
            map_obs = build_hier_map_obs(
                static=self.scene_static,
                stations=self.scene_stations,
                pose=self.scene_pose,
                assigned=assigned or {},
                queues=queues_hz,
            )
            self.last_map_obs = map_obs

        if self.force:
            want_on = True
            reasons.append("force")
        elif self.use_escalate_ai:
            try:
                from ml_research.benchmarks.hier_coord.escalate import (
                    EscalateContext,
                    decide_escalate_ai,
                )
                from ml_research.benchmarks.hier_coord.escalate_map import (
                    EscalateNetMap,
                    decide_escalate_map_ai,
                )

                ctx = EscalateContext(
                    map_hardness=map_h,
                    dropoff_score=float(drop_meta.get("dropoff_score") or drop_score),
                    pickup_score=pickup_s,
                    recent_joint_fail=int(recent_joint_fail),
                    n_assigned=len(assigned or {}),
                    queue_depth=int(qdepth),
                    top_share=float(drop_meta.get("share") or 0.0),
                    assigned_top_n=int(drop_meta.get("assigned_top_n") or 0),
                    was_escalated=bool(self.escalated),
                    calm_streak=int(self.calm_streak),
                )
                if isinstance(self.escalate_net, EscalateNetMap) and map_obs is not None:
                    want_on, ai_src, _p = decide_escalate_map_ai(
                        ctx,
                        map_obs,
                        self.escalate_net,
                        dropoff_escalate_score=float(self.dropoff_escalate_score),
                    )
                else:
                    want_on, ai_src, _p = decide_escalate_ai(
                        ctx,
                        self.escalate_net if self.use_escalate_ai else None,
                        rule_threshold=float(self.threshold),
                        dropoff_escalate_score=float(self.dropoff_escalate_score),
                    )
                reasons.append(ai_src)
                self.n_ai_decisions += 1
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"ai_fallback:{exc.__class__.__name__}")
                want_on = False

        if not want_on and not self.force:
            # Legacy rule path (also fills gaps if AI said easy but map is hard)
            # Skip scalar map_hardness hard-force when map-net already voted.
            if map_h >= float(self.threshold) and not str(ai_src).startswith("mapnet"):
                want_on = True
                reasons.append("map_hardness")
            if drop_score >= float(self.dropoff_escalate_score):
                want_on = True
                kind = str(drop_meta.get("kind") or "dropoff")
                reasons.append(f"{kind}_hotspot")
            if recent_joint_fail > 0:
                want_on = True
                reasons.append(f"joint_fail:{recent_joint_fail}")
            if int(drop_meta.get("assigned_top_n") or 0) >= 3:
                want_on = True
                kind = str(drop_meta.get("kind") or "dropoff")
                tag = f"assigned_same_{kind}"
                if tag not in reasons and f"{kind}_hotspot" not in reasons:
                    reasons.append(tag)

        # SceneDifficultyNet (map+task): p_hard → easy/medium/hard bands
        k_policy = ""
        scene_src = ""
        p_hard = 0.0
        raw_scene = ""
        if (
            self.use_scene_diff
            and self.scene_diff_net is not None
            and map_obs is not None
        ):
            try:
                from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
                    build_scene_feats_live,
                    decide_scene_difficulty,
                )

                feats = build_scene_feats_live(
                    assigned=assigned or {},
                    queues=queues or {},
                    n_agvs=len(self.scene_pose or {}),
                    n_obstacles=len(self.scene_static or []),
                    dropoff_score=float(drop_meta.get("dropoff_score") or drop_score),
                    pickup_score=float(drop_meta.get("pickup_score") or 0.0),
                )
                _is_hard, scene_src, p_hard, _kp = decide_scene_difficulty(
                    map_obs, feats, self.scene_diff_net
                )
                # Prefer net thresholds when present
                t1 = float(
                    getattr(self.scene_diff_net, "easy_max", None) or self.scene_t1
                )
                t2 = float(
                    getattr(self.scene_diff_net, "decision_threshold", None)
                    or self.scene_t2
                )
                self.scene_t1, self.scene_t2 = t1, t2
                raw_scene, k_policy = band_scene_label(p_hard, t1=t1, t2=t2)
                self.last_scene_p_hard = float(p_hard)
                reasons.append(scene_src)
                reasons.append(f"scene_band:{raw_scene}")
                if raw_scene == "hard":
                    want_on = True
                elif raw_scene == "medium":
                    want_on = True
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"scene_diff_fallback:{exc.__class__.__name__}")
                k_policy = ""
                raw_scene = ""

        # Explicit force always wins over SceneDifficultyNet (e.g. SH03 hutong demos).
        if self.force:
            raw_scene = "hard"
            k_policy = k_policy or "full_k"
            want_on = True
            if "force_hard_override" not in reasons:
                reasons.append("force_hard_override")

        hotspot_pressure = (
            float(drop_score) >= float(self.dropoff_escalate_score)
            or float(pickup_s) >= float(self.dropoff_escalate_score)
            or int(recent_joint_fail) > 0
            or int(drop_meta.get("assigned_top_n") or 0) >= 3
        )

        if not raw_scene:
            # No SceneDifficultyNet: do NOT invent hard from mild map_hardness
            # (SH01 ~0.11 must stay easy→M0).
            #
            # Under allow_mode_switch the online policy is: prefer M0, escalate
            # to ECBS only when runtime stall/structure fires (hard_livelock…).
            # Density (map_h ≥ scene_t2) alone must NOT cold-start ECBS —
            # that bypassed automatic switching and broke maps like SH16 where
            # M0 finishes VALID but a hard-open ECBS traj was illegal.
            # Only a real 1-wide topological cut still opens hard.
            if self.force:
                raw_scene, k_policy = "hard", "full_k"
            elif bool(self.allow_mode_switch) and float(self.narrow_cut) >= 1.0:
                raw_scene, k_policy = "hard", "full_k"
                reasons.append("narrow_cut_start_hard")
            elif bool(self.allow_mode_switch):
                # Density / hotspot noted but stay easy→M0; escalate online.
                n_agv = int(len(self.scene_pose or {}))
                fleet = min(1.0, max(0.0, (float(n_agv) - 6.0) / 10.0))
                pressure = 0.65 * float(map_h) + 0.35 * fleet
                raw_scene, k_policy = "easy", "shrink_ok"
                if pressure >= 0.28:
                    reasons.append(f"start_pressure_note:{pressure:.3f}+agv{n_agv}")
                if map_h >= float(self.threshold):
                    reasons.append("map_hardness_keep_easy")
                if map_h >= float(self.scene_t2):
                    reasons.append("map_hardness_defer_ecbs")
                if hotspot_pressure:
                    reasons.append("hotspot_keep_easy")
            else:
                raw_scene, k_policy = "easy", "shrink_ok"
                if map_h >= float(self.threshold):
                    reasons.append("map_hardness_keep_easy")
                if hotspot_pressure:
                    reasons.append("hotspot_keep_easy")

        prev_committed = str(self.committed_scene or "")
        committed, self.scene_calm_streak, down = commit_scene_label(
            self.committed_scene,
            raw_scene,
            calm_streak=int(self.scene_calm_streak),
            calm_needed=int(self.calm_waves_to_release),
            hotspot_pressure=False,
        )
        if down:
            ok_down, veto = self._allow_deescalate(
                sim_t=int(sim_t),
                recent_joint_fail=int(recent_joint_fail),
                no_progress_waves=int(no_progress_waves),
                serial_recover_recent=bool(serial_recover_recent),
            )
            if not ok_down:
                # Freeze band; keep calm ready so we retry as soon as veto clears.
                committed = prev_committed or committed
                self.scene_calm_streak = max(
                    int(self.scene_calm_streak),
                    int(self.calm_waves_to_release),
                )
                down = False
                reasons.append(f"deescalate_veto:{veto}")
            else:
                reasons.append(f"scene_downgrade:{prev_committed}->{committed}")
        if (
            committed != prev_committed
            and _SCENE_RANK.get(committed, 0) > _SCENE_RANK.get(prev_committed or "easy", 0)
            and str(raw_scene) == "hard"
            and committed == "medium"
        ):
            reasons.append("scene_step:easy->medium(toward_hard)")
        if (
            committed != prev_committed
            and _SCENE_RANK.get(committed, 0) > _SCENE_RANK.get(prev_committed or "easy", 0)
        ):
            self.note_escalate(int(sim_t), reason="band_upgrade")
        self.committed_scene = committed
        self.last_scene_label = committed
        self.last_k_policy = k_policy or self.last_k_policy

        deescalated = False
        # Planner escalate flag follows committed band — not easy-band hotspot noise.
        if self.force:
            self.escalated = True
            self.calm_streak = 0
        elif committed in ("medium", "hard"):
            if not self.escalated and any(
                ("hotspot" in r)
                or r.startswith("assigned_same")
                or r.startswith("rule_hotspot")
                or r.startswith("scene_band:hard")
                or r.startswith("map_hardness")
                for r in reasons
            ):
                self.n_runtime_trigger += 1
            self.escalated = True
            self.calm_streak = 0
        else:
            # Easy band: ignore want_on/hotspot for escalated sticky bit.
            self.calm_streak += 1
            if self.escalated and self.calm_streak >= int(self.calm_waves_to_release):
                self.escalated = False
                deescalated = True
                self.n_deescalate += 1
                reasons.append("recovered_fast")
            elif self.escalated:
                reasons.append(f"hysteresis_calm:{self.calm_streak}")
            else:
                if not reasons:
                    reasons.append("easy_default")
                elif hotspot_pressure and "hotspot_keep_easy" not in reasons:
                    reasons.append("hotspot_keep_easy")

        reason = "+".join(reasons) if reasons else "easy_default"
        # Wave size: scene-net k_policy overrides scalar map_hardness when present
        k = int(self.max_active)

        # Planner from committed ternary scene (not raw binary hard/easy).
        if committed == "hard":
            planner = "ecbs"
            reasons.append("planner_ecbs_hard")
        elif committed == "medium":
            planner = "hybrid"
            reasons.append("planner_hybrid_medium")
        else:
            planner = "astar"
            reasons.append("planner_astar_m0")
        reason = "+".join(reasons)

        # ECBS/hybrid k policy:
        # - medium → conflict core up to hard_cap (floor 3) — NOT ≤2 (v2 livelock)
        # - hard → suggested_k capped by hard_cap; joint_fail shrinks
        # - hotspot on shrink_ok → mild shrink
        joint_fail_n = int(recent_joint_fail)
        if planner in ("ecbs", "hybrid"):
            if planner == "hybrid":
                cap = int(self.hard_cap) if int(self.hard_cap) > 0 else 4
                # Prefer enough agents to clear a hub jam; never stick at k=2.
                floor_m = min(3, cap, int(self.max_active))
                k = max(floor_m, min(cap, int(self.max_active), max(1, int(k))))
                # Hotspot / joint fail: still allow up to hard_cap
                if hotspot_pressure or joint_fail_n > 0:
                    k = max(k, min(cap, int(self.max_active)))
                reasons.append(f"hybrid_k_core:{k}")
                reason = "+".join(reasons)
            elif joint_fail_n > 0:
                # Hard/full_k: keep concurrency at hard_cap — peeling to 2/1
                # serializes makespan into tens of thousands of ticks.
                cap = int(self.hard_cap) if int(self.hard_cap) > 0 else 4
                floor = min(3, cap) if committed == "hard" else 2
                if committed == "hard" or k_policy == "full_k":
                    k = max(floor, min(cap, int(self.max_active)))
                    reasons.append(f"keep_k_hard_cap:{k}+fail{joint_fail_n}")
                else:
                    k = suggested_wave_k(
                        float(runtime_h), self.max_active, hard_cap=self.hard_cap
                    )
                    k = min(
                        k,
                        max(
                            floor,
                            int(self.hard_cap)
                            - min(
                                joint_fail_n - 1,
                                max(0, int(self.hard_cap) - floor),
                            ),
                        ),
                    )
                    k = max(floor, min(int(k), int(self.max_active)))
                    reasons.append(
                        f"shrink_joint_fail:{joint_fail_n}+floor{floor}"
                    )
                reason = "+".join(reasons)
            elif k_policy == "full_k" and self.escalated:
                reasons.append("full_k_scene_hard")
                reason = "+".join(reasons)
            elif (
                k_policy == "shrink_ok"
                and self.escalated
                and hotspot_pressure
            ):
                k = suggested_wave_k(
                    float(runtime_h), self.max_active, hard_cap=self.hard_cap
                )
                k = min(k, max(2, int(self.hard_cap)))
                reasons.append("shrink_scene_easy")
                reason = "+".join(reasons)
            elif not k_policy:
                if (
                    self.escalated
                    and hotspot_pressure
                    and map_h < float(self.threshold)
                ):
                    k = suggested_wave_k(
                        float(runtime_h), self.max_active, hard_cap=self.hard_cap
                    )
                    k = min(k, max(2, int(self.hard_cap)))
                    reasons.append("shrink_hotspot")
                    reason = "+".join(reasons)
                elif self.escalated and map_h >= float(self.threshold):
                    reasons.append("full_k_hardmap")
                    reason = "+".join(reasons)

        # A* baseline: WaveNet shrink off; suggested_k unused by claim (full fleet).
        # ECBS/hybrid: escalate + suggested_k drive WaveNet / claim_cap.
        wave_escalate = bool(self.escalated) and planner in ("ecbs", "hybrid")
        if planner == "astar":
            k = int(self.max_active)  # informational only; claim ignores k-cap

        st = GateState(
            escalate=bool(wave_escalate),
            hardness=float(effective),
            map_hardness=map_h,
            runtime_hardness=float(runtime_h),
            dropoff_score=float(drop_score),
            reason=reason,
            obs_ratio=map_h,
            suggested_k=int(k),
            top_dest=str(drop_meta.get("top_dest") or ""),
            top_dest_count=int(drop_meta.get("top_n") or 0),
            deescalated=deescalated,
            planner=str(planner),
            scene_label=str(committed),
        )
        self.last = st
        if st.planner == "astar":
            self.n_planner_astar += 1
        elif st.planner == "hybrid":
            self.n_planner_hybrid += 1
        else:
            self.n_planner_ecbs += 1
        if st.escalate:
            self.n_escalate += 1
        else:
            self.n_passthrough += 1
        self.history.append(
            {
                "wave": self.waves,
                "escalate": st.escalate,
                "planner": st.planner,
                "reason": st.reason,
                "dropoff": st.dropoff_score,
                "runtime_h": st.runtime_hardness,
                "k": st.suggested_k,
                "top_dest": st.top_dest,
                "top_n": st.top_dest_count,
                "deescalated": st.deescalated,
                "ai": ai_src,
                "scene_label": self.last_scene_label,
                "scene_raw": raw_scene,
                "scene_p_hard": round(float(self.last_scene_p_hard), 4),
                "k_policy": self.last_k_policy,
            }
        )
        if self.last_scene_label:
            self.scene_decision_log.append(
                {
                    "wave": self.waves,
                    "label": self.last_scene_label,
                    "raw_label": raw_scene,
                    "p_hard": round(float(self.last_scene_p_hard), 4),
                    "k_policy": self.last_k_policy,
                    "planner": st.planner,
                    "escalate": bool(st.escalate),
                    "suggested_k": int(st.suggested_k),
                    "reason": st.reason,
                    "queue_depth": int(qdepth),
                    "top_dest": st.top_dest,
                    "top_n": st.top_dest_count,
                }
            )
        return st

    def stats_dict(self) -> dict:
        return {
            "map_hardness": float(self.map_hardness),
            "threshold": float(self.threshold),
            "force": bool(self.force),
            "calm_waves_to_release": int(self.calm_waves_to_release),
            "dropoff_escalate_score": float(self.dropoff_escalate_score),
            "use_escalate_ai": bool(self.use_escalate_ai),
            "n_ai_decisions": int(self.n_ai_decisions),
            "waves": int(self.waves),
            "waves_escalated": int(self.n_escalate),
            "waves_passthrough": int(self.n_passthrough),
            "waves_planner_astar": int(self.n_planner_astar),
            "waves_planner_hybrid": int(self.n_planner_hybrid),
            "waves_planner_ecbs": int(self.n_planner_ecbs),
            "runtime_triggers": int(self.n_runtime_trigger),
            "deescalations": int(self.n_deescalate),
            "allow_mode_switch": bool(self.allow_mode_switch),
            "scene_t1": float(self.scene_t1),
            "scene_t2": float(self.scene_t2),
            "committed_scene": str(self.committed_scene),
            "last_escalate_sim_t": int(self.last_escalate_sim_t),
            "deescalate_dwell_sim": int(self.deescalate_dwell_sim),
            "escalate_boost_until_sim_t": int(self.escalate_boost_until_sim_t),
            "last": None
            if self.last is None
            else {
                "escalate": self.last.escalate,
                "reason": self.last.reason,
                "hardness": self.last.hardness,
                "map_hardness": self.last.map_hardness,
                "runtime_hardness": self.last.runtime_hardness,
                "dropoff_score": self.last.dropoff_score,
                "suggested_k": self.last.suggested_k,
                "top_dest": self.last.top_dest,
                "top_dest_count": self.last.top_dest_count,
                "planner": self.last.planner,
                "scene_label": self.last.scene_label,
            },
            "last_scene_label": self.last_scene_label,
            "last_scene_p_hard": float(self.last_scene_p_hard),
            "last_k_policy": self.last_k_policy,
            "scene_decision_log": list(self.scene_decision_log),
            "history_tail": self.history[-12:],
        }


def obstacle_ratio(extra_obstacles: Optional[list], static: Optional[Set[Cell]] = None) -> float:
    n = 0
    if extra_obstacles:
        n = len(extra_obstacles)
    elif static:
        n = max(0, len(static) - 30)
    return float(n) / float(_GRID_CELLS)


def map_hardness(
    *,
    extra_obstacles: Optional[list] = None,
    static: Optional[Set[Cell]] = None,
    free: Optional[Set[Cell]] = None,
) -> float:
    """Scalar in ~[0, 1+]: higher = harder topology."""
    obs_r = obstacle_ratio(extra_obstacles, static)
    free_r = 0.0
    if free is not None:
        free_r = 1.0 - min(1.0, len(free) / float(_GRID_CELLS))
    return float(0.75 * obs_r + 0.25 * free_r)


def narrow_cut_score(free: Optional[Set[Cell]], *, min_side: int = 24) -> float:
    """1.0 when free space is split, or one cell is the only link between two regions.

    Obstacle ratio misses a wide map that is pinched to a one-cell corridor.
    A dead-end pocket smaller than ``min_side`` does not count.
    """
    if not free:
        return 0.0
    cells = [c for c in free if 1 <= int(c[0]) <= 20 and 1 <= int(c[1]) <= 20]
    n = len(cells)
    if n < int(min_side) + 2:
        return 0.0
    index = {c: i for i, c in enumerate(cells)}
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, (x, y) in enumerate(cells):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = index.get((int(x) + dx, int(y) + dy))
            if j is not None and j > i:
                adj[i].append(j)
                adj[j].append(i)

    def _component_sizes(skip: int = -1) -> List[int]:
        seen = [False] * n
        sizes: List[int] = []
        for s in range(n):
            if s == skip or seen[s]:
                continue
            stack = [s]
            seen[s] = True
            sz = 0
            while stack:
                u = stack.pop()
                sz += 1
                for v in adj[u]:
                    if v == skip or seen[v]:
                        continue
                    seen[v] = True
                    stack.append(v)
            sizes.append(sz)
        return sizes

    sizes0 = _component_sizes()
    if sum(1 for s in sizes0 if s >= int(min_side)) >= 2:
        return 1.0

    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    ap = [False] * n
    timer = 0
    for start in range(n):
        if disc[start] != -1:
            continue
        stack: List[Tuple[int, int, int]] = [(start, 0, -1)]
        while stack:
            u, nxt_i, par = stack[-1]
            if disc[u] == -1:
                disc[u] = low[u] = timer
                timer += 1
                parent[u] = par
            if nxt_i < len(adj[u]):
                v = adj[u][nxt_i]
                stack[-1] = (u, nxt_i + 1, par)
                if v == par:
                    continue
                if disc[v] == -1:
                    stack.append((v, 0, u))
                else:
                    low[u] = min(low[u], disc[v])
                continue
            stack.pop()
            if par != -1:
                low[par] = min(low[par], low[u])
                # Non-root cut vertex: a child subtree cannot climb above par.
                if parent[par] != -1 and low[u] >= disc[par]:
                    ap[par] = True
            else:
                children = sum(1 for w in range(n) if parent[w] == u)
                if children > 1:
                    ap[u] = True

    side = int(min_side)
    for i in range(n):
        if not ap[i]:
            continue
        parts = [s for s in _component_sizes(i) if s >= side]
        if len(parts) >= 2:
            return 1.0
    return 0.0


def _task_dest_key(task: dict) -> str:
    d = task.get("destination") or task.get("end_point") or ""
    d = str(d).strip()
    if d:
        return d
    eps = task.get("end_points") or []
    if eps:
        e0 = eps[0]
        try:
            return f"cell:{int(e0[0])},{int(e0[1])}"
        except Exception:  # noqa: BLE001
            return f"cell:{e0}"
    return ""


def _station_hotspot_score(
    keys: List[str],
    *,
    assigned_keys: List[str],
) -> Tuple[float, dict]:
    """Generic concentration score for pickup or dropoff names."""
    if not keys:
        return 0.0, {
            "top": "",
            "top_n": 0,
            "share": 0.0,
            "assigned_top_n": 0,
            "total": 0,
        }
    cnt: Counter = Counter(k for k in keys if k)
    if not cnt:
        return 0.0, {
            "top": "",
            "top_n": 0,
            "share": 0.0,
            "assigned_top_n": 0,
            "total": 0,
        }
    top, top_n = cnt.most_common(1)[0]
    total = int(sum(cnt.values()))
    share = float(top_n) / float(max(1, total))
    asg_cnt: Counter = Counter(k for k in assigned_keys if k)
    assigned_top_n = int(asg_cnt.get(top, 0))
    if asg_cnt:
        assigned_top_n = max(assigned_top_n, int(asg_cnt.most_common(1)[0][1]))

    score = 0.0
    if top_n >= 4:
        score = min(1.0, 0.40 * (top_n / 6.0) + 0.60 * share)
    elif top_n >= 3:
        score = min(1.0, 0.35 * (top_n / 5.0) + 0.55 * share)
    elif share >= 0.7 and total >= 2:
        score = 0.55 * share

    if assigned_top_n >= 3:
        score = max(score, min(1.0, 0.50 + 0.15 * (assigned_top_n - 3)))
    elif assigned_top_n >= 2 and len(assigned_keys) >= 3:
        score = max(score, 0.40)
    return float(score), {
        "top": top,
        "top_n": int(top_n),
        "share": round(share, 4),
        "assigned_top_n": int(assigned_top_n),
        "total": total,
    }


def _task_pickup_key(task: dict) -> str:
    for key in ("pickup_name", "start_point"):
        v = str(task.get(key) or "").strip()
        if v:
            return v
    tid = str(task.get("task_id") or "")
    if "-" in tid:
        return tid.rsplit("-", 1)[0]
    return ""


def dropoff_hotspot_score(
    assigned: Dict[str, dict],
    queues: Dict[str, list],
    *,
    peek: int = 16,
) -> Tuple[float, dict]:
    """Score ∈ [0,1]: dropoff AND/OR pickup concentration (max of both).

    Either shared unload bay or shared pickup stand can stall greedy A* /
    oversized joint waves on an otherwise easy map.
    """
    dest_keys: List[str] = []
    pick_keys: List[str] = []
    asg_dest: List[str] = []
    asg_pick: List[str] = []

    for t in assigned.values():
        d = _task_dest_key(t)
        p = _task_pickup_key(t)
        if d:
            dest_keys.append(d)
            asg_dest.append(d)
        if p:
            pick_keys.append(p)
            asg_pick.append(p)

    n_peek = 0
    stations = list(queues.keys())
    if stations:
        per = max(1, peek // max(1, len(stations)))
        for st in stations:
            for t in (queues.get(st) or [])[:per]:
                d = _task_dest_key(t)
                p = _task_pickup_key(t) or str(st)
                if d:
                    dest_keys.append(d)
                if p:
                    pick_keys.append(p)
                n_peek += 1
                if n_peek >= peek:
                    break
            if n_peek >= peek:
                break

    drop_s, drop_m = _station_hotspot_score(dest_keys, assigned_keys=asg_dest)
    pick_s, pick_m = _station_hotspot_score(pick_keys, assigned_keys=asg_pick)
    score = max(drop_s, pick_s)
    # Prefer annotating the dominant hotspot for logs
    if pick_s > drop_s:
        top, top_n, assigned_top_n = pick_m["top"], pick_m["top_n"], pick_m["assigned_top_n"]
        kind = "pickup"
        share, total = pick_m["share"], pick_m["total"]
    else:
        top, top_n, assigned_top_n = drop_m["top"], drop_m["top_n"], drop_m["assigned_top_n"]
        kind = "dropoff"
        share, total = drop_m["share"], drop_m["total"]

    return float(score), {
        "top_dest": top,  # legacy field name (may be pickup name)
        "top_n": int(top_n),
        "share": share,
        "assigned_top_n": int(assigned_top_n),
        "total": int(total),
        "kind": kind,
        "dropoff_score": float(drop_s),
        "pickup_score": float(pick_s),
    }


def suggested_wave_k(hardness: float, max_active: int, *, hard_cap: int = 4) -> int:
    """Shrink joint window under *runtime* hotspot pressure (easy maps only).

    Map topology hardness must not call this for SH-style hard maps — those
    keep ``max_active`` for pipeline speed.
    """
    k = int(max_active)
    # Cap by hard_cap only — do NOT peel hard_cap-1 (that forced steady k=3
    # under cap=4 and collapsed 400-task mid maps into serial makespans).
    if hardness >= 0.12:
        k = min(k, int(hard_cap) if int(hard_cap) > 0 else k)
    return max(1, k)


def decide_escalate(
    *,
    hardness: float,
    threshold: float = 0.08,
    force: bool = False,
    recent_joint_fail: int = 0,
    max_active: int = 8,
    hard_cap: int = 4,
    dropoff_score: float = 0.0,
    dropoff_escalate_score: float = 0.45,
) -> GateState:
    """Stateless one-shot helper (no hysteresis). Prefer LifelongGateTracker."""
    runtime_h = max(float(dropoff_score), min(1.0, 0.18 * float(max(0, recent_joint_fail))))
    effective = max(float(hardness), runtime_h)
    reasons: List[str] = []
    escalate = False
    if force:
        escalate = True
        reasons.append("force")
    if recent_joint_fail > 0:
        escalate = True
        reasons.append(f"joint_fail:{recent_joint_fail}")
    if dropoff_score >= dropoff_escalate_score:
        escalate = True
        reasons.append("dropoff_hotspot")
    if hardness >= float(threshold):
        escalate = True
        reasons.append("map_hardness")
    if not escalate:
        reasons.append("easy_default")
    if escalate and float(hardness) >= float(threshold):
        k = int(max_active)
        reasons.append("full_k_hardmap")
    elif escalate:
        k = suggested_wave_k(runtime_h, max_active, hard_cap=hard_cap)
        reasons.append("shrink_hotspot")
    else:
        k = int(max_active)
    return GateState(
        escalate=escalate,
        hardness=effective,
        map_hardness=float(hardness),
        runtime_hardness=float(runtime_h),
        dropoff_score=float(dropoff_score),
        reason="+".join(reasons),
        obs_ratio=float(hardness),
        suggested_k=k,
    )
