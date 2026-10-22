"""Hier hook: mats_local Neural MCTS (K≤3) as LNS-style local unstick on M0 plateau.

Paper alignment:
  - MAPF-LNS2: select colliding/crowded subset, replan with priority / local search
  - When-to-Switch / PUSH: escalate only a local window, not the whole fleet
  - MATS-LP CostTracer: local neural MCTS priors (teacher fallback if no ckpt)

Toggle: meta ``use_local_mcts=True`` or env ``AGV_LOCAL_MCTS=1``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]


def _env_on() -> bool:
    v = os.environ.get("AGV_LOCAL_MCTS")
    if v is None:
        return False
    return str(v).strip().lower() not in ("0", "false", "off", "no")


def local_mcts_enabled(meta: Optional[dict] = None) -> bool:
    if isinstance(meta, dict) and meta.get("use_local_mcts") is not None:
        if bool(meta.get("use_local_mcts")):
            return True
    if isinstance(meta, dict) and bool(meta.get("use_tier_switcher")):
        return True
    if _env_on():
        return True
    v = os.environ.get("AGV_TIER_SWITCHER")
    if v is not None and str(v).strip().lower() not in ("0", "false", "off", "no"):
        return True
    if isinstance(meta, dict) and meta.get("use_local_mcts") is not None:
        return bool(meta.get("use_local_mcts"))
    return False


def _static_obstacles(sim) -> Set[Cell]:
    try:
        obs = set(sim.env.get_static_obstacles())
    except Exception:  # noqa: BLE001
        obs = set()
    # Treat map border outside playable as blocked via obs.legal checks
    return {tuple(c) for c in obs}  # type: ignore[misc]


def _disperse_goals(
    agents: List[Any],
    occupied: Set[Cell],
    static_obs: Set[Cell],
    pads: Set[Cell],
) -> None:
    """Assign synthetic evacuation goals so MCTS prefers Move over Wait.

    Idle agents often have goal=None → teacher prior collapses to Wait.
    Push each agent toward a far free cell away from the cluster centroid.
    """
    if not agents:
        return
    cx = sum(a.pos[0] for a in agents) / len(agents)
    cy = sum(a.pos[1] for a in agents) / len(agents)
    forbidden = set(occupied) | set(static_obs) | set(pads)
    used: Set[Cell] = set()
    for a in agents:
        if a.goal is not None:
            continue
        x, y = int(a.pos[0]), int(a.pos[1])
        # Prefer cells farther from centroid (and from peers).
        best = None
        best_score = -1e9
        for r in range(4, 15):
            for dx in range(-r, r + 1):
                dy = r - abs(dx)
                for sdy in (dy, -dy) if dy else (0,):
                    cand = (x + dx, y + sdy)
                    if not (1 <= cand[0] <= 20 and 1 <= cand[1] <= 20):
                        continue
                    if cand in forbidden or cand in used:
                        continue
                    # Maximize distance from centroid; slight preference for larger r.
                    dist_c = abs(cand[0] - cx) + abs(cand[1] - cy)
                    score = dist_c + 0.15 * r
                    if score > best_score:
                        best_score = score
                        best = cand
            if best is not None and best_score >= 6.0:
                break
        if best is not None:
            a.goal = best
            used.add(best)


def apply_local_mcts_unstick(
    sim,
    *,
    k: int = 3,
    n_sims: int = 64,
    meta: Optional[dict] = None,
) -> Dict[str, Any]:
    """Run local MCTS on a crowded idle cluster; relocate movers one hop / hold waiters.

    Returns stats dict: ``{applied, moved, waited, acts, why}``.
    """
    out: Dict[str, Any] = {
        "applied": False,
        "moved": 0,
        "waited": 0,
        "acts": {},
        "why": "",
    }
    if not local_mcts_enabled(meta):
        out["why"] = "disabled"
        return out

    try:
        from ml_research.benchmarks.mats_local.mcts import (
            LocalAgent,
            MCTSConfig,
            NeuralLocalMCTS,
        )
        from ml_research.benchmarks.mats_local.obs import DIRS, apply_action
        from ml_research.benchmarks.mats_local.runtime import (
            extract_agents_from_sim,
            pick_ego_crowded,
        )
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"import:{exc}"
        return out

    prefixes = ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
    # Prefer true-idle / finished-escape AGVs for unstick
    candidates: List[Any] = []
    t_now = int(getattr(sim, "time", 0) or 0)
    for agv in list(getattr(sim, "agvs", None) or []):
        tid = getattr(agv, "task_id", None)
        if tid is None:
            candidates.append(agv)
            continue
        if not str(tid).startswith(prefixes):
            continue
        path = getattr(agv, "path", None) or []
        if not path or t_now >= len(path) - 1:
            candidates.append(agv)

    if len(candidates) < 2:
        out["why"] = "few_idle"
        return out

    # Temporarily expose only candidates to extractor via slim list
    all_agvs = list(getattr(sim, "agvs", None) or [])
    try:
        sim.agvs = candidates
        agents = extract_agents_from_sim(sim, max_agents=12)
    finally:
        sim.agvs = all_agvs

    if len(agents) < 2:
        out["why"] = "few_agents"
        return out

    ordered = pick_ego_crowded(agents)
    subset = ordered[: max(2, min(int(k), 3))]
    obstacles = _static_obstacles(sim)
    occupied, static_obs = sim._occupied_cells_now()
    pads: Set[Cell] = set()
    try:
        pads = set(sim._unload_pad_set() or [])
    except Exception:  # noqa: BLE001
        pads = set()
    _disperse_goals(subset, occupied, static_obs, pads)

    net = None
    try:
        from ml_research.benchmarks.mats_local.network import (
            COSTTRACER_CKPT,
            load_cost_tracer,
        )

        if COSTTRACER_CKPT.is_file():
            net = load_cost_tracer()
    except Exception:  # noqa: BLE001
        net = None

    mcts = NeuralLocalMCTS(
        net=net,
        cfg=MCTSConfig(num_simulations=int(n_sims), k_proximal=len(subset), horizon=6),
        device="cpu",
    )
    acts = mcts.plan(subset, list(obstacles))
    out["acts"] = dict(acts)
    if not acts:
        out["why"] = "empty_plan"
        return out

    name_to_agv = {
        str(getattr(a, "name", "") or getattr(a, "id", "")): a for a in candidates
    }
    moved = 0
    waited = 0
    forced = 0
    goal_hops = 0

    for name, act in acts.items():
        agv = name_to_agv.get(str(name))
        if agv is None:
            continue
        pos = tuple(agv.state[:2])
        if act == "Wait":
            waited += 1
            continue
        delta = DIRS.get(act)
        if not delta:
            waited += 1
            continue
        goal = apply_action(pos, act, obstacles)  # type: ignore[arg-type]
        if goal == pos or goal in occupied:
            waited += 1
            continue
        # Clear finished escape id so relocate can take over
        tid = getattr(agv, "task_id", None)
        if tid is not None and str(tid).startswith(prefixes):
            agv.task_id = None
        # Prefer multi-hop toward synthetic disperse goal when available
        la = next((x for x in subset if x.name == name), None)
        relocate_goal = goal
        if la is not None and la.goal is not None:
            gx, gy = int(la.goal[0]), int(la.goal[1])
            # Take up to 3 steps toward disperse goal along MCTS first hop ray
            if abs(gx - pos[0]) + abs(gy - pos[1]) >= 2:
                relocate_goal = (gx, gy)
                goal_hops += 1
        if sim._relocate_idle_agv(agv, relocate_goal, occupied):
            moved += 1
        elif relocate_goal != goal and sim._relocate_idle_agv(agv, goal, occupied):
            moved += 1
        else:
            waited += 1

    # LNS2-style fallback: if MCTS only Wait, force-spread the subset.
    if moved == 0 and waited > 0 and hasattr(sim, "_pick_holding_cell"):
        forbidden = set(pads) | set(static_obs)
        for a in subset:
            agv = name_to_agv.get(str(a.name))
            if agv is None:
                continue
            tid = getattr(agv, "task_id", None)
            if tid is not None and not str(tid).startswith(prefixes):
                continue
            if tid is not None and str(tid).startswith(prefixes):
                path = getattr(agv, "path", None) or []
                if path and t_now < len(path) - 1:
                    continue
                agv.task_id = None
            pos = tuple(agv.state[:2])
            # Prefer synthetic disperse goal if set
            goal = a.goal if a.goal is not None else None
            if goal is None or goal in occupied or goal in forbidden:
                goal = sim._pick_holding_cell(
                    pos, forbidden, occupied, static_obs, min_r=6, max_r=16
                )
            if goal is None:
                goal = sim._pick_holding_cell(
                    pos, forbidden, occupied, static_obs, min_r=8, max_r=18
                )
            if goal is None:
                continue
            if sim._relocate_idle_agv(agv, goal, occupied):
                forced += 1
                forbidden.add(goal)

    out["moved"] = int(moved + forced)
    out["waited"] = int(waited)
    out["forced_lns"] = int(forced)
    out["goal_hops"] = int(goal_hops)
    out["applied"] = bool(moved or forced or waited)
    tag = "mcts+lns" if forced else ("mcts+goal" if goal_hops else "mcts")
    out["why"] = (
        f"{tag} k={len(subset)} moved={moved} wait={waited} "
        f"forced={forced} hops={goal_hops} net={'1' if net is not None else '0'}"
    )
    if out["applied"]:
        print(
            f"[M0] local-mcts unstick {out['why']} acts={acts}",
            flush=True,
        )
    return out
