"""When-to-Switch 3-tier policy for Hier M0 ↔ local repair ↔ fleet ECBS.

Papers:
  Skrynnik et al., TNNLS 2023/24 *When to Switch* — Assistant switcher:
      keep the fast policy; only switch when it fails / loops.
  Dergachev & Yakovlev 2021 — locally confined MAPF, then **hand back**.
  Hybrid RL-MAPF 2025 — lock subset for T_lock to stop oscillation.
  LNS2+RL (AAAI'25) — adaptive switch from a short quality queue.

Tiers:
  T0 m0     — spacetime A* + SwapNet
  T1 local  — mats_local MCTS + LNS scatter (already in plateau-recover)
  T2 fleet  — full M0→ECBS handoff (expensive for sim)

This module only **gates T2**. T1 is invoked by plateau-recover.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def update_local_switch_state(
    stall_state: Dict[str, Any],
    *,
    local_stats: Optional[dict],
    assign_ok: int,
    plateau_now: int,
    lock_gates: int = 4,
) -> None:
    """Call after a plateau-recover tick (Assistant / T_lock bookkeeping)."""
    stats = local_stats if isinstance(local_stats, dict) else {}
    moved = int(stats.get("moved") or 0)
    forced = int(stats.get("forced_lns") or 0)
    # Count only MCTS-driven hops — LNS force-scatter must NOT grant T_lock
    # (SH09 switch_v3: forced=2 → moved=2 → lock → handoff 725 / sim 3968).
    pure_moved = max(0, moved - forced)
    unstuck = max(0, int(assign_ok)) + (1 if pure_moved >= 2 else 0)
    lock = int(stall_state.get("local_lock") or 0)
    fail = int(stall_state.get("local_fail") or 0)
    # T_lock only after a real unstick (assign or ≥2 MCTS relocations).
    if unstuck > 0:
        stall_state["local_lock"] = max(lock, int(lock_gates))
        stall_state["local_fail"] = 0
        stall_state["local_last_ok_plateau"] = int(plateau_now)
    else:
        stall_state["local_fail"] = fail + 1
        stall_state["local_lock"] = max(0, lock - 1)
    stall_state["local_last_moved"] = int(moved)
    stall_state["local_last_forced"] = int(forced)


def decide_fleet_escalate(
    *,
    stall_want: bool,
    stall_reason: str,
    stall_state: Dict[str, Any],
    sim_per_task: float,
    plateau_now: int,
    sim30_done: float,
    remaining: int,
    enabled: bool = True,
) -> Tuple[bool, str]:
    """Assistant + T_lock gate on top of StallEscalateNet.

    Returns (do_fleet_handoff, reason). If ``enabled`` is False, pass stall through.

    v3b (after SH09 switch_v3 regression 3968):
      - Hybrid T_lock only after assign / ≥2 pure MCTS moves (not LNS force).
      - Otherwise pass StallEscalateNet through — no mid_map soft delay.
    """
    if not enabled:
        return bool(stall_want), str(stall_reason or "")

    lock = int(stall_state.get("local_lock") or 0)
    fail = int(stall_state.get("local_fail") or 0)

    # Hybrid RL-MAPF T_lock only — Assistant already ran T1 in recover.
    if lock > 0:
        return False, f"switcher_tlock={lock}"

    if stall_want and fail >= 2:
        return True, f"{stall_reason}+switcher_local_dead fail={fail}"
    if stall_want:
        return True, str(stall_reason)
    return False, ""


def switcher_enabled(meta: Optional[dict]) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("use_tier_switcher") is not None:
        return bool(meta.get("use_tier_switcher"))
    import os

    v = os.environ.get("AGV_TIER_SWITCHER")
    if v is None:
        return False
    return str(v).strip().lower() not in ("0", "false", "off", "no")
