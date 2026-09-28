"""Urgent-aware SwapNet policy: minimal hard rules + learned soft gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ml_research.benchmarks.common import is_urgent_task

from .features import SwapContext, build_swap_features
from .model import SwapNet


@dataclass
class SwapDecision:
    swap: bool
    source: str
    prob: float = 0.0
    reason: str = ""


FAR_ARRIVAL_GUARD = 2.0


def urgent_aware_decide(
    ctx: SwapContext,
    net: Optional[SwapNet],
    *,
    threshold_normal: float = 0.48,
    threshold_urgent: float = 0.40,
) -> SwapDecision:
    """Hard rules only for irreversible harm; soft boundaries → network.

    Hard NO-SWAP (kept):
      - far almost at dock (sunk cost)
      - near holds urgent while far task is normal (don't preempt)
      - both miss SLA (swap cannot help deadline)

    Soft (learned, NOT hard-gated):
      - gap vs margin, deep queue, urgent already-ok, tiny gap, SLA rescue
    """
    urgent = is_urgent_task(ctx.task_info)
    rt = ctx.task_info.get("remaining_time")
    gap = ctx.far_eta - ctx.near_eta
    margin = max(1.0, float(ctx.margin))

    if ctx.far_eta <= FAR_ARRIVAL_GUARD:
        return SwapDecision(False, "rule", reason="far_at_door")

    near_urgent = ctx.near_task_info and is_urgent_task(ctx.near_task_info)
    if near_urgent and not urgent:
        return SwapDecision(False, "rule", reason="near_holds_urgent")

    if urgent and rt is not None:
        try:
            rt_f = float(rt)
        except (TypeError, ValueError):
            rt_f = None
        if rt_f is not None and rt_f > 0:
            if ctx.near_eta > rt_f and ctx.far_eta > rt_f:
                return SwapDecision(False, "rule", reason="urgent_both_miss_sla")

    x = build_swap_features(ctx)
    if net is None:
        # Fallback without net: conservative margin only
        return SwapDecision(gap >= margin, "rule", reason="margin_only")

    p = net.prob(x)
    thr = threshold_urgent if urgent else threshold_normal
    ok = p >= thr
    return SwapDecision(
        ok,
        "net",
        prob=p,
        reason=f"urgent_thr={thr:.2f}" if urgent else f"normal_thr={thr:.2f}",
    )


def make_swap_policy(
    net: Optional[SwapNet] = None,
    *,
    threshold_normal: float = 0.48,
    threshold_urgent: float = 0.40,
) -> Callable[..., bool]:
    """Callback for station_eta_swap_tick."""

    def policy(
        sim: Any,
        far_agv: Any,
        near_agv: Any,
        pickup_name: str,
        far_eta: float,
        near_eta: float,
        gap: float,
        key: str,
    ) -> bool:
        del gap
        tid = getattr(far_agv, "task_id", None)
        if not tid:
            return False
        task_info = {"task_id": tid, "priority": getattr(far_agv, "priority", "Normal")}
        inflight = getattr(sim, "_inflight_tasks", {}) or {}
        if far_agv.name in inflight:
            task_info = dict(inflight[far_agv.name])
        margin = float(getattr(sim, "_station_eta_margin", 4.0))
        qdepth = 0
        station = pickup_name or key
        if station and hasattr(sim, "task_states"):
            qdepth = len(sim.task_states.get(station, []) or [])
        ctx = SwapContext(
            far_eta=far_eta,
            near_eta=near_eta,
            margin=margin,
            far_dist=far_eta,
            near_dist=near_eta,
            task_info=task_info,
            queue_depth=qdepth,
            near_has_task=bool(getattr(near_agv, "task_id", None)),
            near_task_info=(
                inflight.get(near_agv.name)
                if getattr(near_agv, "task_id", None)
                else None
            ),
        )
        dec = urgent_aware_decide(
            ctx,
            net,
            threshold_normal=threshold_normal,
            threshold_urgent=threshold_urgent,
        )
        stats = getattr(sim, "_replan_stats", None)
        if isinstance(stats, dict):
            if dec.swap:
                stats["swap_policy_yes"] = int(stats.get("swap_policy_yes", 0)) + 1
            else:
                stats["swap_policy_no"] = int(stats.get("swap_policy_no", 0)) + 1
            stats.setdefault("swap_policy_sources", {})
            src = stats["swap_policy_sources"]
            src[dec.source] = int(src.get(dec.source, 0)) + 1
        return dec.swap

    return policy
