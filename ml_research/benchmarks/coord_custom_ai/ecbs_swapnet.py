"""TPTS / SwapNet gate helpers for ECBS assign + mid-wave handoff."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from ml_research.benchmarks.swap_net.features import SwapContext
from ml_research.benchmarks.swap_net.policy import SwapDecision, urgent_aware_decide
from ml_research.common.paths import CKPT

Cell = Tuple[int, int]


def load_swap_decide(
    *,
    enabled: bool,
    margin: float = 4.0,
    threshold_normal: float = 0.48,
    threshold_urgent: float = 0.40,
) -> Tuple[Optional[Callable[..., SwapDecision]], dict]:
    """Return (decide_fn, meta). decide_fn(ctx)->SwapDecision; None if disabled."""
    meta: dict = {"enabled": bool(enabled), "mode": "off", "ckpt": None}
    if not enabled:
        return None, meta

    net = None
    ckpt = CKPT / "swap_net_v3.pt"
    if not ckpt.exists():
        ckpt = CKPT / "swap_net.pt"
    if ckpt.exists():
        try:
            from ml_research.benchmarks.swap_net.model import load_swap_net

            net = load_swap_net(ckpt)
            meta["mode"] = "net"
            meta["ckpt"] = str(ckpt)
        except Exception as exc:  # noqa: BLE001
            meta["mode"] = "rule_fallback"
            meta["load_error"] = str(exc)
    else:
        meta["mode"] = "rule_only"

    def decide(ctx: SwapContext) -> SwapDecision:
        return urgent_aware_decide(
            ctx,
            net,
            threshold_normal=threshold_normal,
            threshold_urgent=threshold_urgent,
        )

    meta["margin"] = float(margin)
    return decide, meta


def tpts_reassign_wave(
    assigned: Dict[str, dict],
    free_agvs: List[str],
    pose: Dict[str, Tuple[int, int, int]],
    static: Set[Cell],
    bfs_len: Callable[[Cell, Cell, Set[Cell]], int],
    decide: Callable[[SwapContext], SwapDecision],
    *,
    margin: float = 4.0,
    queues: Optional[Dict[str, list]] = None,
    stats: Optional[dict] = None,
) -> Dict[str, dict]:
    """Steal wave assignments from far AGVs to nearer free idles (TPTS)."""
    if not assigned or not free_agvs or decide is None:
        return assigned
    out = dict(assigned)
    free = list(free_agvs)
    changed = 0
    # Far owners first (largest ETA)
    owners = sorted(
        out.keys(),
        key=lambda a: (
            -bfs_len(pose[a][:2], tuple(out[a]["pickup_point"]), static),
            a,
        ),
    )
    for far in owners:
        if not free:
            break
        task = out[far]
        pk = tuple(task["pickup_point"])
        far_eta = float(bfs_len(pose[far][:2], pk, static))
        if far_eta >= 10**5:
            continue
        best_near = None
        best_eta = float("inf")
        for a in free:
            e = float(bfs_len(pose[a][:2], pk, static))
            if e < best_eta:
                best_eta = e
                best_near = a
        if best_near is None or best_eta >= float("inf"):
            continue
        gap = far_eta - best_eta
        if gap < 1.0:
            continue
        qdepth = 0
        if queues is not None:
            pname = str(task.get("pickup_name") or "")
            if pname and pname in queues:
                qdepth = len(queues[pname])
        ctx = SwapContext(
            far_eta=far_eta,
            near_eta=best_eta,
            margin=margin,
            far_dist=far_eta,
            near_dist=best_eta,
            task_info=task,
            queue_depth=qdepth,
            near_has_task=False,
            near_task_info=None,
        )
        dec = decide(ctx)
        if stats is not None:
            stats["swap_policy_yes"] = int(stats.get("swap_policy_yes", 0)) + (
                1 if dec.swap else 0
            )
            stats["swap_policy_no"] = int(stats.get("swap_policy_no", 0)) + (
                0 if dec.swap else 1
            )
            src = dict(stats.get("swap_policy_sources") or {})
            src[dec.source] = int(src.get(dec.source, 0)) + 1
            stats["swap_policy_sources"] = src
        if not dec.swap:
            continue
        # Reassign: near takes task, far becomes free
        del out[far]
        out[best_near] = task
        free.remove(best_near)
        free.append(far)
        changed += 1
        if stats is not None:
            stats["station_swap"] = int(stats.get("station_swap", 0)) + 1
        print(
            f"[ECBS-SwapNet] wave-steal {far}(eta={far_eta:.0f}) -> "
            f"{best_near}(eta={best_eta:.0f}) gap={gap:.0f} "
            f"task={task.get('task_id')} src={dec.source}/{dec.reason}",
            flush=True,
        )
    if changed and stats is not None:
        stats["wave_steals"] = int(stats.get("wave_steals", 0)) + changed
    return out


def tpts_mid_handoff(
    pre_next_task: Dict[str, dict],
    pre_next_pick: Dict[str, Cell],
    unload_goals: Dict[str, Cell],
    idle_agvs: List[str],
    pose: Dict[str, Tuple[int, int, int]],
    static: Set[Cell],
    bfs_len: Callable[[Cell, Cell, Set[Cell]], int],
    decide: Callable[[SwapContext], SwapDecision],
    *,
    margin: float = 4.0,
    max_steals: int = 2,
    stats: Optional[dict] = None,
) -> Dict[str, str]:
    """idle -> owner map for mid-wave handoff of early-claimed next tasks."""
    steals: Dict[str, str] = {}
    if not pre_next_task or not idle_agvs or decide is None:
        return steals
    free_idles = list(idle_agvs)
    # Hardest leave legs first (owner unload → next pick)
    owners = sorted(
        pre_next_task.keys(),
        key=lambda a: (
            -bfs_len(
                unload_goals.get(a, pose[a][:2]),
                pre_next_pick.get(a) or tuple(pre_next_task[a]["pickup_point"]),
                static,
            ),
            a,
        ),
    )
    for far in owners:
        if len(steals) >= max_steals or not free_idles:
            break
        task = pre_next_task[far]
        pk = pre_next_pick.get(far) or tuple(task.get("pickup_point"))
        if pk is None:
            continue
        pk = tuple(pk)
        far_from = unload_goals.get(far, pose[far][:2])
        far_eta = float(bfs_len(far_from, pk, static))
        if far_eta >= 10**5:
            continue
        best_near = None
        best_eta = float("inf")
        for a in free_idles:
            e = float(bfs_len(pose[a][:2], pk, static))
            if e < best_eta:
                best_eta = e
                best_near = a
        if best_near is None:
            continue
        gap = far_eta - best_eta
        if gap < margin * 0.5:
            continue
        ctx = SwapContext(
            far_eta=far_eta,
            near_eta=best_eta,
            margin=margin,
            far_dist=far_eta,
            near_dist=best_eta,
            task_info=task,
            queue_depth=0,
            near_has_task=False,
            near_task_info=None,
        )
        dec = decide(ctx)
        if stats is not None:
            stats["swap_policy_yes"] = int(stats.get("swap_policy_yes", 0)) + (
                1 if dec.swap else 0
            )
            stats["swap_policy_no"] = int(stats.get("swap_policy_no", 0)) + (
                0 if dec.swap else 1
            )
            src = dict(stats.get("swap_policy_sources") or {})
            src[dec.source] = int(src.get(dec.source, 0)) + 1
            stats["swap_policy_sources"] = src
        if not dec.swap:
            continue
        steals[best_near] = far
        free_idles.remove(best_near)
        if stats is not None:
            stats["station_swap"] = int(stats.get("station_swap", 0)) + 1
            stats["mid_handoffs"] = int(stats.get("mid_handoffs", 0)) + 1
        print(
            f"[ECBS-SwapNet] mid-handoff {far}(leave~{far_eta:.0f}) -> "
            f"{best_near}(eta={best_eta:.0f}) gap={gap:.0f} "
            f"task={task.get('task_id')} src={dec.source}/{dec.reason}",
            flush=True,
        )
    return steals
