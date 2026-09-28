"""M5 hooks + cancel/restore — minimal subset for station_eta / SwapNet."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .common import maybe_evacuate_on_block


def _at_task_pickup(sim, agv, t) -> bool:
    del t
    tid = getattr(agv, "task_id", None)
    if not tid:
        return False
    info = (getattr(sim, "surface_tasks", None) or {}).get(tid)
    if info is None:
        assigned = getattr(sim, "assigned_task_info", None) or {}
        info = assigned.get(tid)
    if not info:
        inflight = getattr(sim, "_inflight_tasks", None) or {}
        info = inflight.get(agv.name)
    if not info:
        return True
    # Staged convoy cars wait off-pad; never treat staging as pickup.
    if str(info.get("pipeline_role") or "") == "stage":
        return False
    pk = info.get("true_pickup_point") or info.get("pickup_point")
    if not pk:
        return True
    st = agv.state
    return int(st[0]) == int(pk[0]) and int(st[1]) == int(pk[1])


def m5_replan_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return False


def empty_replan_stats() -> Dict[str, int]:
    return {
        "cancel": 0,
        "reassign": 0,
        "local_replan": 0,
        "tried_clear": 0,
        "evacuate": 0,
        "triggers": 0,
        "station_swap": 0,
        "station_block": 0,
    }


def _is_loaded(agv, t: int) -> bool:
    t = int(t)
    best = None
    for s in agv.steps:
        if int(s.get("timestamp", -1)) == t:
            best = s
            break
    if best is None:
        for s in reversed(agv.steps):
            if int(s.get("timestamp", -1)) <= t:
                best = s
                break
    if best is None:
        return False
    return str(best.get("loaded", "")).lower() in ("true", "1", "yes")


def _has_picked_up(agv, tid, t: int, sim=None) -> bool:
    del t
    if sim is not None:
        committed = getattr(sim, "_pickup_committed", None) or set()
        if str(tid) in committed:
            return True
    return _is_loaded(agv, getattr(sim, "time", 0) if sim else 0)


def _truncate_agv_future(sim, agv) -> None:
    t = int(sim.time)
    if agv.path:
        if t < len(agv.path):
            agv.path = list(agv.path[: t + 1])
        if agv.path:
            agv.state = agv.path[min(t, len(agv.path) - 1)]
    agv.steps = [s for s in agv.steps if int(s.get("timestamp", -1)) <= t]
    traj = sim.env.moving_obstacles.get(agv.name)
    if traj and t < len(traj):
        sim.env.moving_obstacles[agv.name] = list(traj[: t + 1])


def _nkey(tid: str) -> list:
    import re

    return [int(p) if p.isdigit() else p for p in re.split(r"([0-9]+)", str(tid))]


def _restore_task_to_surface(sim, info: dict) -> None:
    """Put cancelled unloaded task back into the station FIFO and refresh surface.

    Pipeline-safe: re-insert by task-id order (not forced to absolute front when
    earlier work remains), then expose the next *unreserved* surface head.
    """
    tid = info["task_id"]
    name = info["pickup_name"]
    queue_item = {
        "task_id": tid,
        "priority": info.get("priority", "Normal"),
        "pickup_point": info["pickup_point"],
        "end_points": list(info.get("end_points") or []),
        "destination": info.get("destination"),
        "remaining_time": info.get("remaining_time"),
        "numbers_before_urgent": info.get("numbers_before_urgent", -1),
    }
    rest = [
        x
        for x in list(sim.task_states.get(name, []))
        if str(x.get("task_id") or "") != str(tid)
    ]
    rest.append(queue_item)
    rest.sort(key=lambda x: _nkey(str(x.get("task_id") or "")))
    sim.task_states[name] = rest
    if hasattr(sim, "promote_next_unreserved"):
        sim.promote_next_unreserved(name)
    else:
        for sid, sinfo in list(sim.surface_tasks.items()):
            if sinfo.get("pickup_name") == name:
                del sim.surface_tasks[sid]
        sim.surface_tasks[tid] = {
            "priority": queue_item["priority"],
            "pickup_point": queue_item["pickup_point"],
            "end_points": list(queue_item["end_points"]),
            "destination": queue_item["destination"],
            "remaining_time": queue_item.get("remaining_time"),
            "numbers_before_urgent": queue_item.get("numbers_before_urgent", -1),
            "pickup_name": name,
        }


def cancel_inflight(sim, agv, reason: str = "") -> bool:
    tid = agv.task_id
    if tid is None or str(tid).startswith("escape_"):
        return False
    info = (getattr(sim, "_inflight_tasks", None) or {}).get(agv.name)
    if not info:
        return False
    committed = getattr(sim, "_pickup_committed", None) or set()
    if str(tid) in committed:
        return False
    if _is_loaded(agv, sim.time):
        return False
    cd = getattr(sim, "_m5_cancel_cd", {})
    if int(cd.get(agv.name, -1)) > int(sim.time):
        return False

    _truncate_agv_future(sim, agv)
    _restore_task_to_surface(sim, info)
    agv.task_id = None
    agv.priority = False
    sim._inflight_tasks.pop(agv.name, None)
    sim.tried_tasks = {(a, t) for a, t in sim.tried_tasks if a != agv.name}
    sim.assign_cooldown.pop(agv.name, None)
    if not hasattr(sim, "_m5_cancel_cd"):
        sim._m5_cancel_cd = {}
    sim._m5_cancel_cd[agv.name] = int(sim.time) + 12
    stats = getattr(sim, "_replan_stats", None)
    if isinstance(stats, dict):
        stats["cancel"] = int(stats.get("cancel", 0)) + 1
    print(f"[station] cancel AGV {agv.name} task {tid} ({reason})", flush=True)
    return True


def ensure_m5_hooks(sim, **kwargs) -> Any:
    """Track in-flight assignments for station_eta / SwapNet."""
    del kwargs
    if getattr(sim, "_m5_hooks_installed", False):
        return sim
    sim._m5_hooks_installed = True
    if not hasattr(sim, "_inflight_tasks"):
        sim._inflight_tasks = {}
    if not hasattr(sim, "_replan_stats") or sim._replan_stats is None:
        sim._replan_stats = empty_replan_stats()

    orig_update_agvs = sim.update_agvs

    def hooked_update_agvs(assigned_task, path, steps, check_ends=None):
        kin = orig_update_agvs(assigned_task, path, steps, check_ends=check_ends)
        if kin is None:
            return None
        payload = dict(assigned_task)
        if "end_points" in payload:
            payload["end_points"] = list(payload["end_points"])
        sim._inflight_tasks[assigned_task["agv"]] = payload
        return kin

    sim.update_agvs = hooked_update_agvs

    orig_update_state = sim.update_agv_state

    def hooked_update_state():
        orig_update_state()
        for agv in sim.agvs:
            tid = agv.task_id
            # Do not drop inflight while the latest step is still loaded — that
            # orphaned carriers at M0→wave handoff (task_carry premature unload).
            cur = None
            try:
                t_now = int(getattr(sim, "time", 0) or 0)
                steps = list(getattr(agv, "steps", None) or [])
                for s in reversed(steps):
                    if int(s.get("timestamp", -1)) <= t_now:
                        cur = s
                        break
            except Exception:  # noqa: BLE001
                cur = None
            still_loaded = bool(
                cur is not None
                and str(cur.get("loaded", "")).lower() in ("true", "1", "yes")
                and str(cur.get("task-id") or "").strip()
                and not str(cur.get("task-id") or "").startswith("escape_")
            )
            if still_loaded:
                continue
            if tid is None or str(tid).startswith("escape_"):
                sim._inflight_tasks.pop(agv.name, None)

    sim.update_agv_state = hooked_update_state
    return sim
