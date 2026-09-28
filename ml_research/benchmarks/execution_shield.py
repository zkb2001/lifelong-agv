"""Execution-time vertex/swap repair for spacetime-A* baseline exports.

The planner records full paths ahead of time; with a rolling moving-obstacle
horizon, distant segments can still overlap. This shield only overwrites the
current tick (hold / nudge) — it does not shift future waypoints (avoids livelock).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple


def _grid_hi(sim) -> int:
    mod = getattr(sim, "__class__", None)
    del mod
    try:
        from ml_research.benchmarks.common import get_mod

        m = get_mod()
        gs = getattr(m, "DEFAULT_GRID_SIZE", (21, 21))
        return int(gs[0])
    except Exception:
        return 21


def _sync_moving_obstacles_from_path(sim, agv) -> None:
    sim.env.moving_obstacles[agv.name] = list(agv.path)


def _set_pose_at_t(sim, agv, t: int, x: int, y: int, pitch: int) -> None:
    """Overwrite path/steps/obstacles at time t only."""
    pose = (int(x), int(y), int(t), int(pitch))
    if t < len(agv.path):
        agv.path[t] = pose
    else:
        while len(agv.path) < t:
            last = agv.path[-1]
            agv.path.append((last[0], last[1], len(agv.path), last[3]))
        if len(agv.path) == t:
            agv.path.append(pose)
        else:
            agv.path[t] = pose

    for s in agv.steps:
        if int(s["timestamp"]) == t:
            s["X"], s["Y"], s["pitch"] = int(x), int(y), int(pitch)
            break
    else:
        loaded = "FALSE"
        dest = ""
        emerg = "FALSE"
        tid = ""
        for s in reversed(agv.steps):
            if int(s.get("timestamp", -1)) <= t:
                loaded = s.get("loaded", "FALSE")
                dest = s.get("destination", "")
                emerg = s.get("Emergency", "FALSE")
                tid = str(s.get("task-id") or "")
                break
        agv.steps.append(
            {
                "timestamp": t,
                "name": agv.name,
                "X": int(x),
                "Y": int(y),
                "pitch": int(pitch),
                "loaded": loaded,
                "destination": dest,
                "Emergency": emerg,
                "task-id": tid,
            }
        )
    agv.state = pose
    _sync_moving_obstacles_from_path(sim, agv)


def _one_step_toward(
    src: Tuple[int, int], dst: Tuple[int, int], pitch: int
) -> Tuple[int, int, int]:
    if src == dst:
        return src[0], src[1], pitch
    x, y = src
    tx, ty = dst
    if x < tx:
        return x + 1, y, 0
    if x > tx:
        return x - 1, y, 180
    if y < ty:
        return x, y + 1, 90
    if y > ty:
        return x, y - 1, 270
    return x, y, pitch


def patch_conflict_free_execution(sim) -> None:
    """Monkey-patch update_agv_state: repair vertex/swap at current sim.time."""
    if getattr(sim, "_conflict_shield_patched", False):
        return
    sim._conflict_shield_patched = True

    from ml_research.benchmarks.common import get_mod

    mod = get_mod()
    orig_update = sim.update_agv_state
    hi = _grid_hi(sim)

    def _pos_at_traj_ts(trajectory, t):
        if not trajectory:
            return None
        t = int(t)
        for p in trajectory:
            if int(p[2]) == t:
                return p[:2]
        if t < int(trajectory[0][2]):
            return trajectory[0][:2]
        if t > int(trajectory[-1][2]):
            return trajectory[-1][:2]
        last = trajectory[0][:2]
        for p in trajectory:
            if int(p[2]) <= t:
                last = p[:2]
            else:
                break
        return last

    mod._pos_at_traj = _pos_at_traj_ts

    def _fix_premature_unload_step(agv, t: int) -> None:
        """Shield hold can leave a planned unload step off the dropoff ring."""
        prev_s = cur_s = None
        for s in agv.steps:
            ts = int(s.get("timestamp", -1))
            if ts == t - 1:
                prev_s = s
            elif ts == t:
                cur_s = s
        if not prev_s or not cur_s:
            return
        loaded_prev = str(prev_s.get("loaded", "")).lower() in ("true", "1", "yes")
        loaded_cur = str(cur_s.get("loaded", "")).lower() in ("true", "1", "yes")
        tid_cur = str(cur_s.get("task-id") or "").strip()
        tid_prev = str(prev_s.get("task-id") or "").strip()
        if not loaded_prev or loaded_cur or tid_cur:
            return
        cell = (int(cur_s["X"]), int(cur_s["Y"]))
        tid = tid_prev or str(getattr(agv, "task_id", "") or "").strip()
        # Task-specific pad only — foreign hubs (Shanghai while dest=Beijing) are NOT ok.
        if tid and hasattr(sim, "_is_unload_interaction_cell"):
            if sim._is_unload_interaction_cell(cell, tid=tid):
                return
        elif sim._is_unload_interaction_cell(cell):
            return
        cur_s["loaded"] = prev_s.get("loaded", "TRUE")
        cur_s["task-id"] = prev_s.get("task-id", "")
        cur_s["destination"] = prev_s.get("destination", "")
        cur_s["Emergency"] = prev_s.get("Emergency", "FALSE")

    def _pickup_point_for(agv):
        tid = getattr(agv, "task_id", None)
        if not tid or str(tid).startswith("escape_"):
            return None
        for store in (
            getattr(sim, "surface_tasks", None),
            getattr(sim, "assigned_task_info", None),
            getattr(sim, "_inflight_tasks", None),
        ):
            if not store:
                continue
            info = store.get(tid) if tid in store else None
            if info is None and isinstance(store, dict):
                info = store.get(agv.name)
            if isinstance(info, dict):
                pk = info.get("pickup_point")
                if pk is not None and len(pk) >= 2:
                    return (int(pk[0]), int(pk[1]))
        return None

    def _restore_pickup_pose(agv, t: int) -> None:
        """If shield nudged off the pickup pad on the rising-edge tick, snap back.

        Only snap when Manhattan-adjacent (legal one-step). Far rising-edges are
        delayed (clear loaded) — teleporting onto the pad created illegal_motion.
        """
        pk = _pickup_point_for(agv)
        if pk is None or not agv.path or t >= len(agv.path):
            return
        cur_s = None
        prev_s = None
        for s in agv.steps:
            ts = int(s.get("timestamp", -1))
            if ts == t:
                cur_s = s
            elif ts == t - 1:
                prev_s = s
        if not cur_s:
            return
        loaded_cur = str(cur_s.get("loaded", "")).lower() in ("true", "1", "yes")
        loaded_prev = (
            str(prev_s.get("loaded", "")).lower() in ("true", "1", "yes") if prev_s else False
        )
        cell = (int(cur_s["X"]), int(cur_s["Y"]))
        if cell == pk:
            return
        if loaded_cur and not loaded_prev:
            manh_cur = abs(cell[0] - pk[0]) + abs(cell[1] - pk[1])
            # Snap ONLY when the current cell is already adjacent to the pad.
            # (manh_prev==1 alone caused teleports after conflict nudges.)
            if manh_cur == 1:
                # Do not snap onto an occupied pickup pad (causes vertex collision).
                occupied = False
                for other in sim.agvs:
                    if other.name == agv.name or not other.path:
                        continue
                    if t < len(other.path):
                        ox, oy = int(other.path[t][0]), int(other.path[t][1])
                    else:
                        ox, oy = int(other.path[-1][0]), int(other.path[-1][1])
                    if (ox, oy) == pk:
                        occupied = True
                        break
                if occupied:
                    cur_s["loaded"] = "FALSE"
                    cur_s["task-id"] = ""
                    cur_s["destination"] = ""
                    cur_s["Emergency"] = "FALSE"
                    return
                pit = int(agv.path[t][3]) if t < len(agv.path) else int(cur_s.get("pitch") or 0)
                _set_pose_at_t(sim, agv, t, pk[0], pk[1], pit)
                for s in agv.steps:
                    if int(s.get("timestamp", -1)) == t:
                        s["loaded"] = "TRUE"
                        tid = str(getattr(agv, "task_id", "") or "") or str(
                            (prev_s or {}).get("task-id") or ""
                        )
                        if tid:
                            s["task-id"] = tid
                        dest = str((prev_s or {}).get("destination") or "")
                        if not dest:
                            info = None
                            for store in (
                                getattr(sim, "assigned_task_info", None),
                                getattr(sim, "surface_tasks", None),
                                getattr(sim, "_inflight_tasks", None),
                            ):
                                if not store:
                                    continue
                                info = store.get(tid) or store.get(agv.name)
                                if info:
                                    break
                            if isinstance(info, dict):
                                dest = str(info.get("destination") or "")
                        if dest:
                            s["destination"] = dest
                        break
                return
            # Far from pad: delay pickup metadata (do not teleport).
            cur_s["loaded"] = "FALSE"
            cur_s["task-id"] = ""
            cur_s["destination"] = ""
            cur_s["Emergency"] = "FALSE"
            return
        planned = agv.path[t] if t < len(agv.path) else None
        if (
            planned
            and (int(planned[0]), int(planned[1])) == pk
            and cell != pk
            and loaded_cur
        ):
            manh = abs(cell[0] - pk[0]) + abs(cell[1] - pk[1])
            if manh == 1:
                _set_pose_at_t(sim, agv, t, pk[0], pk[1], int(planned[3]))

    def safe_update_agv_state():
        orig_update()
        t = int(sim.time)
        pos: Dict[str, Tuple[int, int]] = {}
        pitch: Dict[str, int] = {}
        prev: Dict[str, Tuple[int, int]] = {}
        for agv in sim.agvs:
            if not agv.path:
                continue
            p = agv.path[t] if t < len(agv.path) else agv.path[-1]
            pos[agv.name] = (int(p[0]), int(p[1]))
            pitch[agv.name] = int(p[3])
            if t > 0 and (t - 1) < len(agv.path):
                prev[agv.name] = agv.path[t - 1][:2]
            elif agv.path:
                prev[agv.name] = agv.path[min(len(agv.path) - 1, t - 1)][:2]
            else:
                prev[agv.name] = pos[agv.name]

        static_obs = set(sim.env.get_static_obstacles())

        owners: Dict[Tuple[int, int], list] = defaultdict(list)
        for name, cell in pos.items():
            owners[cell].append(name)

        for cell, names in list(owners.items()):
            if len(names) <= 1:
                continue
            keep = None
            # Prefer the agent whose pickup/dropoff pad is this cell.
            for n in names:
                agv = next(a for a in sim.agvs if a.name == n)
                pk = _pickup_point_for(agv)
                if pk is not None and pk == cell:
                    keep = n
                    break
            if keep is None:
                for n in names:
                    if prev.get(n) == cell:
                        keep = n
                        break
            if keep is None:

                def _busy_key(n):
                    agv = next(a for a in sim.agvs if a.name == n)
                    tid = agv.task_id
                    has_real = bool(tid) and not str(tid).startswith("escape_")
                    return (0 if has_real else 1, n)

                keep = sorted(names, key=_busy_key)[0]
            occupied = {pos[n] for n in pos if n == keep or n not in names}
            occupied.add(cell)
            for n in names:
                if n == keep:
                    continue
                hold = prev.get(n, cell)
                new_xy = None
                new_pitch = pitch[n]
                if hold != cell and hold not in occupied and hold not in static_obs:
                    new_xy = hold
                else:
                    for dx, dy, pit in ((1, 0, 0), (-1, 0, 180), (0, 1, 90), (0, -1, 270)):
                        cand = (cell[0] + dx, cell[1] + dy)
                        if not (1 <= cand[0] <= hi and 1 <= cand[1] <= hi):
                            continue
                        if cand in static_obs or cand in occupied:
                            continue
                        new_xy, new_pitch = cand, pit
                        break
                if new_xy is None:
                    new_xy = hold
                occupied.add(new_xy)
                pos[n] = new_xy
                pitch[n] = new_pitch
                agv = next(a for a in sim.agvs if a.name == n)
                _set_pose_at_t(sim, agv, t, new_xy[0], new_xy[1], new_pitch)

        names = list(pos.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = prev.get(a), prev.get(b)
                if pa is None or pb is None:
                    continue
                # Edge swap: both must hold. Holding only the loser lands both on
                # the same cell (winner already occupies loser's previous cell).
                if pos[a] == pb and pos[b] == pa and pos[a] != pos[b]:
                    pos[a] = pa
                    pos[b] = pb
                    agv_a = next(x for x in sim.agvs if x.name == a)
                    agv_b = next(x for x in sim.agvs if x.name == b)
                    _set_pose_at_t(sim, agv_a, t, pa[0], pa[1], pitch[a])
                    _set_pose_at_t(sim, agv_b, t, pb[0], pb[1], pitch[b])

        # Pitch-only continuity: competition forbids move+turn in one tick.
        # Strafe: keep previous pitch while moving one cell.
        if t > 0:
            for agv in sim.agvs:
                if not agv.path or t >= len(agv.path) or (t - 1) >= len(agv.path):
                    continue
                px, py = int(agv.path[t - 1][0]), int(agv.path[t - 1][1])
                pp = int(agv.path[t - 1][3]) % 360
                cx, cy = int(agv.path[t][0]), int(agv.path[t][1])
                cp = int(agv.path[t][3]) % 360
                if abs(px - cx) + abs(py - cy) == 1 and pp != cp:
                    _set_pose_at_t(sim, agv, t, cx, cy, pp)

        for agv in sim.agvs:
            _fix_premature_unload_step(agv, t)

        # Unit-step before pickup snap so we measure adjacency from the
        # post-clamp previous cell → pad (avoids undoing the snap afterward).
        if t > 0 and hasattr(sim, "_enforce_unit_step_at"):
            for agv in sim.agvs:
                sim._enforce_unit_step_at(agv, t)

        for agv in sim.agvs:
            _restore_pickup_pose(agv, t)

        # Second vertex pass: swap dual-hold / restore can recreate overlaps.
        pos2: Dict[str, Tuple[int, int]] = {}
        for agv in sim.agvs:
            if not agv.path or t >= len(agv.path):
                continue
            p = agv.path[t]
            pos2[agv.name] = (int(p[0]), int(p[1]))
        owners2: Dict[Tuple[int, int], list] = defaultdict(list)
        for name, cell in pos2.items():
            owners2[cell].append(name)
        for cell, names in list(owners2.items()):
            if len(names) <= 1:
                continue
            keep = names[0]
            for n in names:
                if prev.get(n) == cell:
                    keep = n
                    break
            occupied = {pos2[n] for n in pos2 if n == keep or n not in names}
            occupied.add(cell)
            for n in names:
                if n == keep:
                    continue
                hold = prev.get(n, cell)
                # Never leave the loser on the contested cell (that kept SH01
                # Megatron/Sideswipe both at (13,10)).
                if hold == cell or hold in occupied or hold in static_obs:
                    hold = None
                    for dx, dy, pit in ((1, 0, 0), (-1, 0, 180), (0, 1, 90), (0, -1, 270)):
                        cand = (cell[0] + dx, cell[1] + dy)
                        if not (1 <= cand[0] <= hi and 1 <= cand[1] <= hi):
                            continue
                        if cand in static_obs or cand in occupied:
                            continue
                        hold = cand
                        pitch[n] = pit
                        break
                    if hold is None:
                        # Dual-hold: revert loser to previous pose if possible.
                        hold = prev.get(n, cell)
                        if hold == cell or hold in static_obs:
                            # Last resort: stay put but prefer any free neighbor
                            # of prev; else keep prev even if overlapping briefly
                            # is avoided by forcing keep to prev too below.
                            hold = prev.get(n, cell)
                occupied.add(hold)
                pos2[n] = hold
                agv = next(a for a in sim.agvs if a.name == n)
                _set_pose_at_t(sim, agv, t, hold[0], hold[1], pitch.get(n, 0))
            # If loser could not leave cell, also freeze the keeper on prev.
            for n in names:
                if pos2.get(n) == cell and n != keep:
                    for m in names:
                        pm = prev.get(m)
                        if pm is not None and pm not in static_obs:
                            pos2[m] = pm
                            agv = next(a for a in sim.agvs if a.name == m)
                            _set_pose_at_t(sim, agv, t, pm[0], pm[1], pitch.get(m, 0))
                    break
        if t > 0 and hasattr(sim, "_enforce_unit_step_at"):
            for agv in sim.agvs:
                sim._enforce_unit_step_at(agv, t)
        # Pickup snap must win over the second unit-step pass when adjacent.
        for agv in sim.agvs:
            _restore_pickup_pose(agv, t)

        # Final strafe pass: any remaining move+turn → keep move, restore prev pitch.
        if t > 0:
            for agv in sim.agvs:
                if not agv.path or t >= len(agv.path) or (t - 1) >= len(agv.path):
                    continue
                px, py = int(agv.path[t - 1][0]), int(agv.path[t - 1][1])
                pp = int(agv.path[t - 1][3]) % 360
                cx, cy = int(agv.path[t][0]), int(agv.path[t][1])
                cp = int(agv.path[t][3]) % 360
                if abs(px - cx) + abs(py - cy) == 1 and pp != cp:
                    _set_pose_at_t(sim, agv, t, cx, cy, pp)

    sim.update_agv_state = safe_update_agv_state


def finalize_task_carry(sim, *args, **kwargs):
    return None


def _sanitize_appended_path(path):
    return path
