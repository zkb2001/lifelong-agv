"""Reproduce phantom load: loaded=TRUE before visiting pickup_point."""
from __future__ import annotations

import contextlib
import io
import os

os.environ.pop("AGV_GPU_BFS", None)

from ml_research.benchmarks.allocators import make_complete_allocator, patch_allocator
from ml_research.benchmarks.common import (
    get_mod,
    patch_conflict_free_execution,
    patch_moving_obstacle_horizon,
    set_pair_cost_mode,
)
from ml_research.benchmarks.lifelong.task_generator import (
    LifelongTaskGenerator,
    install_lifelong_generator,
)
from ml_research.benchmarks.m5_replan import ensure_m5_hooks, m5_replan_enabled
from ml_research.benchmarks.path_fallback import patch_path_planning_fallback
from ml_research.common.paths import POSITION_CSV


def main() -> int:
    set_pair_cost_mode("astar")
    mod = get_mod(force_reload=True)
    start_points, end_points, agv_list = mod.get_object_position(str(POSITION_CSV))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = {name: [] for name in start_points}

    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )

    patch_conflict_free_execution(sim)
    patch_moving_obstacle_horizon(sim)
    patch_path_planning_fallback(sim)
    allocator = make_complete_allocator(
        horizon=40, enable_replan=m5_replan_enabled(True), aggressive=False, station_eta=False
    )
    patch_allocator(sim, allocator)
    ensure_m5_hooks(sim, horizon=40, enable_replan=True, aggressive=False)

    gen = LifelongTaskGenerator(
        pickups=list(start_points.keys()),
        dropoffs=list(end_points.keys()),
        start_points=start_points,
        end_points=end_points,
        mod=mod,
        seed=42,
        min_backlog=12,
        max_backlog=28,
        inject_every=3,
        urgent_prob=0.12,
        catalog_csv=None,
    )
    gen.seed(sim)
    install_lifelong_generator(sim, gen)

    phantoms = []
    bad_unloads = []

    orig_update = sim.update_agvs

    def audited_update(assigned_task, path, steps):
        pickup = tuple(assigned_task["pickup_point"][:2])
        tid = assigned_task["task_id"]
        # Inspect planned steps before stitch
        seen_pickup = False
        plan_issues = []
        for s in steps:
            cell = (int(s["X"]), int(s["Y"]))
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            stid = str(s.get("task-id") or "")
            if cell == pickup:
                seen_pickup = True
            if loaded and stid == tid and not seen_pickup and cell != pickup:
                plan_issues.append(
                    {
                        "phase": "pre_stitch_plan",
                        "t": int(s["timestamp"]),
                        "cell": cell,
                        "pickup": pickup,
                        "tid": tid,
                        "agv": assigned_task["agv"],
                        "sim_t": int(sim.time),
                    }
                )
                break
        if plan_issues:
            phantoms.extend(plan_issues)
            print("PLAN_PHANTOM", plan_issues[0], flush=True)

        # path length / last_t before stitch
        agv = next(a for a in sim.agvs if a.name == assigned_task["agv"])
        last_t = agv.path[-1][2] if agv.path else None
        n_before = len(steps)
        kept = [s for s in steps if last_t is None or s["timestamp"] > last_t]
        seen_pickup = False
        for s in kept:
            cell = (int(s["X"]), int(s["Y"]))
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            stid = str(s.get("task-id") or "")
            if cell == pickup:
                seen_pickup = True
            if loaded and stid == tid and not seen_pickup and cell != pickup:
                phantoms.append(
                    {
                        "phase": "after_filter_by_last_t",
                        "t": int(s["timestamp"]),
                        "cell": cell,
                        "pickup": pickup,
                        "tid": tid,
                        "agv": assigned_task["agv"],
                        "sim_t": int(sim.time),
                        "last_t": last_t,
                        "n_steps": n_before,
                        "n_kept": len(kept),
                    }
                )
                print("FILTER_PHANTOM", phantoms[-1], flush=True)
                break

        return orig_update(assigned_task, path, steps)

    sim.update_agvs = audited_update

    # Also scan executed steps each tick
    for _ in range(500):
        sim.time_forward()
        for agv in sim.agvs:
            t = int(sim.time)
            cur = None
            prev = None
            for s in agv.steps:
                ts = int(s["timestamp"])
                if ts == t:
                    cur = s
                if ts == t - 1:
                    prev = s
            if not cur or not prev:
                continue
            pl = str(prev.get("loaded", "")).lower() in ("true", "1", "yes")
            cl = str(cur.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(cur.get("task-id") or "").strip()
            if cl and not pl and tid:
                info = (getattr(sim, "_inflight_tasks", {}) or {}).get(agv.name) or {}
                pickup = info.get("pickup_point")
                if pickup:
                    pickup = tuple(pickup[:2])
                else:
                    # from task_states / catalog via destination
                    pickup = None
                    for st, q in (sim.task_states or {}).items():
                        for item in q:
                            if item.get("task_id") == tid:
                                pickup = tuple(item["pickup_point"][:2])
                    if pickup is None:
                        for st, q in start_points.items():
                            pass
                cell = (int(cur["X"]), int(cur["Y"]))
                if pickup and cell != pickup:
                    phantoms.append(
                        {
                            "phase": "executed",
                            "t": t,
                            "agv": agv.name,
                            "tid": tid,
                            "cell": cell,
                            "pickup": pickup,
                        }
                    )
                    print("EXEC_PHANTOM", phantoms[-1], flush=True)

            # bad unload
            if pl and not cl:
                ptid = str(prev.get("task-id") or "").strip()
                if not ptid:
                    continue
                drops = set()
                info = (getattr(sim, "_inflight_tasks", {}) or {}).get(agv.name) or {}
                for p in info.get("end_points") or []:
                    drops.add(tuple(p[:2]))
                # also from destination name
                dest = prev.get("destination") or info.get("destination")
                if dest and dest in end_points:
                    _, pts = mod.get_end_points(dest, end_points)
                    drops |= set(pts)
                cell = (int(cur["X"]), int(cur["Y"]))
                if drops and cell not in drops:
                    bad_unloads.append(
                        {"t": t, "agv": agv.name, "tid": ptid, "cell": cell, "drops": list(drops)[:4]}
                    )
                    print("BAD_UNLOAD", bad_unloads[-1], flush=True)

    print("=== SUMMARY ===")
    print("phantoms", len(phantoms))
    print("bad_unloads", len(bad_unloads))
    for p in phantoms[:20]:
        print(p)
    for u in bad_unloads[:20]:
        print(u)
    return 0 if not phantoms and not bad_unloads else 1


if __name__ == "__main__":
    raise SystemExit(main())
