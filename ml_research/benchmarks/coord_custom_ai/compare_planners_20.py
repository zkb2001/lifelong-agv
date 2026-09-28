"""Fair planner bake-off on SH_custom_01..20 (shape curriculum, not hutong-only).

Methods (industrial / production-adjacent MAPD stack):
  1. M0_spacetime_Astar   — greedy dispatch + spacetime A* reservations
  2. Astar_SwapNet        — same planner + SwapNet station swap gate
  3. M2_priority_PP       — priority planning (WHCA*/PP style) + spacetime A*
  4. Pipeline_ECBS        — windowed joint ECBS (Weighted-CBS), industrial MAPF core

All methods share the same map obstacles and the same fixed task CSV per slot
(seeded subset of the map's task file). Metrics: makespan (sim_time when all
done / forced stop), total move cells, waits, turns, completion ratio.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml_research.benchmarks.allocators import allocator_priority_planning, patch_allocator
from ml_research.benchmarks.common import (
    analyze_trajectory_conflicts,
    load_scenario,
    patch_conflict_free_execution,
    patch_extra_obstacles,
    patch_moving_obstacle_horizon,
    run_sim_loop,
    write_trajectory,
)
from ml_research.benchmarks.coord_custom_ai.compare_pipeline_baseline import (
    select_tasks_from_states,
    traj_motion_stats,
    write_task_subset,
)
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.common.paths import CKPT, RESULTS

OUT = RESULTS / "coord_custom_ai" / "planner_compare_20"
TASKS = OUT / "fixed_tasks"
POS = OUT / "fixed_positions"
TRAJ = OUT / "trajectories"


def _gen_or_load_fixed_position(
    slot: int,
    meta: dict,
    *,
    max_agvs: int,
) -> Path:
    """Keep stations; keep at most ``max_agvs`` AGVs (sorted by name) for fair speed."""
    POS.mkdir(parents=True, exist_ok=True)
    out = POS / f"SH_custom_{slot:02d}_agv{max_agvs}.csv"
    if out.is_file():
        return out
    src = Path(meta["position_csv"])
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(src)
    # Preserve original header order
    with src.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    stations = [r for r in rows if str(r.get("type") or "").strip() != "agv"]
    agvs = [r for r in rows if str(r.get("type") or "").strip() == "agv"]
    agvs.sort(key=lambda r: str(r.get("name") or ""))
    kept = agvs[: max(1, int(max_agvs))]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in stations + kept:
            w.writerow({k: r.get(k, "") for k in header})
    return out


def _gen_or_load_fixed_tasks(
    slot: int,
    meta: dict,
    *,
    n_tasks: int,
    seed: int,
    position_csv: Path,
) -> Path:
    """Deterministic task subset: shuffle station queues with seed, then RR take n."""
    TASKS.mkdir(parents=True, exist_ok=True)
    out = TASKS / f"SH_custom_{slot:02d}_n{n_tasks}_s{seed}.csv"
    if out.is_file():
        return out

    _, _, _, task_states, _ = load_scenario(
        Path(meta["task_csv"]), Path(position_csv)
    )
    # Seeded shuffle within each station queue so different seeds ≠ same head-of-queue
    rng = random.Random(seed + slot * 1009)
    shuffled = {}
    for st, q in task_states.items():
        qq = [dict(t) for t in q]
        rng.shuffle(qq)
        shuffled[st] = qq
    subset = select_tasks_from_states(shuffled, n_tasks)
    write_task_subset(subset, out)
    return out


def _finalize_from_traj(
    *,
    method: str,
    meta: dict,
    traj: Path,
    tasks_total: int,
    tasks_completed: int,
    sim_time: int,
    wall: float,
    notes: str = "",
    extra: Optional[dict] = None,
) -> dict:
    mot = traj_motion_stats(traj) if traj.is_file() else {}
    try:
        val = validate_hybrid_trajectory(meta, traj) if traj.is_file() else {"ok": False}
    except Exception as exc:  # noqa: BLE001
        val = {"ok": False, "error": str(exc)}
    row = {
        "method": method,
        "slot": int(meta["slot"]),
        "scenario_id": meta["id"],
        "tasks_total": int(tasks_total),
        "tasks_completed": int(tasks_completed),
        "completion_ratio": round(tasks_completed / max(1, tasks_total), 4),
        "sim_time": int(sim_time),
        "total_move_steps": int(mot.get("total_move_steps") or 0),
        "total_wait_steps": int(mot.get("total_wait_steps") or 0),
        "total_turn_steps": int(mot.get("total_turn_steps") or 0),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val)
        if "issues" in val or "ok" in val
        else str(val.get("error") or ""),
        "wall_seconds": round(float(wall), 2),
        "trajectory": str(traj),
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def _run_sim_method(
    *,
    method: str,
    task_csv: Path,
    position_csv: Path,
    extra_obstacles: Sequence,
    max_time: int,
    wall_timeout: float,
    traj_path: Path,
    enable_swapnet: bool = False,
    allocator=None,
    pure_engine: bool = False,
) -> dict:
    """Run spacetime-A* engine (optionally SwapNet / PP allocator).

    pure_engine=True matches run_lifelong_baseline / old L100 video path:
    no execution shield, MOVING_OBSTACLE_HORIZON=0, no mid-run trajectory checkpoints.
    """
    # Import swapnet fork once so time_forward gate exists; only fires if flag set.
    if enable_swapnet:
        import simulation.engine_swapnet  # noqa: F401

    mod, env, agv_states, task_states, n_tasks = load_scenario(
        task_csv, position_csv, force_reload=True
    )
    if extra_obstacles:
        patch_extra_obstacles(env, list(extra_obstacles))

    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    if pure_engine:
        patch_moving_obstacle_horizon(sim, horizon=0)
    else:
        patch_moving_obstacle_horizon(sim)
        patch_conflict_free_execution(sim)

    if allocator is not None:
        patch_allocator(sim, allocator)

    if enable_swapnet:
        from ml_research.benchmarks.m5_replan import ensure_m5_hooks
        from ml_research.benchmarks.station_eta import ensure_station_eta_hooks
        from ml_research.benchmarks.swap_net.hooks import attach_swap_net

        ensure_m5_hooks(sim, horizon=40, enable_replan=True, aggressive=False)
        ensure_station_eta_hooks(sim)
        ckpt = CKPT / "swap_net_v3.pt"
        if not ckpt.exists():
            ckpt = CKPT / "swap_net.pt"
        attach_swap_net(
            sim,
            ckpt=ckpt if ckpt.exists() else None,
            threshold_normal=0.48,
            threshold_urgent=0.40,
        )
        sim._swapnet_enabled = True
        if not hasattr(sim, "_replan_stats") or not isinstance(sim._replan_stats, dict):
            sim._replan_stats = {}
        sim._replan_stats.setdefault("station_swap", 0)

    ckpt_every = 0 if pure_engine else 200
    t0 = time.perf_counter()
    steps, forced = run_sim_loop(
        sim,
        max_time=max_time,
        label=method,
        checkpoint_path=traj_path if ckpt_every > 0 else None,
        checkpoint_every=ckpt_every,
        wall_timeout=wall_timeout,
    )
    wall = time.perf_counter() - t0
    write_trajectory(traj_path, steps)
    left = sum(len(v) for v in sim.task_states.values())
    completed = n_tasks - left
    conf = analyze_trajectory_conflicts(steps)
    rs = getattr(sim, "_replan_stats", {}) or {}
    return {
        "sim_time": int(sim.time),
        "tasks_completed": int(completed),
        "tasks_total": int(n_tasks),
        "forced_stop": bool(forced and left > 0),
        "wall_seconds": wall,
        "conflict_free": bool(conf.get("conflict_free")),
        "collisions": int(conf.get("collisions") or 0),
        "swaps": int(conf.get("swaps") or 0),
        "station_swap": int(rs.get("station_swap", 0) or 0),
        "trajectory": traj_path,
    }


def run_slot(
    slot: int,
    *,
    n_tasks: int,
    seed: int,
    max_time: int,
    wall_timeout: float,
    methods: Sequence[str],
    max_active_ecbs: int = 6,
    max_agvs: int = 8,
    wall_ecbs: Optional[float] = None,
) -> List[dict]:
    meta = load_custom_meta(slot, max_sim_time=max_time, wall_timeout=wall_timeout)
    if meta is None:
        return [{"slot": slot, "error": "meta missing"}]

    pos_csv = _gen_or_load_fixed_position(slot, meta, max_agvs=max_agvs)
    task_csv = _gen_or_load_fixed_tasks(
        slot, meta, n_tasks=n_tasks, seed=seed, position_csv=pos_csv
    )
    obs = meta.get("extra_obstacles") or []
    TRAJ.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    # ECBS still reads AGVs from meta position via load_scenario(task, meta pos)
    # unless we pass trimmed pos — patch by writing sidecar and using task_csv only
    # for engines; for ECBS inject trimmed fleet via temporary meta position swap.
    wanted = {m.lower() for m in methods}
    print(
        f"[{meta['id']}] fleet≤{max_agvs} tasks={n_tasks} pos={pos_csv.name}",
        flush=True,
    )

    if "m0" in wanted or "astar" in wanted:
        print(f"[{meta['id']}] M0 spacetime A* …", flush=True)
        traj = TRAJ / f"{meta['id']}_n{n_tasks}_M0.csv"
        t0 = time.perf_counter()
        try:
            rep = _run_sim_method(
                method="M0_spacetime_Astar",
                task_csv=task_csv,
                position_csv=pos_csv,
                extra_obstacles=obs,
                max_time=max_time,
                wall_timeout=wall_timeout,
                traj_path=traj,
                enable_swapnet=False,
                allocator=None,
            )
            rows.append(
                _finalize_from_traj(
                    method="M0_spacetime_Astar",
                    meta=meta,
                    traj=Path(rep["trajectory"]),
                    tasks_total=rep["tasks_total"],
                    tasks_completed=rep["tasks_completed"],
                    sim_time=rep["sim_time"],
                    wall=rep["wall_seconds"],
                    notes="greedy dispatch + spacetime A*",
                    extra={
                        "forced_stop": rep["forced_stop"],
                        "conflict_free": rep["conflict_free"],
                        "collisions": rep["collisions"],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "method": "M0_spacetime_Astar",
                    "slot": slot,
                    "scenario_id": meta["id"],
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    if "swapnet" in wanted or "astar_swapnet" in wanted:
        print(f"[{meta['id']}] A* + SwapNet …", flush=True)
        traj = TRAJ / f"{meta['id']}_n{n_tasks}_SwapNet.csv"
        t0 = time.perf_counter()
        try:
            rep = _run_sim_method(
                method="Astar_SwapNet",
                task_csv=task_csv,
                position_csv=pos_csv,
                extra_obstacles=obs,
                max_time=max_time,
                wall_timeout=wall_timeout,
                traj_path=traj,
                enable_swapnet=True,
                allocator=None,
            )
            rows.append(
                _finalize_from_traj(
                    method="Astar_SwapNet",
                    meta=meta,
                    traj=Path(rep["trajectory"]),
                    tasks_total=rep["tasks_total"],
                    tasks_completed=rep["tasks_completed"],
                    sim_time=rep["sim_time"],
                    wall=rep["wall_seconds"],
                    notes="greedy + spacetime A* + SwapNet swap gate",
                    extra={
                        "forced_stop": rep["forced_stop"],
                        "conflict_free": rep["conflict_free"],
                        "collisions": rep["collisions"],
                        "station_swap": rep.get("station_swap", 0),
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "method": "Astar_SwapNet",
                    "slot": slot,
                    "scenario_id": meta["id"],
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    if "m2" in wanted or "pp" in wanted:
        print(f"[{meta['id']}] M2 priority PP …", flush=True)
        traj = TRAJ / f"{meta['id']}_n{n_tasks}_M2.csv"
        t0 = time.perf_counter()
        try:
            rep = _run_sim_method(
                method="M2_priority_PP",
                task_csv=task_csv,
                position_csv=pos_csv,
                extra_obstacles=obs,
                max_time=max_time,
                wall_timeout=wall_timeout,
                traj_path=traj,
                enable_swapnet=False,
                allocator=allocator_priority_planning,
            )
            rows.append(
                _finalize_from_traj(
                    method="M2_priority_PP",
                    meta=meta,
                    traj=Path(rep["trajectory"]),
                    tasks_total=rep["tasks_total"],
                    tasks_completed=rep["tasks_completed"],
                    sim_time=rep["sim_time"],
                    wall=rep["wall_seconds"],
                    notes="priority planning + spacetime A*",
                    extra={
                        "forced_stop": rep["forced_stop"],
                        "conflict_free": rep["conflict_free"],
                        "collisions": rep["collisions"],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "method": "M2_priority_PP",
                    "slot": slot,
                    "scenario_id": meta["id"],
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    if "ecbs" in wanted or "pipeline_ecbs" in wanted:
        print(f"[{meta['id']}] Pipeline ECBS …", flush=True)
        t0 = time.perf_counter()
        try:
            # Patch the name bound inside solve_ecbs (not only scenes module).
            import ml_research.benchmarks.coord_custom_ai.solve_ecbs as ecbs_mod

            orig_load = ecbs_mod.load_custom_meta

            def _load_trimmed(slot_n, **kw):
                m = orig_load(slot_n, **kw)
                if m is not None and int(slot_n) == int(slot):
                    m = dict(m)
                    m["position_csv"] = str(pos_csv.resolve())
                    m["n_agvs"] = max_agvs
                return m

            ecbs_mod.load_custom_meta = _load_trimmed  # type: ignore[assignment]
            try:
                rep = solve_ecbs(
                    slot=slot,
                    max_tasks=0,
                    max_active=min(max_active_ecbs, max_agvs),
                    weight=1.5,
                    plan="joint",
                    initial_park=False,
                    pipeline=True,
                    task_csv=task_csv,
                )
            finally:
                ecbs_mod.load_custom_meta = orig_load  # type: ignore[assignment]
            traj = Path(str(rep.get("trajectory") or ""))
            rows.append(
                _finalize_from_traj(
                    method="Pipeline_ECBS",
                    meta=meta,
                    traj=traj,
                    tasks_total=int(rep.get("tasks_total") or n_tasks),
                    tasks_completed=int(rep.get("tasks_completed") or 0),
                    sim_time=int(rep.get("sim_time") or 0),
                    wall=float(rep.get("wall_seconds") or (time.perf_counter() - t0)),
                    notes=f"windowed joint ECBS w=1.5 k={max_active_ecbs}",
                    extra={
                        "validate_ok": bool(rep.get("validate_ok")),
                        "validate_summary": rep.get("validate_summary") or "",
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "method": "Pipeline_ECBS",
                    "slot": slot,
                    "scenario_id": meta["id"],
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    return rows


def _write_table(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "slot",
        "scenario_id",
        "method",
        "tasks_completed",
        "tasks_total",
        "completion_ratio",
        "sim_time",
        "total_move_steps",
        "total_wait_steps",
        "total_turn_steps",
        "validate_ok",
        "wall_seconds",
        "station_swap",
        "error",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _aggregate(rows: List[dict]) -> List[dict]:
    by: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("error"):
            continue
        by.setdefault(str(r.get("method")), []).append(r)
    out = []
    for method, rs in sorted(by.items()):
        n = len(rs)
        if not n:
            continue

        def avg(key: str) -> float:
            vals = [float(r.get(key) or 0) for r in rs]
            return round(sum(vals) / n, 2)

        done_all = sum(1 for r in rs if int(r.get("tasks_completed") or 0) >= int(r.get("tasks_total") or 1))
        out.append(
            {
                "method": method,
                "n_maps": n,
                "maps_fully_done": done_all,
                "avg_sim_time": avg("sim_time"),
                "avg_move_cells": avg("total_move_steps"),
                "avg_wait_steps": avg("total_wait_steps"),
                "avg_turn_steps": avg("total_turn_steps"),
                "avg_completion_ratio": avg("completion_ratio"),
                "avg_wall_seconds": avg("wall_seconds"),
            }
        )
    return out


def _print_tables(rows: List[dict], agg: List[dict]) -> None:
    print("\n===== PER-MAP RESULTS =====", flush=True)
    hdr = (
        f"{'slot':>4} {'method':<20} {'done':>8} {'sim_t':>7} "
        f"{'moves':>7} {'waits':>7} {'turns':>7} {'ok':>5} {'wall':>7}"
    )
    print(hdr, flush=True)
    for r in sorted(rows, key=lambda x: (int(x.get("slot") or 0), str(x.get("method")))):
        if r.get("error"):
            print(
                f"{int(r.get('slot') or 0):>4} {str(r.get('method')):<20} ERROR {r['error'][:60]}",
                flush=True,
            )
            continue
        print(
            f"{int(r['slot']):>4} {str(r['method']):<20} "
            f"{r.get('tasks_completed')}/{r.get('tasks_total')} "
            f"{int(r.get('sim_time') or 0):>7} "
            f"{int(r.get('total_move_steps') or 0):>7} "
            f"{int(r.get('total_wait_steps') or 0):>7} "
            f"{int(r.get('total_turn_steps') or 0):>7} "
            f"{str(bool(r.get('validate_ok'))):>5} "
            f"{float(r.get('wall_seconds') or 0):>7.1f}",
            flush=True,
        )

    print("\n===== AGGREGATE (mean over maps) =====", flush=True)
    print(
        f"{'method':<20} {'maps':>4} {'full':>4} {'sim_t':>8} {'moves':>8} "
        f"{'waits':>8} {'ratio':>7} {'wall':>8}",
        flush=True,
    )
    for a in agg:
        print(
            f"{a['method']:<20} {a['n_maps']:>4} {a['maps_fully_done']:>4} "
            f"{a['avg_sim_time']:>8.1f} {a['avg_move_cells']:>8.1f} "
            f"{a['avg_wait_steps']:>8.1f} {a['avg_completion_ratio']:>7.3f} "
            f"{a['avg_wall_seconds']:>8.1f}",
            flush=True,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="20-map planner compare (fixed tasks)")
    ap.add_argument("--slots", type=str, default="1-20", help="e.g. 1-20 or 1,2,4")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-time", type=int, default=2000)
    ap.add_argument("--wall", type=float, default=200.0)
    ap.add_argument(
        "--methods",
        type=str,
        default="m0,swapnet,m2,ecbs",
        help="comma list: m0,swapnet,m2,ecbs",
    )
    ap.add_argument("--max-active-ecbs", type=int, default=6)
    ap.add_argument("--max-agvs", type=int, default=8, help="cap fleet size for fair/tractable runs")
    args = ap.parse_args(argv)

    import os

    # Shorter reservation window → much faster spacetime A* on dense maps
    os.environ.setdefault("MOVING_OBSTACLE_HORIZON", "12")

    slots: List[int] = []
    for part in args.slots.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            slots.extend(range(int(a), int(b) + 1))
        elif part:
            slots.append(int(part))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    # Fresh task/pos caches when fleet size changes
    all_rows: List[dict] = []
    print(
        f"[compare] slots={slots} n_tasks={args.n_tasks} seed={args.seed} "
        f"max_agvs={args.max_agvs} methods={methods} max_time={args.max_time} wall={args.wall}",
        flush=True,
    )

    for slot in slots:
        rows = run_slot(
            slot,
            n_tasks=args.n_tasks,
            seed=args.seed,
            max_time=args.max_time,
            wall_timeout=args.wall,
            methods=methods,
            max_active_ecbs=args.max_active_ecbs,
            max_agvs=args.max_agvs,
        )
        all_rows.extend(rows)
        # incremental save
        _write_table(all_rows, OUT / "per_map.csv")
        (OUT / "per_map.json").write_text(
            json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[compare] slot {slot} done ({len(rows)} methods) → saved per_map.csv",
            flush=True,
        )

    agg = _aggregate(all_rows)
    (OUT / "aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        if agg:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)

    _print_tables(all_rows, agg)
    print(f"\n[compare] results → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
