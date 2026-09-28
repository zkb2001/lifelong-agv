"""Compare M0 engine baseline vs current hierarchical on SH01 demo-400 tasks."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

from ml_research.benchmarks.common import (
    load_scenario,
    patch_conflict_free_execution,
    patch_extra_obstacles,
    patch_moving_obstacle_horizon,
    write_trajectory,
)
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.benchmarks.hier_coord.run_easy_demo_400_video import gen_demo_400_tasks
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import write_task_csv
from ml_research.benchmarks.verify_fifo import analyze_fifo
from ml_research.common.paths import RESULTS

OUT = RESULTS / "hier_coord" / "easy_demo_400"


def run_m0_progress(
    task_csv: Path,
    position_csv: Path,
    *,
    extra_obstacles,
    max_time: int = 500000,
    wall_timeout: float = 1800.0,
    traj_path: Path,
    progress_every: int = 100,
) -> dict:
    mod, env, agv_states, task_states, n_tasks = load_scenario(task_csv, position_csv)
    if extra_obstacles:
        patch_extra_obstacles(env, list(extra_obstacles))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    patch_moving_obstacle_horizon(sim)
    patch_conflict_free_execution(sim)

    t0 = time.perf_counter()
    forced = False
    last_done = -1
    while not sim.all_over():
        with contextlib.redirect_stdout(buf):
            sim.time_forward()
        elapsed = time.perf_counter() - t0
        left = sum(len(v) for v in sim.task_states.values())
        done = n_tasks - left
        if sim.time % progress_every == 0 or done != last_done:
            if done != last_done or sim.time % progress_every == 0:
                print(
                    f"[M0] t={sim.time} done={done}/{n_tasks} "
                    f"surface={len(sim.surface_tasks)} wall={elapsed:.1f}s",
                    flush=True,
                )
                last_done = done
        if sim.time >= max_time:
            forced = True
            break
        if elapsed >= wall_timeout:
            forced = True
            print(f"[M0] wall timeout {wall_timeout}s", flush=True)
            break

    steps = sim.steps_reorganize()
    write_trajectory(traj_path, steps)
    left = sum(len(v) for v in sim.task_states.values())
    completed = n_tasks - left
    return {
        "method": "M0_baseline_greedy",
        "sim_time": int(sim.time),
        "tasks_completed": int(completed),
        "tasks_total": int(n_tasks),
        "completion_ratio": round(completed / max(1, n_tasks), 4),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "forced_stop": bool(forced and left > 0),
        "trajectory": str(traj_path),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    task_csv = OUT / "SH01_demo400_tasks.csv"
    tasks = gen_demo_400_tasks(seed=42)
    write_task_csv(task_csv, tasks)
    print(f"[cmp] tasks={len(tasks)} -> {task_csv}", flush=True)

    meta = load_custom_meta(1, max_sim_time=500000, wall_timeout=1e9)
    assert meta
    extra = meta.get("extra_obstacles") or []
    pos = Path(meta["position_csv"])

    print("[cmp] === M0 baseline (engine) ===", flush=True)
    m0_traj = OUT / "SH01_demo400_m0_traj.csv"
    m0 = run_m0_progress(
        task_csv,
        pos,
        extra_obstacles=extra,
        max_time=500000,
        wall_timeout=1800.0,
        traj_path=m0_traj,
        progress_every=50,
    )
    m0_val = validate_hybrid_trajectory(meta, m0_traj)
    m0_fifo = analyze_fifo(task_csv, m0_traj)
    m0_out = {
        **m0,
        "validate_ok": bool(m0_val.get("ok")),
        "validate_summary": format_validation_summary(m0_val),
        "fifo_ok": bool(m0_fifo.get("ok")),
        "n_fifo": len(m0_fifo.get("fifo_violations") or []),
        "n_display": len(m0_fifo.get("display_mismatch") or []),
    }
    print("[cmp] M0", json.dumps(m0_out, ensure_ascii=False), flush=True)
    (OUT / "cmp_m0_400.json").write_text(
        json.dumps(m0_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[cmp] === hierarchical current ===", flush=True)
    meta2 = dict(meta)
    meta2["id"] = "SH01_easy_demo400_cmp"
    meta2["task_csv"] = str(task_csv.resolve())
    meta2["n_tasks"] = 400
    t1 = time.perf_counter()
    hier = solve_ecbs(
        slot=1,
        meta=meta2,
        max_tasks=0,
        max_active=0,
        weight=1.5,
        time_limit=35.0,
        plan="joint",
        pipeline=True,
        turn_aware=True,
        initial_park=False,
        use_hierarchical=True,
        use_wavenet=True,
        wave_hard_cap=4,
        plan_horizon=24,
        exec_horizon=12,
    )
    hier_out = {
        **{
            k: hier.get(k)
            for k in (
                "sim_time",
                "tasks_completed",
                "tasks_total",
                "completion_ratio",
                "wall_seconds",
                "validate_ok",
                "validate_summary",
                "tasks_failed",
                "trajectory",
            )
        },
        "wall_total": round(time.perf_counter() - t1, 2),
        "hierarchical": hier.get("hierarchical"),
    }
    print(
        "[cmp] HIER",
        json.dumps({k: hier_out[k] for k in hier_out if k != "hierarchical"}, ensure_ascii=False),
        flush=True,
    )
    (OUT / "cmp_hier_400.json").write_text(
        json.dumps(hier_out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\n=== COMPARISON ===", flush=True)
    print(f"{'metric':22s} {'M0':>14s} {'HIER':>14s}", flush=True)
    for k in (
        "sim_time",
        "tasks_completed",
        "completion_ratio",
        "validate_ok",
        "wall_seconds",
    ):
        mv = m0_out.get(k)
        hv = hier_out.get(k) if k != "wall_seconds" else hier_out.get("wall_total")
        if k == "wall_seconds":
            mv = m0_out.get("wall_seconds")
        print(f"{k:22s} {str(mv):>14s} {str(hv):>14s}", flush=True)
    print(
        f"{'fifo_ok':22s} {str(m0_out.get('fifo_ok')):>14s} {'(in validate)':>14s}",
        flush=True,
    )
    if m0_out.get("sim_time") and hier_out.get("sim_time"):
        print(
            f"hier/m0 sim ratio = {hier_out['sim_time'] / max(1, m0_out['sim_time']):.2f}x",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
