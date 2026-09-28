"""Compare pipeline ECBS vs M0 baseline on identical task set."""
from __future__ import annotations

import csv
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ml_research.benchmarks.common import load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.benchmarks.runner import run_m0

Cell = Tuple[int, int]


def select_tasks_from_states(task_states: dict, max_tasks: int) -> List[Tuple[str, dict]]:
    queues = {n: [dict(t) for t in q] for n, q in task_states.items() if q}
    kept: List[Tuple[str, dict]] = []
    while len(kept) < max_tasks:
        prog = False
        for n in list(queues):
            if queues[n] and len(kept) < max_tasks:
                kept.append((n, queues[n].pop(0)))
                prog = True
        if not prog:
            break
    return kept


def write_task_subset(tasks: List[Tuple[str, dict]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["task_id", "start_point", "end_point", "priority", "remaining_time"]
        )
        for station, t in tasks:
            w.writerow(
                [
                    t["task_id"],
                    station,
                    t["destination"],
                    t.get("priority", "Normal"),
                    t.get("remaining_time") if t.get("remaining_time") is not None else "None",
                ]
            )


def traj_motion_stats(traj_path: Path) -> dict:
    """Per-AGV and fleet motion stats from dense trajectory CSV."""
    by_agv: Dict[str, List[Tuple[int, Cell, int]]] = defaultdict(list)
    max_t = 0
    with open(traj_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            t = int(row["timestamp"])
            c = (int(row["X"]), int(row["Y"]))
            pitch = int(row["pitch"])
            max_t = max(max_t, t)
            by_agv[name].append((t, c, pitch))

    total_move = 0
    total_wait = 0
    total_turn = 0
    per_agv: Dict[str, dict] = {}
    for name, seq in by_agv.items():
        seq.sort(key=lambda x: x[0])
        mv = wt = tr = 0
        for i in range(1, len(seq)):
            _, p0, pi0 = seq[i - 1]
            _, p1, pi1 = seq[i]
            d = abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])
            if d == 1:
                mv += 1
            elif d == 0:
                if pi0 != pi1:
                    tr += 1
                else:
                    wt += 1
        per_agv[name] = {"move": mv, "wait": wt, "turn": tr}
        total_move += mv
        total_wait += wt
        total_turn += tr

    return {
        "sim_time": max_t,
        "total_move_steps": total_move,
        "total_wait_steps": total_wait,
        "total_turn_steps": total_turn,
        "total_motion_steps": total_move + total_wait + total_turn,
        "per_agv": per_agv,
        "n_agvs": len(by_agv),
    }


def main() -> int:
    slot = 3
    max_tasks = 20
    meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=7200.0)
    assert meta

    task_csv = Path(meta["task_csv"])
    pos_csv = Path(meta["position_csv"])
    _, _, _, task_states, n_all = load_scenario(task_csv, pos_csv)
    subset = select_tasks_from_states(task_states, max_tasks)
    task_ids = [t["task_id"] for _, t in subset]
    print(f"[cmp] slot={slot} tasks={len(subset)}/{n_all} ids={task_ids[:5]}…", flush=True)

    cmp_dir = Path(__file__).resolve().parents[2] / "results" / "coord_custom_ai" / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = cmp_dir / f"SH_custom_{slot:02d}_tasks{max_tasks}.csv"
    write_task_subset(subset, sub_csv)

    # --- M0 baseline (same tasks, same map/obstacles) ---
    print("[cmp] running M0 baseline …", flush=True)
    t0 = time.perf_counter()
    rep_m0 = run_m0(
        sub_csv,
        position_csv=pos_csv,
        max_time=int(meta["max_sim_time"]),
        scenario_tag=f"SH_custom_{slot:02d}_m0_t{max_tasks}",
        wall_timeout=7200.0,
        extra_obstacles=meta.get("extra_obstacles"),
    )
    wall_m0 = time.perf_counter() - t0
    traj_m0 = Path(rep_m0["trajectory"])
    val_m0 = validate_hybrid_trajectory(meta, traj_m0)
    mot_m0 = traj_motion_stats(traj_m0)

    # --- Pipeline ECBS ---
    print("[cmp] running pipeline ECBS …", flush=True)
    t0 = time.perf_counter()
    rep_pipe = solve_ecbs(
        slot=slot,
        max_tasks=max_tasks,
        max_active=6,
        weight=1.5,
        plan="joint",
        initial_park=False,
        pipeline=True,
    )
    wall_pipe = time.perf_counter() - t0
    traj_pipe = Path(rep_pipe["trajectory"])
    val_pipe = validate_hybrid_trajectory(meta, traj_pipe)
    mot_pipe = traj_motion_stats(traj_pipe)

    rows = [
        {
            "method": "M0_baseline_greedy",
            "sim_time": rep_m0["sim_time"],
            "tasks_completed": rep_m0["tasks_completed"],
            "tasks_total": rep_m0["tasks_total"],
            "validate_ok": val_m0.get("ok"),
            "validate_summary": format_validation_summary(val_m0),
            "total_move_steps": mot_m0["total_move_steps"],
            "total_wait_steps": mot_m0["total_wait_steps"],
            "total_turn_steps": mot_m0["total_turn_steps"],
            "wall_seconds": round(wall_m0, 1),
            "trajectory": str(traj_m0),
        },
        {
            "method": "pipeline_ecbs_joint",
            "sim_time": rep_pipe["sim_time"],
            "tasks_completed": rep_pipe["tasks_completed"],
            "tasks_total": rep_pipe["tasks_total"],
            "validate_ok": rep_pipe.get("validate_ok"),
            "validate_summary": rep_pipe.get("validate_summary"),
            "total_move_steps": mot_pipe["total_move_steps"],
            "total_wait_steps": mot_pipe["total_wait_steps"],
            "total_turn_steps": mot_pipe["total_turn_steps"],
            "wall_seconds": round(wall_pipe, 1),
            "trajectory": str(traj_pipe),
        },
    ]

    out_json = cmp_dir / f"SH_custom_{slot:02d}_t{max_tasks}_cmp.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SH_custom_03 · 20 tasks · 相同任务集 ===", flush=True)
    print(f"{'方法':<28} {'完成':>8} {'sim_t':>7} {'移动格':>8} {'等待':>8} {'转向':>8} {'validate':>10}", flush=True)
    for r in rows:
        print(
            f"{r['method']:<28} "
            f"{r['tasks_completed']}/{r['tasks_total']:>6} "
            f"{r['sim_time']:>7} "
            f"{r['total_move_steps']:>8} "
            f"{r['total_wait_steps']:>8} "
            f"{r['total_turn_steps']:>8} "
            f"{str(r['validate_ok']):>10}",
            flush=True,
        )

    m0, pipe = rows[0], rows[1]
    if m0["sim_time"] and pipe["sim_time"]:
        dt = m0["sim_time"] - pipe["sim_time"]
        pct = 100.0 * dt / m0["sim_time"]
        print(
            f"\n完成时间: pipeline 比 baseline {'快' if dt > 0 else '慢'} "
            f"{abs(dt)}s ({abs(pct):.1f}%)",
            flush=True,
        )
    dm = m0["total_move_steps"] - pipe["total_move_steps"]
    print(
        f"行进距离(移动格): pipeline vs baseline 差 {dm:+d} 格 "
        f"({100.0*dm/max(1,m0['total_move_steps']):+.1f}%)",
        flush=True,
    )
    print(f"\n[cmp] saved {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
