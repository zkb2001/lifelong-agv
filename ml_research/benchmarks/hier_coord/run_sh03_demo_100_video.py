"""SH03 × 100 tasks demo: M0 + SwapNet + RulePark → optional MP4."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import write_task_csv
from ml_research.benchmarks.scenarios.generator import ALL_DESTS, ALL_STATIONS
from ml_research.common.paths import RESULTS
from ml_research.tools.render_video import render_mp4

OUT = RESULTS / "hier_coord" / "sh03_demo_100"
VIDEO_DIR = OUT / "videos"


def gen_demo_100_tasks(*, seed: int = 42) -> List[dict]:
    """100 tasks with mild hub windows (scaled from easy_demo_400)."""
    rng = random.Random(seed)
    counters: Dict[str, int] = {}
    rows: List[dict] = []

    def tid(st: str) -> str:
        counters[st] = counters.get(st, 0) + 1
        return f"{st}-{counters[st]}"

    def add(st: str, dest: str, phase: str) -> None:
        rows.append(
            {
                "task_id": tid(st),
                "start_point": st,
                "end_point": dest,
                "priority": "Normal",
                "remaining_time": "None",
                "_phase": phase,
            }
        )

    # 1–10 random
    for _ in range(10):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_1_10")
    # 11–25 → Beijing
    for _ in range(15):
        add(rng.choice(ALL_STATIONS), "Beijing", "hub_beijing_11_25")
    # 26–40 random
    for _ in range(15):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_26_40")
    # 41–80 → Tianjin
    for _ in range(40):
        add(rng.choice(ALL_STATIONS), "Tianjin", "hub_tianjin_41_80")
    # 81–100 random
    for _ in range(20):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_81_100")

    assert len(rows) == 100, len(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="mp4 length seconds; default = sim_time/4",
    )
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--wave-hard-cap", type=int, default=4)
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument(
        "--m0-rulepark",
        action="store_true",
        help="Old path: M0+RulePark (default is baseline+ECBS wave).",
    )
    ap.add_argument(
        "--allow-mode-switch",
        action="store_true",
        help="A*/M0 first, degrade→ECBS (no force_hard). Implies legacy wave path.",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_custom_meta(
        int(args.slot),
        max_sim_time=500000,
        wall_timeout=float(args.wall_timeout) if float(args.wall_timeout) > 0 else 1e9,
        n_tasks=100,
    )
    if not meta:
        raise SystemExit(f"slot {args.slot} missing")

    task_csv = OUT / f"SH{args.slot:02d}_demo100_tasks.csv"
    tasks = gen_demo_100_tasks(seed=int(args.seed))
    write_task_csv(task_csv, tasks)
    phase_map = {str(t["task_id"]): str(t.get("_phase") or "") for t in tasks}
    print(f"[demo] SH{args.slot:02d} wrote {task_csv} ({len(tasks)} tasks)", flush=True)

    switchable = bool(args.allow_mode_switch)
    use_ecbs_wave = (not bool(args.m0_rulepark)) or switchable
    meta = dict(meta)
    if switchable:
        meta_id = f"SH{args.slot:02d}_demo100_switch"
        tag = "switch"
        stack = "switchable_m0_ecbs"
    elif use_ecbs_wave and not args.m0_rulepark:
        meta_id = f"SH{args.slot:02d}_demo100_ecbs_wave"
        tag = "ecbs_wave"
        stack = "baseline_ecbs_wave"
    else:
        meta_id = f"SH{args.slot:02d}_demo100"
        tag = "m0"
        stack = "m0_rulepark"
        use_ecbs_wave = False
    meta["id"] = meta_id
    meta["task_csv"] = str(task_csv.resolve())
    meta["n_tasks"] = 100
    meta["wall_timeout"] = float(args.wall_timeout)
    meta["progress_every"] = int(args.progress_every)
    meta["allow_mode_switch"] = bool(switchable)
    meta["legacy_wave_ecbs"] = bool(use_ecbs_wave or switchable)
    meta["traffic_recovery"] = not (use_ecbs_wave or switchable)
    meta["gate_every"] = max(25, int(args.progress_every) // 2)
    meta["m0_assign_fail_handoff"] = 16
    if switchable:
        meta["serial_recover_budget"] = 6
        meta["handback_calm"] = 5
        meta["handback_dwell"] = 200
        meta["m0_handback"] = True
        meta["max_handbacks"] = 2
        meta["m0_min_handoff_t"] = 150
        meta["m0_throughput_drop_streak"] = 3
        # Handback-only A* room (cold M0 stays at engine default ~0.35s).
        meta["m0_handback_astar_wall_s"] = 2.0
        # Field default: conflict shield OFF.

    t0 = time.perf_counter()
    if switchable:
        print(
            f"[demo] solve switchable M0<->ECBS slot={args.slot} tasks=100 "
            f"force_hard=0 allow_mode_switch=1 hard_cap={args.wave_hard_cap}",
            flush=True,
        )
        rep = solve_ecbs(
            slot=int(args.slot),
            meta=meta,
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
            use_swapnet=True,
            hierarchical_force=False,
            wave_hard_cap=int(args.wave_hard_cap),
            use_traffic_accel=False,
            use_lane_rules=False,
            task_phase_by_id=phase_map,
            plan_horizon=24,
            exec_horizon=12,
            allow_mode_switch=True,
            joint_core="ecbs",
            legacy_wave_ecbs=True,
            traffic_recovery=False,
        )
    elif use_ecbs_wave:
        print(
            f"[demo] solve baseline+ECBS-wave slot={args.slot} tasks=100 "
            f"joint_core=ecbs hard_cap={args.wave_hard_cap} force_hard=1",
            flush=True,
        )
        rep = solve_ecbs(
            slot=int(args.slot),
            meta=meta,
            max_tasks=0,
            max_active=0,
            weight=1.5,
            time_limit=35.0,
            plan="joint",
            pipeline=True,
            turn_aware=True,
            initial_park=False,
            use_hierarchical=True,
            use_wavenet=False,
            use_swapnet=False,
            hierarchical_force=True,
            wave_hard_cap=int(args.wave_hard_cap),
            use_traffic_accel=False,
            use_lane_rules=False,
            task_phase_by_id=phase_map,
            plan_horizon=24,
            exec_horizon=12,
            allow_mode_switch=False,
            joint_core="ecbs",
            legacy_wave_ecbs=True,
            traffic_recovery=False,
        )
    else:
        print(
            f"[demo] solve M0+SwapNet+RulePark slot={args.slot} tasks=100",
            flush=True,
        )
        rep = solve_ecbs(
            slot=int(args.slot),
            meta=meta,
            max_tasks=0,
            max_active=0,
            weight=1.5,
            time_limit=35.0,
            plan="joint",
            pipeline=True,
            turn_aware=True,
            initial_park=False,
            use_hierarchical=False,
            use_wavenet=False,
            use_swapnet=True,
            hierarchical_force=False,
            wave_hard_cap=4,
            use_traffic_accel=False,
            use_lane_rules=False,
            task_phase_by_id=phase_map,
            plan_horizon=24,
            exec_horizon=12,
            allow_mode_switch=False,
            joint_core="prioritized",
            legacy_wave_ecbs=False,
            traffic_recovery=True,
        )
    wall = round(time.perf_counter() - t0, 2)
    hier = rep.get("hierarchical") or {}
    rep_out = {
        k: rep.get(k)
        for k in (
            "sim_time",
            "completion_ratio",
            "validate_ok",
            "tasks_completed",
            "tasks_total",
            "tasks_failed",
            "wall_seconds",
            "trajectory",
            "validate_summary",
            "n_hard_wall",
            "n_collisions",
            "n_swaps",
            "n_illegal_motion",
            "n_fifo",
            "n_pickup_cell",
            "n_task_carry",
            "n_premature_unload",
            "n_foreign_unload",
            "n_display_mismatch",
            "method",
        )
    }
    rep_out.update(
        {
            "demo_wall": wall,
            "slot": args.slot,
            "task_csv": str(task_csv),
            "stack": stack,
            "hierarchical": {
                "last_reason": hier.get("last_reason"),
                "last_planner": hier.get("last_planner"),
                "waves_astar": hier.get("waves_astar"),
                "waves_ecbs": hier.get("waves_ecbs"),
                "waves_hybrid": hier.get("waves_hybrid"),
                "waves_escalated": hier.get("waves_escalated"),
                "m0_handoff": hier.get("m0_handoff"),
                "m0_handoff_t": hier.get("m0_handoff_t"),
                "m0_handback": hier.get("m0_handback"),
                "allow_mode_switch": hier.get("allow_mode_switch"),
                "hardness": hier.get("hardness"),
                "traffic_stats": hier.get("traffic_stats"),
            },
            "traffic_recovery": rep.get("traffic_recovery"),
        }
    )
    summary = OUT / f"SH{args.slot:02d}_demo100_{tag}_summary.json"
    summary.write_text(json.dumps(rep_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep_out, indent=2, ensure_ascii=False), flush=True)

    done = int(rep.get("tasks_completed") or 0)
    ratio = float(rep.get("completion_ratio") or 0)
    traj = Path(rep["trajectory"]) if rep.get("trajectory") else None
    if traj is None or not traj.exists():
        print("[demo] no trajectory — fail", flush=True)
        return 1

    # Always try video on partial too (SH03 may not finish 100).
    if args.skip_video:
        print(f"[demo] skip video done={done}/100 ratio={ratio:.3f}", flush=True)
        return 0 if ratio >= 0.999 else 1

    sim_t = int(rep.get("sim_time") or 0)
    if args.duration is not None and float(args.duration) > 0:
        duration_s = float(args.duration)
    else:
        duration_s = max(1.0, float(sim_t) / 4.0)

    mp4 = VIDEO_DIR / f"SH{args.slot:02d}_demo100_{tag}_{int(round(duration_s))}s.mp4"
    print(
        f"[demo] render mp4 duration≈{duration_s:.1f}s (sim={sim_t}, sim/4) "
        f"done={done}/100 → {mp4}",
        flush=True,
    )
    rc = render_mp4(
        traj,
        mp4,
        task_csv=task_csv,
        position_csv=Path(meta["position_csv"]),
        obstacles_json=Path(meta["map_json"]) if meta.get("map_json") else None,
        speed=1.0,
        duration=float(duration_s),
    )
    if rc != 0 or not mp4.exists():
        print(f"[demo] video render failed rc={rc}", flush=True)
        return 1
    print(f"[demo] OK video={mp4} size={mp4.stat().st_size} done={done}/100", flush=True)
    return 0 if ratio >= 0.999 else 2


if __name__ == "__main__":
    raise SystemExit(main())
