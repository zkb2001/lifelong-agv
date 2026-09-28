"""Easy-map hierarchical demo: 400 phased tasks → validate → ~2m30s MP4.

Task schedule (1-based index, seed-controlled):
  1–29     random dropoff
  30–60    all dropoff → Beijing
  61–99    random dropoff
  100–300  all dropoff → Tianjin
  301–400  random dropoff
  = 400
"""
from __future__ import annotations

import argparse
import json
import os
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

OUT = RESULTS / "hier_coord" / "easy_demo_400"
VIDEO_DIR = RESULTS / "hier_coord" / "easy_demo_400" / "videos"


def gen_demo_400_tasks(*, seed: int = 42) -> List[dict]:
    """Build 400 tasks with Beijing / Tianjin dropoff concentration windows."""
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

    # 1–29: random
    for _ in range(29):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_1_29")
    # 30–60: all → Beijing (31 tasks)
    for _ in range(31):
        add(rng.choice(ALL_STATIONS), "Beijing", "hub_beijing_30_60")
    # 61–99: random
    for _ in range(39):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_61_99")
    # 100–300: all → Tianjin (201 tasks)
    for _ in range(201):
        add(rng.choice(ALL_STATIONS), "Tianjin", "hub_tianjin_100_300")
    # 301–400: random
    for _ in range(100):
        add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "random_301_400")

    assert len(rows) == 400, len(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=1, help="easy map slot (default SH01)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=35.0)
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="mp4 length seconds; default = sim_time/4",
    )
    ap.add_argument("--plan-horizon", type=int, default=24)
    ap.add_argument("--exec-horizon", type=int, default=12)
    ap.add_argument(
        "--wall-timeout",
        type=float,
        default=1800.0,
        help="M0 engine wall-clock cap seconds (0=no cap)",
    )
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument(
        "--allow-mode-switch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --legacy-wave-ecbs: start A*/M0, escalate to hybrid/ECBS "
        "when SceneDifficulty says medium/hard (no force_hard). "
        "Without legacy flag this is ignored.",
    )
    ap.add_argument(
        "--legacy-wave-ecbs",
        action="store_true",
        help="Use SceneDifficulty / wave ECBS path. Default pairs with "
        "force_hard; add --allow-mode-switch for A*↔ECBS switching.",
    )
    ap.add_argument(
        "--no-traffic-recovery",
        action="store_true",
        help="Disable TrafficNet parking/semaphore recovery on M0 path.",
    )
    ap.add_argument(
        "--joint-core",
        choices=("prioritized", "ecbs"),
        default="ecbs",
        help="Joint core for --legacy-wave-ecbs (default ecbs, same as SH03)",
    )
    ap.add_argument("--wave-hard-cap", type=int, default=4)
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--traj", type=str, default="")
    ap.add_argument("--task-csv", type=str, default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_custom_meta(
        int(args.slot),
        max_sim_time=500000,
        wall_timeout=float(args.wall_timeout) if float(args.wall_timeout) > 0 else 1e9,
    )
    if not meta:
        raise SystemExit(f"slot {args.slot} missing")

    task_csv = (
        Path(args.task_csv) if args.task_csv else OUT / f"SH{args.slot:02d}_demo400_tasks.csv"
    )
    phase_map: Dict[str, str] = {}
    if not args.skip_solve:
        tasks = gen_demo_400_tasks(seed=int(args.seed))
        write_task_csv(task_csv, tasks)
        phase_map = {str(t["task_id"]): str(t.get("_phase") or "") for t in tasks}
        phase_path = OUT / f"SH{args.slot:02d}_demo400_phases.json"
        phase_path.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "n_tasks": len(tasks),
                    "schedule": {
                        "random_1_29": 29,
                        "hub_beijing_30_60": 31,
                        "random_61_99": 39,
                        "hub_tianjin_100_300": 201,
                        "random_301_400": 100,
                    },
                    "hubs": {"30-60": "Beijing", "100-300": "Tianjin"},
                    "phases": phase_map,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[demo] wrote {task_csv} ({len(tasks)} tasks)", flush=True)
        bj = sum(1 for t in tasks[29:60] if t["end_point"] == "Beijing")
        tj = sum(1 for t in tasks[99:300] if t["end_point"] == "Tianjin")
        print(
            f"[demo] check beijing[30-60]={bj}/31 tianjin[100-300]={tj}/201",
            flush=True,
        )

    use_legacy = bool(args.legacy_wave_ecbs)
    switchable = bool(use_legacy and args.allow_mode_switch)
    meta = dict(meta)
    if switchable:
        meta_id = f"SH{args.slot:02d}_easy_demo400_switch"
    elif use_legacy:
        meta_id = f"SH{args.slot:02d}_easy_demo400_ecbs_wave"
    else:
        meta_id = f"SH{args.slot:02d}_easy_demo400"
    meta["id"] = meta_id
    meta["task_csv"] = str(task_csv.resolve())
    meta["n_tasks"] = 400
    meta["wall_timeout"] = float(args.wall_timeout)
    meta["progress_every"] = int(args.progress_every)
    meta["allow_mode_switch"] = bool(switchable)
    meta["legacy_wave_ecbs"] = bool(use_legacy)
    meta["traffic_recovery"] = (
        False if bool(use_legacy) else not bool(args.no_traffic_recovery)
    )
    if switchable:
        meta["gate_every"] = max(25, int(args.progress_every) // 2)
        meta["m0_assign_fail_handoff"] = 16
        meta["m0_min_handoff_t"] = 150
        meta["m0_throughput_drop_streak"] = 3
        meta["handback_calm"] = 5
        meta["handback_dwell"] = 200
        meta["m0_handback"] = True
        meta["max_handbacks"] = 2
        meta["serial_recover_budget"] = 6
        # Handback-only A* room. Do NOT raise cold-start AGV_ASTAR_WALL_S —
        # env/default 0.35 keeps SH01 wall-clock sane; warm resume sets 2.0.
        meta["m0_handback_astar_wall_s"] = 2.0
        # Switchable: always allow M0→wave yield. Escalate is decided by
        # runtime stall/hardness signals (not by slot id). AGV_M0_HANDOFF=0
        # remains an explicit debug opt-out only.
        meta["m0_handoff_enabled"] = str(
            os.environ.get("AGV_M0_HANDOFF", "1")
        ).strip().lower() not in ("0", "false", "no", "off")
        if "AGV_PROMOTE_ON_ASSIGN" not in os.environ:
            # Surface multitask: expose next FIFO head on assign (same-station
            # still capped at 1 approacher in engine allocate).
            os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
        if "AGV_STATION_MAX_UNLOADED" not in os.environ:
            os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
        if "AGV_ASTAR_WALL_S" not in os.environ:
            # Match VALID SH01: 0.35 under-budgets A* → fail/retry thrash.
            os.environ["AGV_ASTAR_WALL_S"] = "2.0"
        # Field default: conflict shield OFF (teleport risk). Do not force on.

    if not args.skip_solve:
        t0 = time.perf_counter()
        # --legacy-wave-ecbs: force_hard ECBS, or switchable A*/M0↔ECBS.
        # else: M0+SwapNet+RulePark.
        if use_legacy:
            force_hard = not switchable
            use_wavenet = bool(switchable)
            # Switchable: keep ECBS capability ready; gate chooses astar/hybrid/ecbs.
            # Non-switchable force-hard also uses requested joint_core.
            if switchable:
                joint = "ecbs"
            else:
                joint = str(args.joint_core)
            print(
                f"[demo] solve {'switchable A*/ECBS' if switchable else 'baseline+ECBS-wave'} "
                f"slot={args.slot} tasks=400 joint_core={joint} "
                f"force_hard={int(force_hard)} allow_mode_switch={int(switchable)} "
                f"wavenet={int(use_wavenet)}",
                flush=True,
            )
            rep = solve_ecbs(
                slot=int(args.slot),
                meta=meta,
                max_tasks=0,
                max_active=0,
                weight=1.5,
                time_limit=float(args.time_limit),
                plan="joint",
                pipeline=True,
                turn_aware=True,
                initial_park=False,
                use_hierarchical=True,
                use_wavenet=use_wavenet,
                use_swapnet=bool(switchable),  # M0+SwapNet on switchable easy start
                hierarchical_force=bool(force_hard),
                wave_hard_cap=int(args.wave_hard_cap),
                use_traffic_accel=False,
                use_lane_rules=False,
                task_phase_by_id=phase_map or None,
                plan_horizon=int(args.plan_horizon),
                exec_horizon=int(args.exec_horizon),
                allow_mode_switch=bool(switchable),
                joint_core=joint,
                legacy_wave_ecbs=True,
                traffic_recovery=False,
            )
        else:
            print(
                f"[demo] solve M0+SwapNet+RulePark slot={args.slot} tasks=400 "
                f"traffic={meta['traffic_recovery']}",
                flush=True,
            )
            rep = solve_ecbs(
                slot=int(args.slot),
                meta=meta,
                max_tasks=0,
                max_active=0,
                weight=1.5,
                time_limit=float(args.time_limit),
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
                task_phase_by_id=phase_map or None,
                plan_horizon=int(args.plan_horizon),
                exec_horizon=int(args.exec_horizon),
                allow_mode_switch=False,
                joint_core=str(args.joint_core),
                legacy_wave_ecbs=False,
                traffic_recovery=not bool(args.no_traffic_recovery),
            )
        wall = round(time.perf_counter() - t0, 2)
        hier = rep.get("hierarchical") or {}
        rep_out = {
            **{
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
                    "n_pickup_no_dwell",
                    "n_unload_no_dwell",
                    "n_fifo",
                    "n_pickup_cell",
                    "n_task_carry",
                    "n_premature_unload",
                    "n_foreign_unload",
                    "n_display_mismatch",
                    "method",
                    "plan_horizon",
                    "exec_horizon",
                )
            },
            "demo_wall": wall,
            "slot": args.slot,
            "task_csv": str(task_csv),
            "schedule": {
                "random_1_29": 29,
                "hub_beijing_30_60": 31,
                "random_61_99": 39,
                "hub_tianjin_100_300": 201,
                "random_301_400": 100,
            },
            "hierarchical": {
                "waves_astar": hier.get("waves_astar"),
                "waves_ecbs": hier.get("waves_ecbs"),
                "waves_hybrid": hier.get("waves_hybrid"),
                "waves_escalated": hier.get("waves_escalated"),
                "last_reason": hier.get("last_reason"),
                "last_planner": hier.get("last_planner"),
                "last_scene_label": hier.get("last_scene_label"),
                "m0_handoff": hier.get("m0_handoff"),
                "m0_handoff_t": hier.get("m0_handoff_t"),
                "m0_handoff_label": hier.get("m0_handoff_label"),
                "allow_mode_switch": hier.get("allow_mode_switch"),
                "traffic_recovery": hier.get("traffic_recovery"),
                "traffic_stats": hier.get("traffic_stats"),
            },
            "mode": "switchable" if switchable else ("ecbs_wave" if use_legacy else "m0"),
            "traffic_recovery": rep.get("traffic_recovery"),
        }
        summary = OUT / (
            f"SH{args.slot:02d}_demo400_switch_summary.json"
            if switchable
            else f"SH{args.slot:02d}_demo400_summary.json"
        )
        summary.write_text(
            json.dumps(rep_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(rep_out, indent=2, ensure_ascii=False), flush=True)
        done_ok = (
            float(rep.get("completion_ratio") or 0) >= 0.999
            and not (rep.get("tasks_failed") or [])
        )
        # Hard fail includes unload / surface-FIFO checks — no soft-pass that
        # ignores premature/foreign unload or non-surface pickups.
        hard_bad = (
            int(rep.get("n_collisions") or 0)
            + int(rep.get("n_swaps") or 0)
            + int(rep.get("n_hard_wall") or 0)
            + int(rep.get("n_illegal_motion") or 0)
            + int(rep.get("n_pickup_no_dwell") or 0)
            + int(rep.get("n_unload_no_dwell") or 0)
            + int(rep.get("n_fifo") or 0)
            + int(rep.get("n_pickup_cell") or 0)
            + int(rep.get("n_task_carry") or 0)
            + int(rep.get("n_premature_unload") or 0)
            + int(rep.get("n_foreign_unload") or 0)
            + int(rep.get("n_display_mismatch") or 0)
        )
        ok = done_ok and bool(rep.get("validate_ok")) and hard_bad == 0
        if not ok:
            print(
                f"[demo] FAIL validate/completion "
                f"validate={rep.get('validate_summary')} hard_bad={hard_bad}",
                flush=True,
            )
            return 1
        traj = Path(rep["trajectory"])
    else:
        if not args.traj:
            raise SystemExit("--traj required with --skip-solve")
        traj = Path(args.traj)
        if not task_csv.exists():
            raise SystemExit(f"task csv missing: {task_csv}")

    if args.skip_video:
        print(f"[demo] skip video traj={traj}", flush=True)
        return 0

    sim_t = 0
    if not args.skip_solve:
        sim_t = int((rep or {}).get("sim_time") or 0)
    if sim_t <= 0 and traj.exists():
        try:
            import pandas as pd

            sim_t = int(pd.read_csv(traj)["timestamp"].max())
        except Exception:  # noqa: BLE001
            sim_t = 0
    if args.duration is not None and float(args.duration) > 0:
        duration_s = float(args.duration)
    else:
        duration_s = max(1.0, float(sim_t) / 4.0)

    tag = "switch_" if switchable else ""
    mp4 = VIDEO_DIR / f"SH{args.slot:02d}_easy_demo400_{tag}{int(round(duration_s))}s.mp4"
    print(
        f"[demo] render mp4 duration≈{duration_s:.1f}s (sim={sim_t}, sim/4) → {mp4}",
        flush=True,
    )
    obs = Path(meta["map_json"]) if meta.get("map_json") else None
    rc = render_mp4(
        traj,
        mp4,
        task_csv=task_csv,
        position_csv=Path(meta["position_csv"]),
        obstacles_json=obs,
        speed=1.0,
        duration=float(duration_s),
    )
    if rc != 0 or not mp4.exists():
        print(f"[demo] video render failed rc={rc}", flush=True)
        return 1
    print(f"[demo] OK video={mp4} size={mp4.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
