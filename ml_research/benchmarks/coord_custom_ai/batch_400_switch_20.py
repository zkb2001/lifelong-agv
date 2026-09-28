"""Batch: SH_custom_01..20 × 400 phased tasks with current switchable stack.

Uses the same task generator as easy_demo_400 and the live solve_ecbs path
(detour-conditional corridor clear included). Writes incremental JSONL + summary.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.hier_coord.run_easy_demo_400_video import gen_demo_400_tasks
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import write_task_csv
from ml_research.common.paths import RESULTS

OUT = RESULTS / "coord_custom_ai" / "batch_400_switch_20"
JSONL = OUT / "results.jsonl"
SUMMARY = OUT / "SUMMARY.md"
SUMMARY_JSON = OUT / "summary.json"
EXCEL_DIR = OUT / "excel"
_EXCEL_ROWS = 1_000_000


def _write_slot_excel(slot: int, rep: Dict[str, Any], wall: float) -> str:
    """One workbook per map: summary sheet plus the trajectory."""
    import pandas as pd

    EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    path = EXCEL_DIR / f"SH{int(slot):02d}.xlsx"
    hier = rep.get("hierarchical") or {}
    summary = pd.DataFrame(
        [
            ("地图", f"SH{int(slot):02d}"),
            ("完成单数", rep.get("tasks_completed")),
            ("总单数", rep.get("tasks_total")),
            ("仿真时间", rep.get("sim_time")),
            ("墙钟秒", wall),
            ("校验", "VALID" if rep.get("validate_ok") else "INVALID"),
            ("校验摘要", rep.get("validate_summary")),
            ("碰撞", rep.get("n_collisions")),
            ("对穿", rep.get("n_swaps")),
            ("非法动作", rep.get("n_illegal_motion")),
            ("FIFO", rep.get("n_fifo")),
            ("规划器", hier.get("last_planner")),
            ("地图难度", hier.get("hardness")),
            ("窄口", hier.get("narrow_cut")),
            ("M0交接", hier.get("m0_handoff")),
            ("ECBS波次", hier.get("waves_ecbs")),
            ("A星波次", hier.get("waves_astar")),
            ("轨迹文件", rep.get("trajectory")),
        ],
        columns=["项目", "值"],
    )
    traj_path = Path(str(rep.get("trajectory") or ""))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="结果", index=False)
        if traj_path.is_file():
            traj = pd.read_csv(traj_path)
            if len(traj) == 0:
                traj.to_excel(writer, sheet_name="轨迹", index=False)
            else:
                for i, start in enumerate(range(0, len(traj), _EXCEL_ROWS)):
                    name = "轨迹" if i == 0 else f"轨迹{i + 1}"
                    traj.iloc[start : start + _EXCEL_ROWS].to_excel(
                        writer, sheet_name=name, index=False
                    )
    print(f"[excel] {path}", flush=True)
    return str(path)


def _run_slot(slot: int, *, seed: int, wall_timeout: float, hard_cap: int) -> Dict[str, Any]:
    meta = load_custom_meta(
        int(slot),
        max_sim_time=500000,
        wall_timeout=float(wall_timeout) if float(wall_timeout) > 0 else 1e9,
        n_tasks=400,
    )
    if not meta:
        return {"slot": slot, "error": "meta_missing"}

    task_csv = OUT / f"SH{slot:02d}_demo400_tasks.csv"
    tasks = gen_demo_400_tasks(seed=int(seed))
    write_task_csv(task_csv, tasks)
    phase_map = {str(t["task_id"]): str(t.get("_phase") or "") for t in tasks}

    # Unified switchable stack for every map: yield ON, escalate by signals.
    # Do not special-case slot. Clear sticky env from prior experiments.
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    # Same-station convoy (max≥2) lets a later FIFO head arrive first →
    # surface_fifo / surface_head INVALID. Cap at 1 approacher per station.
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    # Balanced A*: too low (0.08) floods assign_fail and forces early ECBS
    # (high sim_t); too high crawls wall-clock. Boost every N ticks for hard assigns.
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"
    os.environ["AGV_ASTAR_BOOST_EVERY"] = "10"
    os.environ["AGV_ASTAR_BOOST_S"] = "0.45"

    meta = dict(meta)
    meta_id = f"SH{slot:02d}_m0ecbs400"
    meta["id"] = meta_id
    meta["task_csv"] = str(task_csv.resolve())
    meta["n_tasks"] = 400
    meta["wall_timeout"] = float(wall_timeout)
    meta["progress_every"] = 25
    meta["allow_mode_switch"] = True
    meta["legacy_wave_ecbs"] = True
    meta["gate_every"] = 25
    # Prefer M0 longer: SH02 ~9–16 sim/task when M0 runs; early ECBS blew sim_t.
    # Escalate only after recover window (plateau≥20), not idle@12.
    # NOTE: traffic_recovery=True disables allow_yield (no plateau-recover). Keep off.
    meta["m0_assign_fail_handoff"] = 64
    meta["traffic_recovery"] = False
    meta["m0_min_handoff_t"] = 200
    meta["m0_min_handoff_wall_s"] = 120.0
    meta["m0_wall_tick_s"] = 2.5
    meta["m0_wall_tick_streak"] = 4
    meta["m0_wall_starve_s"] = 280.0
    meta["m0_wall_starve_done"] = 50
    meta["m0_wall_starve_min_t"] = 120
    meta["m0_wall_window_s"] = 90.0
    meta["m0_wall_window_min_sim"] = 30
    meta["m0_wall_window_min_done"] = 3
    meta["m0_wall_done_cost_s"] = 20.0
    meta["m0_wall_done_cost_min_done"] = 20
    meta["m0_throughput_drop_streak"] = 4
    meta["m0_done_plateau_streak"] = 5
    meta["handback_calm"] = 5
    meta["handback_dwell"] = 200
    meta["m0_handback"] = False
    meta["max_handbacks"] = 2
    meta["serial_recover_budget"] = 6
    meta["m0_handback_astar_wall_s"] = 2.0
    meta["m0_handoff_enabled"] = True
    # Opening planner is map structure (hardness / one-cell cut), not wall clock.
    # Runtime M0→ECBS only when completions actually stall.
    meta["wall_first_ecbs"] = False
    meta["wall_first_require_inefficient"] = True
    meta["ecbs_prefer_full_k"] = False
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASTAR_BOOST_EVERY"] = "10"
    os.environ["AGV_ASTAR_BOOST_S"] = "0.45"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"

    # Delivery waves use prioritized ST; ECBS core only when gate demands.
    joint = "prioritized"
    t0 = time.perf_counter()
    print(
        f"\n===== SH{slot:02d} ×400 switch joint={joint} "
        f"auto_yield=1 hard_cap={hard_cap} =====",
        flush=True,
    )
    rep = solve_ecbs(
        slot=int(slot),
        meta=meta,
        max_tasks=0,
        max_active=6,
        weight=1.5,
        time_limit=35.0,
        plan="joint",
        pipeline=True,
        turn_aware=True,
        initial_park=False,
        use_hierarchical=True,
        use_wavenet=False,
        use_swapnet=True,
        hierarchical_force=False,
        wave_hard_cap=int(hard_cap),
        use_traffic_accel=False,
        use_lane_rules=False,
        task_phase_by_id=phase_map,
        plan_horizon=24,
        exec_horizon=12,
        allow_mode_switch=True,
        joint_core=joint,
        legacy_wave_ecbs=True,
        traffic_recovery=False,
    )
    wall = round(time.perf_counter() - t0, 2)
    hier = rep.get("hierarchical") or {}
    excel = ""
    try:
        excel = _write_slot_excel(int(slot), rep, wall)
    except Exception as exc:  # noqa: BLE001
        excel = f"excel_fail:{type(exc).__name__}: {exc}"
        print(f"[excel] SH{slot:02d} {excel}", flush=True)
    row = {
        "slot": slot,
        "scenario_id": meta_id,
        "sim_time": rep.get("sim_time"),
        "completion_ratio": rep.get("completion_ratio"),
        "validate_ok": rep.get("validate_ok"),
        "validate_summary": rep.get("validate_summary"),
        "tasks_completed": rep.get("tasks_completed"),
        "tasks_total": rep.get("tasks_total"),
        "wall_seconds": wall,
        "method": rep.get("method"),
        "joint_core": joint,
        "m0_handoff_enabled": meta["m0_handoff_enabled"],
        "waves_astar": hier.get("waves_astar"),
        "waves_ecbs": hier.get("waves_ecbs"),
        "waves_hybrid": hier.get("waves_hybrid"),
        "last_planner": hier.get("last_planner"),
        "m0_handoff": hier.get("m0_handoff"),
        "hardness": hier.get("hardness"),
        "narrow_cut": hier.get("narrow_cut"),
        "trajectory": rep.get("trajectory"),
        "excel": excel,
        "error": None,
    }
    print(
        f"[done] SH{slot:02d} sim={row['sim_time']} valid={row['validate_ok']} "
        f"ratio={row['completion_ratio']} wall={wall}s "
        f"summary={row['validate_summary']}",
        flush=True,
    )
    return row


def _write_summary(rows: List[Dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("validate_ok")]
    bad = [r for r in rows if not r.get("validate_ok") and not r.get("error")]
    lines = [
        "# Batch 400-switch × SH01–SH20",
        "",
        f"- maps: {len(rows)}",
        f"- VALID: {len(ok)}/{len(rows)}",
        f"- INVALID/incomplete: {len(bad)}",
        "",
        "| Slot | sim_t | VALID | done | wall_s | planner | waves_ecbs | waves_astar | notes |",
        "|------|-------|-------|------|--------|---------|------------|-------------|-------|",
    ]
    for r in sorted(rows, key=lambda x: int(x.get("slot") or 0)):
        if r.get("error"):
            lines.append(
                f"| SH{int(r['slot']):02d} | - | - | - | - | - | - | - | {r['error']} |"
            )
            continue
        notes = str(r.get("validate_summary") or "")[:40]
        lines.append(
            f"| SH{int(r['slot']):02d} | {r.get('sim_time')} | "
            f"{'Y' if r.get('validate_ok') else 'N'} | "
            f"{r.get('tasks_completed')}/{r.get('tasks_total')} | "
            f"{r.get('wall_seconds')} | {r.get('last_planner')} | "
            f"{r.get('waves_ecbs')} | {r.get('waves_astar')} | {notes} |"
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY_JSON.write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wall-timeout", type=float, default=3600.0)
    # Larger ECBS/hybrid core (SH03-style concurrency). Dest-pad claim still
    # bounds how many new pickups enter; delivery moves the full carrier set.
    ap.add_argument("--wave-hard-cap", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done_slots = set()
    rows: List[Dict[str, Any]] = []
    if args.resume and JSONL.is_file():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done_slots.add(int(r.get("slot") or -1))
        print(f"[resume] loaded {len(done_slots)} slots", flush=True)
    else:
        JSONL.write_text("", encoding="utf-8")

    for slot in range(int(args.start), int(args.end) + 1):
        if slot in done_slots:
            print(f"[skip] SH{slot:02d} already done", flush=True)
            continue
        try:
            row = _run_slot(
                slot,
                seed=int(args.seed),
                wall_timeout=float(args.wall_timeout),
                hard_cap=int(args.wave_hard_cap),
            )
        except Exception as exc:  # noqa: BLE001
            row = {"slot": slot, "error": f"{type(exc).__name__}: {exc}", "validate_ok": False}
            print(f"[FAIL] SH{slot:02d} {row['error']}", flush=True)
        with JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows = [r for r in rows if int(r.get("slot") or -1) != slot] + [row]
        _write_summary(rows)

    _write_summary(rows)
    print(f"\n[batch] wrote {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
