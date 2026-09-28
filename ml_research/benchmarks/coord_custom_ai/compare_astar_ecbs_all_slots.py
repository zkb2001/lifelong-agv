"""Batch compare M0 spacetime-A* baseline vs Pipeline ECBS on SH_custom_01..20.

Uses existing paper A* trajectories (100 tasks each) when present; runs missing
Pipeline ECBS (max_active=6, full task queue, --no-initial-park).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ml_research.benchmarks.coord_custom_ai.compare_pipeline_baseline import (
    traj_motion_stats,
)
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER_ASTAR = (
    ROOT
    / "results"
    / "curriculum_shape"
    / "paper_baseline_astar_custom"
    / "trajectories"
)
OUT = ROOT / "results" / "coord_custom_ai" / "compare"
ECBS_TRAJ = ROOT / "results" / "coord_custom_ai" / "ecbs" / "trajectories"


def count_tasks_in_traj(traj: Path) -> int:
    seen: set = set()
    with open(traj, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = (row.get("task-id") or "").strip()
            if tid:
                seen.add(tid)
    return len(seen)


def analyze_traj(meta: dict, traj: Path) -> dict:
    mot = traj_motion_stats(traj)
    val = validate_hybrid_trajectory(meta, traj)
    return {
        "sim_time": int(mot["sim_time"]),
        "total_move_steps": int(mot["total_move_steps"]),
        "total_wait_steps": int(mot["total_wait_steps"]),
        "total_turn_steps": int(mot["total_turn_steps"]),
        "tasks_seen_in_traj": count_tasks_in_traj(traj),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "trajectory": str(traj.resolve()),
    }


def run_pipeline_ecbs(slot: int, *, max_active: int = 6) -> dict:
    rep = solve_ecbs(
        slot=slot,
        max_tasks=0,
        max_active=max_active,
        weight=1.5,
        plan="joint",
        initial_park=False,
        pipeline=True,
    )
    out = OUT / f"SH_custom_{slot:02d}_pipeline_k{max_active}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


def row_for_slot(
    slot: int,
    *,
    max_active: int = 6,
    skip_ecbs: bool = False,
    force_ecbs: bool = False,
) -> dict:
    meta = load_custom_meta(slot, max_sim_time=500000, wall_timeout=1e9)
    if meta is None:
        return {"slot": slot, "error": "meta missing"}

    sid = meta["id"]
    n_tasks = int(meta.get("n_tasks") or 100)
    row: dict = {
        "slot": slot,
        "scenario_id": sid,
        "tasks_total": n_tasks,
    }

    astar = PAPER_ASTAR / f"{sid}.csv"
    if astar.is_file():
        t0 = time.perf_counter()
        a = analyze_traj(meta, astar)
        row["astar"] = {
            **a,
            "method": "M0_spacetime_Astar",
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }
    else:
        row["astar"] = {"error": f"missing {astar}"}

    pipe_cache = OUT / f"{sid}_pipeline_k{max_active}.json"
    traj_pipe = ECBS_TRAJ / f"{sid}_ecbs_joint_w1.5_k{max_active}.csv"

    if not skip_ecbs and (force_ecbs or not pipe_cache.is_file()):
        print(f"[batch] run pipeline ECBS {sid} …", flush=True)
        t0 = time.perf_counter()
        try:
            rep = run_pipeline_ecbs(slot, max_active=max_active)
            wall = round(time.perf_counter() - t0, 1)
            traj_pipe = Path(str(rep.get("trajectory") or ""))
        except Exception as exc:
            row["pipeline_ecbs"] = {"error": str(exc)}
            return row
    elif pipe_cache.is_file():
        rep = json.loads(pipe_cache.read_text(encoding="utf-8"))
        if not isinstance(rep, dict):
            row["pipeline_ecbs"] = {"error": f"bad cache: {type(rep)}"}
            return row
        wall = rep.get("wall_seconds")
        traj_pipe = Path(str(rep.get("trajectory") or ECBS_TRAJ / f"{sid}_ecbs_joint_w1.5_k{max_active}.csv"))
    else:
        rep = None
        wall = None
        traj_pipe = ECBS_TRAJ / f"{sid}_ecbs_joint_w1.5_k{max_active}.csv"

    if isinstance(rep, dict) and traj_pipe.is_file():
        t0 = time.perf_counter()
        p = analyze_traj(meta, traj_pipe)
        row["pipeline_ecbs"] = {
            **p,
            "method": "pipeline_ecbs_joint",
            "tasks_completed": rep.get("tasks_completed"),
            "completion_ratio": rep.get("completion_ratio"),
            "wall_seconds_run": wall,
            "wall_seconds_analyze": round(time.perf_counter() - t0, 2),
        }
        if p["sim_time"] and row.get("astar", {}).get("sim_time"):
            ast = row["astar"]["sim_time"]
            pip = p["sim_time"]
            row["delta_sim_time"] = ast - pip
            row["delta_sim_time_pct"] = round(100.0 * (ast - pip) / max(1, ast), 1)
        am = row.get("astar", {}).get("total_move_steps")
        pm = p.get("total_move_steps")
        if am is not None and pm is not None:
            row["delta_move_steps"] = am - pm
            row["delta_move_steps_pct"] = round(
                100.0 * (am - pm) / max(1, am), 1
            )
    elif not skip_ecbs:
        row["pipeline_ecbs"] = {"error": "pipeline run failed or traj missing"}

    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="1-20", help="e.g. 1-20 or 3,5,7")
    ap.add_argument("--max-active", type=int, default=6)
    ap.add_argument("--skip-ecbs", action="store_true", help="only analyze A*")
    ap.add_argument("--force-ecbs", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="merge into existing JSON; skip slots with successful pipeline",
    )
    args = ap.parse_args()

    if "-" in args.slots and "," not in args.slots:
        a, b = args.slots.split("-", 1)
        slots = list(range(int(a), int(b) + 1))
    else:
        slots = [int(x) for x in args.slots.split(",") if x.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "SH_custom_all_astar_vs_pipeline.json"
    by_slot: Dict[int, dict] = {}
    if args.resume and out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prev, list):
                for r in prev:
                    if isinstance(r, dict) and "slot" in r:
                        by_slot[int(r["slot"])] = r
        except Exception as exc:
            print(f"[batch] resume load failed: {exc}", flush=True)

    results: List[dict] = []
    for slot in slots:
        pipe_cache = OUT / f"SH_custom_{slot:02d}_pipeline_k{args.max_active}.json"
        existing = by_slot.get(slot) or {}
        pe = existing.get("pipeline_ecbs") if isinstance(existing, dict) else None
        already_ok = (
            isinstance(pe, dict)
            and pe.get("validate_ok") is True
            and int(pe.get("tasks_completed") or 0) >= 100
            and pipe_cache.is_file()
        )
        if args.resume and already_ok and not args.force_ecbs:
            print(f"[batch] === slot {slot} SKIP (cached ok) ===", flush=True)
            results.append(existing)
            by_slot[slot] = existing
        else:
            print(f"[batch] === slot {slot} ===", flush=True)
            try:
                row = row_for_slot(
                    slot,
                    max_active=args.max_active,
                    skip_ecbs=args.skip_ecbs,
                    force_ecbs=args.force_ecbs,
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                row = {"slot": slot, "error": str(exc)}
            results.append(row)
            by_slot[slot] = row
        # incremental save (keep all known slots sorted)
        merged = [by_slot[s] for s in sorted(by_slot)]
        out_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # summary table
    print("\n=== A* baseline vs Pipeline ECBS (100 tasks, k=6) ===", flush=True)
    print(
        f"{'slot':>4}  {'A* sim':>8}  {'pipe sim':>8}  {'Δt':>8}  "
        f"{'A* move':>8}  {'pipe move':>9}  {'A* val':>6}  {'pipe val':>8}  "
        f"{'A* tasks':>8}  {'pipe done':>9}",
        flush=True,
    )
    for r in results:
        if "error" in r and "astar" not in r:
            print(f"{r.get('slot', '?'):>4}  ERROR {r['error']}")
            continue
        a = r.get("astar") or {}
        p = r.get("pipeline_ecbs") or {}
        print(
            f"{r.get('slot', 0):>4}  "
            f"{a.get('sim_time', '-'):>8}  "
            f"{p.get('sim_time', '-'):>8}  "
            f"{r.get('delta_sim_time', '-'):>8}  "
            f"{a.get('total_move_steps', '-'):>8}  "
            f"{p.get('total_move_steps', '-'):>9}  "
            f"{str(a.get('validate_ok', '-')):>6}  "
            f"{str(p.get('validate_ok', '-')):>8}  "
            f"{a.get('tasks_seen_in_traj', '-'):>8}  "
            f"{p.get('tasks_completed', '-'):>9}",
            flush=True,
        )

    print(f"\n[batch] saved {OUT / 'SH_custom_all_astar_vs_pipeline.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
