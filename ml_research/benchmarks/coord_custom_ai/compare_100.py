"""SH01–20 × official 100: five-way bake-off for hierarchical A*+ECBS.

Methods (same maps / official task CSV / validate_hybrid_trajectory):

  pibt  — reactive PIBT lifelong (``pibt_mapd``)
  m0    — engine spacetime A* (no ECBS wave / no handoff)
  pp    — priority-planning allocator + spacetime A*
  ecbs  — windowed joint ECBS only (legacy wave, no M0 hierarchy)
  hier  — M0 + structure/stall handoff + ECBS (champion stack)

Results: ``results/coord_custom_ai/compare_100/``
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ml_research.benchmarks.allocators import allocator_priority_planning, patch_allocator
from ml_research.benchmarks.common import (
    load_scenario,
    patch_conflict_free_execution,
    patch_extra_obstacles,
    patch_moving_obstacle_horizon,
    run_sim_loop,
    write_trajectory,
)
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.benchmarks.pibt_mapd.solve_pibt import solve_pibt
from ml_research.common.paths import RESULTS

OUT = RESULTS / "coord_custom_ai" / "compare_100"
JSONL = OUT / "results.jsonl"
SUMMARY = OUT / "SUMMARY.md"
SUMMARY_JSON = OUT / "summary.json"
TRAJ = OUT / "trajectories"
PIBT_CACHE = RESULTS / "coord_custom_ai" / "pibt_mapd_100"

METHODS = ("pibt", "m0", "pp", "ecbs", "hier")
METHOD_LABEL = {
    "pibt": "PIBT_reactive",
    "m0": "M0_spacetime_Astar",
    "pp": "PP_priority_Astar",
    "ecbs": "ECBS_windowed",
    "hier": "Hier_M0_ECBS",
}


def _base_meta(slot: int, *, wall_timeout: float, max_sim_time: int) -> dict:
    meta = load_custom_meta(
        int(slot),
        max_sim_time=int(max_sim_time),
        wall_timeout=float(wall_timeout),
        n_tasks=100,
    )
    assert meta, f"missing meta slot={slot}"
    meta = dict(meta)
    meta["n_tasks"] = 100
    meta["wall_timeout"] = float(wall_timeout)
    meta["max_sim_time"] = int(max_sim_time)
    return meta


def _row_from_rep(
    *,
    method: str,
    slot: int,
    sid: str,
    rep: dict,
    wall: float,
    notes: str = "",
) -> dict:
    iss = {}
    # Prefer validator fields already on solve_* reports.
    val_ok = rep.get("validate_ok")
    val_sum = rep.get("validate_summary")
    if val_ok is None and rep.get("trajectory"):
        try:
            meta = {
                "id": sid,
                "slot": slot,
                "task_csv": rep.get("task_csv") or "",
                "position_csv": rep.get("position_csv") or "",
                "map_json": rep.get("map_json") or "",
                "extra_obstacles": rep.get("extra_obstacles") or [],
                "n_tasks": int(rep.get("tasks_total") or 100),
            }
            # Re-validate only if caller did not; compare runners pass meta-aware reps.
            pass
        except Exception:
            pass
    return {
        "method": METHOD_LABEL.get(method, method),
        "method_id": method,
        "slot": int(slot),
        "scenario_id": sid,
        "tasks_total": int(rep.get("tasks_total") or 100),
        "tasks_completed": int(rep.get("tasks_completed") or 0),
        "completion_ratio": float(
            rep.get("completion_ratio")
            if rep.get("completion_ratio") is not None
            else (
                int(rep.get("tasks_completed") or 0)
                / max(1, int(rep.get("tasks_total") or 100))
            )
        ),
        "sim_time": int(rep.get("sim_time") or 0),
        "wall_seconds": round(float(wall), 2),
        "validate_ok": bool(val_ok) if val_ok is not None else False,
        "validate_summary": str(val_sum or ""),
        "n_collisions": int(rep.get("n_collisions") or 0),
        "n_swaps": int(rep.get("n_swaps") or 0),
        "n_fifo": int(rep.get("n_fifo") or 0),
        "trajectory": str(rep.get("trajectory") or ""),
        "notes": notes,
        "error": None,
    }


def _validate_attach(meta: dict, row: dict) -> dict:
    traj = Path(str(row.get("trajectory") or ""))
    if not traj.is_file():
        row["validate_ok"] = False
        row["validate_summary"] = "missing_trajectory"
        return row
    try:
        val = validate_hybrid_trajectory(meta, traj)
        iss = val.get("issues") or {}
        row["validate_ok"] = bool(val.get("ok"))
        row["validate_summary"] = format_validation_summary(val)
        row["n_collisions"] = int(iss.get("n_collisions") or 0)
        row["n_swaps"] = int(iss.get("n_swaps") or 0)
        row["n_fifo"] = int(iss.get("n_fifo_violations") or 0)
    except Exception as exc:  # noqa: BLE001
        row["validate_ok"] = False
        row["validate_summary"] = f"validate_error:{type(exc).__name__}:{exc}"
    return row


def _run_engine(
    *,
    method_id: str,
    meta: dict,
    max_sim_time: int,
    wall_timeout: float,
    allocator=None,
) -> Tuple[dict, float]:
    """M0 / PP via simulation engine + spacetime A*."""
    TRAJ.mkdir(parents=True, exist_ok=True)
    sid = str(meta["id"])
    traj_path = TRAJ / f"{sid}_{method_id}.csv"
    task_csv = Path(meta["task_csv"])
    position_csv = Path(meta["position_csv"])
    mod, env, agv_states, task_states, n_tasks = load_scenario(
        task_csv, position_csv, force_reload=True
    )
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    patch_moving_obstacle_horizon(sim)
    patch_conflict_free_execution(sim)
    if allocator is not None:
        patch_allocator(sim, allocator)
    t0 = time.perf_counter()
    steps, _forced = run_sim_loop(
        sim,
        max_time=int(max_sim_time),
        label=method_id,
        checkpoint_path=None,
        checkpoint_every=0,
        wall_timeout=float(wall_timeout),
    )
    wall = time.perf_counter() - t0
    write_trajectory(traj_path, steps)
    left = sum(len(v) for v in sim.task_states.values())
    completed = int(n_tasks) - int(left)
    return {
        "tasks_total": int(n_tasks),
        "tasks_completed": max(0, completed),
        "completion_ratio": round(max(0, completed) / max(1, int(n_tasks)), 4),
        "sim_time": int(getattr(sim, "time", 0) or 0),
        "wall_seconds": wall,
        "trajectory": str(traj_path),
    }, wall


def _hier_meta(meta: dict) -> dict:
    m = dict(meta)
    m["progress_every"] = 50
    m["allow_mode_switch"] = True
    m["legacy_wave_ecbs"] = True
    m["gate_every"] = 25
    m["m0_handoff_enabled"] = True
    m["m0_assign_fail_handoff"] = 64
    m["traffic_recovery"] = False
    m["m0_min_handoff_t"] = 120
    m["m0_min_handoff_wall_s"] = 60.0
    m["wall_first_ecbs"] = False
    m["ecbs_prefer_full_k"] = False
    return m


def _run_pibt(
    slot: int,
    meta: dict,
    *,
    wall_timeout: float,
    max_sim_time: int,
    reuse: bool,
) -> Tuple[dict, float]:
    if reuse:
        cached = PIBT_CACHE / f"SH{int(slot):02d}_pibt100_pibt.json"
        if cached.is_file():
            rep = json.loads(cached.read_text(encoding="utf-8"))
            print(f"[pibt] reuse {cached.name}", flush=True)
            return rep, float(rep.get("wall_seconds") or 0)
    m = dict(meta)
    m["id"] = f"SH{int(slot):02d}_cmp100_pibt"
    t0 = time.perf_counter()
    rep = solve_pibt(
        int(slot),
        meta=m,
        wall_timeout=float(wall_timeout),
        max_sim_time=int(max_sim_time),
        progress_every=5000,
    )
    return rep, time.perf_counter() - t0


def run_method(
    method: str,
    slot: int,
    *,
    wall_timeout: float,
    max_sim_time: int,
    reuse_pibt: bool,
) -> dict:
    meta = _base_meta(slot, wall_timeout=wall_timeout, max_sim_time=max_sim_time)
    label = METHOD_LABEL[method]
    sid = f"SH{int(slot):02d}_cmp100_{method}"
    meta["id"] = sid
    print(f"\n----- SH{int(slot):02d} / {label} -----", flush=True)
    t0 = time.perf_counter()
    try:
        if method == "pibt":
            rep, wall = _run_pibt(
                slot,
                meta,
                wall_timeout=wall_timeout,
                max_sim_time=max_sim_time,
                reuse=reuse_pibt,
            )
            notes = "reactive PIBT lifelong"
        elif method == "m0":
            # Default solve_ecbs path = M0 engine (no legacy wave).
            os.environ["AGV_M0_HANDOFF"] = "0"
            rep = solve_ecbs(
                slot=int(slot),
                meta=dict(meta),
                max_tasks=0,
                max_active=6,
                weight=1.5,
                time_limit=35.0,
                plan="joint",
                pipeline=True,
                turn_aware=True,
                use_hierarchical=False,
                use_swapnet=False,
                legacy_wave_ecbs=False,
                traffic_recovery=False,
                allow_mode_switch=False,
            )
            wall = float(rep.get("wall_seconds") or (time.perf_counter() - t0))
            notes = "M0 spacetime A* engine (no ECBS)"
        elif method == "pp":
            rep, wall = _run_engine(
                method_id="pp",
                meta=meta,
                max_sim_time=max_sim_time,
                wall_timeout=wall_timeout,
                allocator=allocator_priority_planning,
            )
            notes = "priority PP allocator + spacetime A*"
        elif method == "ecbs":
            os.environ["AGV_M0_HANDOFF"] = "0"
            m = dict(meta)
            m["legacy_wave_ecbs"] = True
            m["allow_mode_switch"] = False
            m["m0_handoff_enabled"] = False
            m["progress_every"] = 50
            rep = solve_ecbs(
                slot=int(slot),
                meta=m,
                max_tasks=0,
                max_active=6,
                weight=1.5,
                time_limit=35.0,
                plan="joint",
                pipeline=True,
                turn_aware=True,
                use_hierarchical=False,
                hierarchical_force=False,
                use_swapnet=False,
                legacy_wave_ecbs=True,
                joint_core="ecbs",
                traffic_recovery=False,
                allow_mode_switch=False,
                plan_horizon=24,
                exec_horizon=12,
                wave_hard_cap=4,
            )
            wall = float(rep.get("wall_seconds") or (time.perf_counter() - t0))
            notes = "windowed joint ECBS only"
        elif method == "hier":
            os.environ["AGV_M0_HANDOFF"] = "1"
            os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
            os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
            os.environ["AGV_ASTAR_WALL_S"] = "0.20"
            os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
            os.environ["AGV_TRIED_TTL"] = "8"
            os.environ["AGV_QUIET_ASSIGN"] = "1"
            m = _hier_meta(meta)
            rep = solve_ecbs(
                slot=int(slot),
                meta=m,
                max_tasks=0,
                max_active=6,
                weight=1.5,
                time_limit=35.0,
                plan="joint",
                pipeline=True,
                turn_aware=True,
                use_hierarchical=True,
                hierarchical_force=False,
                use_swapnet=True,
                use_wavenet=False,
                legacy_wave_ecbs=True,
                joint_core="prioritized",
                traffic_recovery=False,
                allow_mode_switch=True,
                plan_horizon=24,
                exec_horizon=12,
                wave_hard_cap=4,
            )
            wall = float(rep.get("wall_seconds") or (time.perf_counter() - t0))
            notes = "M0 + stall/structure handoff + ECBS"
        else:
            raise ValueError(f"unknown method {method}")
    except Exception as exc:  # noqa: BLE001
        return {
            "method": label,
            "method_id": method,
            "slot": int(slot),
            "scenario_id": sid,
            "error": f"{type(exc).__name__}: {exc}",
            "validate_ok": False,
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "tasks_completed": 0,
            "tasks_total": 100,
            "completion_ratio": 0.0,
            "sim_time": 0,
            "notes": "",
            "trajectory": "",
        }

    row = _row_from_rep(
        method=method,
        slot=slot,
        sid=sid,
        rep=rep,
        wall=wall,
        notes=notes,
    )
    # Always re-validate with this slot's meta for a uniform rule set.
    vmeta = dict(meta)
    vmeta["id"] = sid
    if not row.get("trajectory") and rep.get("trajectory"):
        row["trajectory"] = str(rep["trajectory"])
    row = _validate_attach(vmeta, row)
    print(
        f"[cmp] SH{slot:02d}/{method} done={row['tasks_completed']}/{row['tasks_total']} "
        f"valid={row['validate_ok']} sim={row['sim_time']} wall={row['wall_seconds']}s "
        f"{row['validate_summary']}",
        flush=True,
    )
    return row


def _write_summary(rows: List[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Pivot: rows by slot × method
    slots = sorted({int(r["slot"]) for r in rows if r.get("slot") is not None})
    methods = [m for m in METHODS if any(r.get("method_id") == m for r in rows)]
    lines = [
        "# Compare 100 — SH01–20 × official 100",
        "",
        "Methods: " + ", ".join(METHOD_LABEL[m] for m in methods),
        "",
        "## Per-map completion (done/100) and VALID",
        "",
        "| Slot | "
        + " | ".join(m for m in methods)
        + " |",
        "|------|"
        + "|".join(["------"] * len(methods))
        + "|",
    ]
    by = {(int(r["slot"]), r.get("method_id")): r for r in rows}
    for s in slots:
        cells = []
        for m in methods:
            r = by.get((s, m))
            if not r or r.get("error"):
                cells.append(r.get("error", "err")[:24] if r else "-")
            else:
                mark = "Y" if r.get("validate_ok") else "N"
                cells.append(f"{r.get('tasks_completed')}/100 {mark}")
        lines.append(f"| SH{s:02d} | " + " | ".join(cells) + " |")

    lines += ["", "## Aggregate", ""]
    lines.append("| Method | maps | full100 | VALID | avg_done | avg_sim | avg_wall |")
    lines.append("|--------|------|---------|-------|----------|---------|----------|")
    for m in methods:
        rs = [r for r in rows if r.get("method_id") == m and not r.get("error")]
        if not rs:
            continue
        n = len(rs)
        full = sum(1 for r in rs if int(r.get("tasks_completed") or 0) >= 100)
        valid = sum(1 for r in rs if r.get("validate_ok"))
        avg_done = sum(int(r.get("tasks_completed") or 0) for r in rs) / n
        avg_sim = sum(int(r.get("sim_time") or 0) for r in rs) / n
        avg_wall = sum(float(r.get("wall_seconds") or 0) for r in rs) / n
        lines.append(
            f"| {METHOD_LABEL[m]} | {n} | {full}/{n} | {valid}/{n} | "
            f"{avg_done:.1f} | {avg_sim:.0f} | {avg_wall:.1f} |"
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY_JSON.write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Five-way compare on official 100 tasks")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument(
        "--methods",
        type=str,
        default="pibt,m0,pp,ecbs,hier",
        help="comma list: pibt,m0,pp,ecbs,hier",
    )
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument("--max-sim-time", type=int, default=200000)
    ap.add_argument("--reuse-pibt", action="store_true", help="reuse pibt_mapd_100 JSONs")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    wanted = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    for m in wanted:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}; choose from {METHODS}")

    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    done_keys = set()
    if args.resume and JSONL.is_file():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done_keys.add((int(r.get("slot") or -1), str(r.get("method_id") or "")))
        print(f"[resume] {len(done_keys)} rows", flush=True)
    else:
        JSONL.write_text("", encoding="utf-8")

    for slot in range(int(args.start), int(args.end) + 1):
        for method in wanted:
            key = (slot, method)
            if key in done_keys:
                print(f"[skip] SH{slot:02d}/{method}", flush=True)
                continue
            row = run_method(
                method,
                slot,
                wall_timeout=float(args.wall_timeout),
                max_sim_time=int(args.max_sim_time),
                reuse_pibt=bool(args.reuse_pibt),
            )
            with JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows = [
                r
                for r in rows
                if not (
                    int(r.get("slot") or -1) == slot
                    and str(r.get("method_id") or "") == method
                )
            ] + [row]
            _write_summary(rows)

    _write_summary(rows)
    print(f"\n[compare_100] wrote {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
