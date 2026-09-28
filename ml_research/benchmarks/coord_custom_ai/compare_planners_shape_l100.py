"""Fair planner compare on curriculum_shape L100 (videos batch).

Maps: SH01..SH20 from ml_research/results/curriculum_shape/scenarios/*_L100.json
Tasks: full 100-task CSV per map (same file for all methods).
M0 baseline can be loaded from paper_baseline_astar/L100_astar_metrics.csv.

Methods: M0 A*, A*+SwapNet, M2 PP, Pipeline ECBS.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ml_research.benchmarks.allocators import allocator_priority_planning
from ml_research.benchmarks.coord_custom_ai.compare_pipeline_baseline import traj_motion_stats
from ml_research.benchmarks.coord_custom_ai.compare_planners_20 import (
    _aggregate,
    _finalize_from_traj,
    _print_tables,
    _run_sim_method,
    _write_table,
)
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.common.paths import RESULTS

SCENARIOS = RESULTS / "curriculum_shape" / "scenarios"
BASELINE_CSV = (
    RESULTS / "curriculum_shape" / "paper_baseline_astar" / "L100_astar_metrics.csv"
)
DEFAULT_OUT = RESULTS / "curriculum_shape" / "planner_compare_l100_pure"
HEAVY_OUT = RESULTS / "curriculum_shape" / "planner_compare_l100_fresh"
OUT = DEFAULT_OUT
TRAJ = OUT / "trajectories"


def _method_keys(methods: Sequence[str]) -> set[str]:
    wanted: set[str] = set()
    for m in methods:
        k = m.lower()
        if k in ("m0", "astar"):
            wanted.add("m0")
        elif k in ("swapnet", "astar_swapnet"):
            wanted.add("swapnet")
        elif k in ("m2", "pp"):
            wanted.add("m2")
        elif k in ("ecbs", "pipeline_ecbs"):
            wanted.add("ecbs")
        else:
            wanted.add(k)
    return wanted


def _row_method_key(row: dict) -> str:
    method = str(row.get("method") or "")
    if method.startswith("M0"):
        return "m0"
    if "SwapNet" in method:
        return "swapnet"
    if "M2" in method or "PP" in method:
        return "m2"
    if "ECBS" in method:
        return "ecbs"
    return method.lower()


def _load_existing_rows(out_dir: Path) -> List[dict]:
    path = out_dir / "per_map.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    csv_path = out_dir / "per_map.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def _done_keys(rows: List[dict]) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for r in rows:
        if r.get("error"):
            continue
        sid = str(r.get("scenario_id") or "")
        if sid:
            done.add((sid, _row_method_key(r)))
    return done


def load_l100_metas(ids: Optional[Sequence[str]] = None) -> List[dict]:
    metas: List[dict] = []
    for i, p in enumerate(sorted(SCENARIOS.glob("*_L100.json")), start=1):
        data = json.loads(p.read_text(encoding="utf-8"))
        if ids and data.get("id") not in ids:
            continue
        meta = dict(data)
        meta["slot"] = i
        meta.setdefault("max_sim_time", 5000)
        meta.setdefault("wall_timeout", 7200.0)
        metas.append(meta)
    return metas


def _load_cached_m0(meta: dict) -> Optional[dict]:
    if not BASELINE_CSV.is_file():
        return None
    sid = meta["id"]
    with BASELINE_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scenario_id") == sid:
                traj = Path(row.get("trajectory") or "")
                if not traj.is_file():
                    return None
                mot = traj_motion_stats(traj)
                return _finalize_from_traj(
                    method="M0_spacetime_Astar",
                    meta=meta,
                    traj=traj,
                    tasks_total=int(row.get("n_tasks") or meta.get("n_tasks") or 100),
                    tasks_completed=int(row.get("tasks_completed") or 0),
                    sim_time=int(row.get("sim_time") or mot.get("sim_time") or 0),
                    wall=float(row.get("wall_solve_seconds") or 0),
                    notes="cached paper_baseline_astar L100",
                    extra={
                        "forced_stop": str(row.get("forced_stop", "")).lower() == "true",
                        "conflict_free": str(row.get("conflict_free", "")).lower() == "true",
                        "collisions": int(row.get("collisions") or 0),
                    },
                )
    return None


def run_scenario(
    meta: dict,
    *,
    methods: Sequence[str],
    max_time: int,
    wall_timeout: float,
    max_active_ecbs: int,
    use_cached_m0: bool,
    pure_engine: bool,
    skip_done: Optional[set[tuple[str, str]]] = None,
) -> List[dict]:
    sid = str(meta["id"])
    task_csv = Path(meta["task_csv"])
    pos_csv = Path(meta["position_csv"])
    obs = meta.get("extra_obstacles") or []
    n_tasks = int(meta.get("n_tasks") or 100)
    n_agvs = int(meta.get("n_agvs") or 8)
    TRAJ.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    wanted = _method_keys(methods)
    done = skip_done or set()

    mode = "pure" if pure_engine else "heavy"
    print(f"[{sid}] tasks={n_tasks} agvs={n_agvs} mode={mode}", flush=True)
    sim_notes = "pure engine greedy + spacetime A*"
    swap_notes = "pure engine + SwapNet swap gate"
    m2_notes = "pure engine + priority planning"

    m0_done = ("m0" in wanted and (sid, "m0") in done)
    if m0_done:
        print(f"[{sid}] M0 skip (already done)", flush=True)

    if ("m0" in wanted) and use_cached_m0 and not m0_done:
        cached = _load_cached_m0(meta)
        if cached:
            print(f"[{sid}] M0 cached OK sim_t={cached.get('sim_time')}", flush=True)
            rows.append(cached)
            m0_done = True

    if ("m0" in wanted) and not m0_done:
        print(f"[{sid}] M0 spacetime A* …", flush=True)
        traj = TRAJ / f"{sid}_M0.csv"
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
                pure_engine=pure_engine,
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
                    notes=sim_notes,
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
                    "scenario_id": sid,
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    if "swapnet" in wanted and (sid, "swapnet") in done:
        print(f"[{sid}] SwapNet skip (already done)", flush=True)
    elif "swapnet" in wanted:
        print(f"[{sid}] A* + SwapNet …", flush=True)
        traj = TRAJ / f"{sid}_SwapNet.csv"
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
                pure_engine=pure_engine,
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
                    notes=swap_notes,
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
                    "scenario_id": sid,
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    if ("m2" in wanted) and (sid, "m2") in done:
        print(f"[{sid}] M2 skip (already done)", flush=True)
    elif "m2" in wanted:
        print(f"[{sid}] M2 priority PP …", flush=True)
        traj = TRAJ / f"{sid}_M2.csv"
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
                pure_engine=pure_engine,
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
                    notes=m2_notes,
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
                    "scenario_id": sid,
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    if ("ecbs" in wanted) and (sid, "ecbs") in done:
        print(f"[{sid}] ECBS skip (already done)", flush=True)
    elif "ecbs" in wanted:
        print(f"[{sid}] Pipeline ECBS …", flush=True)
        t0 = time.perf_counter()
        try:
            rep = solve_ecbs(
                slot=0,
                meta=meta,
                max_tasks=0,
                max_active=min(max_active_ecbs, max(n_agvs, 1)),
                weight=1.5,
                plan="joint",
                initial_park=False,
                pipeline=True,
                task_csv=task_csv,
            )
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
                    notes=f"windowed ECBS w=1.5 k={min(max_active_ecbs, n_agvs)}",
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
                    "scenario_id": sid,
                    "error": str(exc),
                    "wall_seconds": round(time.perf_counter() - t0, 2),
                }
            )

    for r in rows:
        r.setdefault("scenario_id", sid)
        r.setdefault("map_shape", meta.get("map_shape", ""))
        r.setdefault("n_agvs", n_agvs)
        r.setdefault("tasks_total", n_tasks)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="curriculum_shape L100 planner compare")
    ap.add_argument("--ids", type=str, default="", help="comma scenario ids, default all L100")
    ap.add_argument("--max-time", type=int, default=5000)
    ap.add_argument("--wall", type=float, default=7200.0)
    ap.add_argument(
        "--methods",
        type=str,
        default="m0,swapnet,m2,ecbs",
    )
    ap.add_argument("--max-active-ecbs", type=int, default=6)
    ap.add_argument("--use-cached-m0", action="store_true", default=False)
    ap.add_argument("--no-cached-m0", action="store_false", dest="use_cached_m0")
    ap.add_argument(
        "--pure-engine",
        action="store_true",
        default=True,
        help="fast path like run_lifelong_baseline (default)",
    )
    ap.add_argument(
        "--heavy-pipeline",
        action="store_false",
        dest="pure_engine",
        help="horizon=12 + execution shield + checkpoints (slow, for ECBS-fair rules)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="output dir (default: planner_compare_l100_pure or _fresh if heavy)",
    )
    args = ap.parse_args(argv)

    if args.pure_engine:
        os.environ["MOVING_OBSTACLE_HORIZON"] = "0"
    else:
        os.environ.setdefault("MOVING_OBSTACLE_HORIZON", "12")

    global OUT, TRAJ
    if args.out:
        OUT = Path(args.out)
        if not OUT.is_absolute():
            OUT = RESULTS / "curriculum_shape" / OUT
    elif args.pure_engine:
        OUT = DEFAULT_OUT
    else:
        OUT = HEAVY_OUT
    TRAJ = OUT / "trajectories"

    ids = [x.strip() for x in args.ids.split(",") if x.strip()] or None
    metas = load_l100_metas(ids)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = _load_existing_rows(OUT)
    done_keys = _done_keys(all_rows)
    print(
        f"[shape-L100] n_maps={len(metas)} tasks=100 methods={methods} "
        f"mode={'pure' if args.pure_engine else 'heavy'} "
        f"cached_m0={args.use_cached_m0} wall={args.wall} out={OUT} "
        f"resume={len(done_keys)} done",
        flush=True,
    )

    for meta in metas:
        sid = str(meta["id"])
        rows = run_scenario(
            meta,
            methods=methods,
            max_time=args.max_time,
            wall_timeout=args.wall,
            max_active_ecbs=args.max_active_ecbs,
            use_cached_m0=args.use_cached_m0,
            pure_engine=args.pure_engine,
            skip_done=done_keys,
        )
        for r in rows:
            mk = _row_method_key(r)
            key = (sid, mk)
            done_keys.add(key)
            all_rows = [
                x
                for x in all_rows
                if not (str(x.get("scenario_id")) == sid and _row_method_key(x) == mk)
            ]
            all_rows.append(r)
        _write_table(all_rows, OUT / "per_map.csv")
        (OUT / "per_map.json").write_text(
            json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    agg = _aggregate(all_rows)
    with (OUT / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        if agg:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)
    (OUT / "aggregate.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print_tables(all_rows, agg)
    print(f"\n[shape-L100] saved → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
