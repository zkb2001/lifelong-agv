"""A/B: hotspot lifelong-style batches — baseline A* vs ECBS ± wave hierarchical.

Generates 100-task CSVs where pickup and/or dropoff are concentrated, then
compares:
  1) pure spacetime-A* baseline (M0 / engine greedy)
  2) ECBS pipeline without hierarchical wave
  3) ECBS pipeline with hierarchical + WaveNet (dropoff/pickup hotspot gate)

Open warehouse map (no extra obstacles) so the failure mode is congestion at
shared stations, not maze topology.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.runner import run_m0
from ml_research.common.data_utils import load_main_copy
from ml_research.common.paths import POSITION_CSV, RESULTS

OUT = RESULTS / "lifelong" / "hotspot_wave_ab"
OUT.mkdir(parents=True, exist_ok=True)


def _station_names(position_csv: Path) -> Tuple[List[str], List[str]]:
    mod = load_main_copy(force_reload=True)
    sp, ep, _ = mod.get_object_position(str(position_csv))
    return sorted(sp.keys()), sorted(ep.keys())


def write_hotspot_tasks(
    path: Path,
    *,
    n: int,
    mode: str,
    pickups: Sequence[str],
    dropoffs: Sequence[str],
    pick: Optional[str] = None,
    drop: Optional[str] = None,
) -> dict:
    """mode: same_drop | same_pick | same_both | diverse"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pick = pick or pickups[0]
    drop = drop or dropoffs[0]
    rows: List[dict] = []
    for i in range(1, n + 1):
        if mode == "same_drop":
            # rotate pickups, fix dropoff
            pk = pickups[(i - 1) % len(pickups)]
            do = drop
        elif mode == "same_pick":
            pk = pick
            do = dropoffs[(i - 1) % len(dropoffs)]
        elif mode == "same_both":
            pk = pick
            do = drop
        elif mode == "diverse":
            pk = pickups[(i - 1) % len(pickups)]
            do = dropoffs[(i - 1) % len(dropoffs)]
        else:
            raise ValueError(mode)
        rows.append(
            {
                "task_id": f"{pk}-{i}",
                "start_point": pk,
                "end_point": do,
                "priority": "Normal",
                "remaining_time": "None",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "start_point",
                "end_point",
                "priority",
                "remaining_time",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return {
        "path": str(path),
        "n": n,
        "mode": mode,
        "pick": pick if mode in ("same_pick", "same_both") else "rotating",
        "drop": drop if mode in ("same_drop", "same_both") else "rotating",
    }


def _open_meta(task_csv: Path, position_csv: Path, tag: str) -> dict:
    return {
        "id": f"open_hotspot_{tag}",
        "slot": 0,
        "task_csv": str(task_csv.resolve()),
        "position_csv": str(Path(position_csv).resolve()),
        "extra_obstacles": [],
        "n_tasks": 0,
        "max_sim_time": 300000,
        "wall_timeout": 1e9,
        "map_json": None,
    }


def run_arm_baseline(
    task_csv: Path,
    position_csv: Path,
    *,
    max_time: int,
    wall_timeout: Optional[float],
    tag: str,
) -> dict:
    print(f"\n=== BASELINE A*  [{tag}] ===", flush=True)
    t0 = time.perf_counter()
    rep = run_m0(
        task_csv,
        position_csv=position_csv,
        max_time=max_time,
        wall_timeout=wall_timeout,
        scenario_tag=f"hotspot_{tag}",
        save_trajectory=True,
    )
    rep["wall_ab"] = round(time.perf_counter() - t0, 2)
    rep["arm"] = "baseline_astar"
    return rep


def run_arm_ecbs(
    task_csv: Path,
    position_csv: Path,
    *,
    tag: str,
    hierarchical: bool,
    max_tasks: int,
    max_active: int,
    time_limit: float,
) -> dict:
    label = "ecbs+hier" if hierarchical else "ecbs"
    print(f"\n=== {label.upper()}  [{tag}] ===", flush=True)
    meta = _open_meta(task_csv, position_csv, f"{tag}_{label}")
    t0 = time.perf_counter()
    rep = solve_ecbs(
        slot=0,
        meta=meta,
        max_tasks=max_tasks,
        max_active=max_active,
        weight=1.5,
        time_limit=time_limit,
        plan="joint",
        pipeline=True,
        turn_aware=True,
        initial_park=False,
        deliver_batch=0,
        task_csv=task_csv,
        use_hierarchical=hierarchical,
        use_wavenet=hierarchical,
        hierarchical_force=False,
        hierarchical_threshold=0.08,
        wave_hard_cap=4,
    )
    rep["wall_ab"] = round(time.perf_counter() - t0, 2)
    rep["arm"] = label
    return rep


def _fmt_row(mode: str, arms: Dict[str, dict]) -> str:
    parts = [f"{mode:<12}"]
    for key in ("baseline_astar", "ecbs", "ecbs+hier"):
        r = arms.get(key) or {}
        sim = r.get("sim_time", "-")
        wall = r.get("wall_seconds", r.get("wall_ab", "-"))
        cr = r.get("completion_ratio", "-")
        ok = r.get("validate_ok", r.get("conflict_free", ""))
        parts.append(f"sim={sim} wall={wall}s cr={cr} ok={ok}")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tasks", type=int, default=100)
    ap.add_argument("--max-active", type=int, default=6)
    ap.add_argument("--max-time", type=int, default=8000, help="baseline sim cap")
    ap.add_argument("--wall-timeout", type=float, default=180.0)
    ap.add_argument("--ecbs-time-limit", type=float, default=25.0)
    ap.add_argument("--position", type=Path, default=POSITION_CSV)
    ap.add_argument(
        "--modes",
        type=str,
        default="same_drop,same_pick,same_both,diverse",
        help="comma list of hotspot modes",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="only ECBS ± hierarchical (faster)",
    )
    ap.add_argument(
        "--skip-ecbs",
        action="store_true",
        help="only baseline A*",
    )
    args = ap.parse_args()

    pickups, dropoffs = _station_names(args.position)
    pick = "Tiger"
    drop = "Beijing"
    if pick not in pickups:
        pick = pickups[0]
    if drop not in dropoffs:
        drop = dropoffs[0]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary: dict = {
        "n_tasks": args.n_tasks,
        "pick": pick,
        "drop": drop,
        "position": str(args.position),
        "modes": {},
    }

    print(
        f"[HOTSPOT-AB] n={args.n_tasks} pick={pick} drop={drop} "
        f"modes={modes} max_active={args.max_active}",
        flush=True,
    )

    for mode in modes:
        task_csv = OUT / f"tasks_{mode}_n{args.n_tasks}.csv"
        meta_t = write_hotspot_tasks(
            task_csv,
            n=args.n_tasks,
            mode=mode,
            pickups=pickups,
            dropoffs=dropoffs,
            pick=pick,
            drop=drop,
        )
        print(f"\n######## mode={mode} csv={task_csv.name} ########", flush=True)
        arms: Dict[str, dict] = {}

        if not args.skip_baseline:
            arms["baseline_astar"] = run_arm_baseline(
                task_csv,
                args.position,
                max_time=args.max_time,
                wall_timeout=args.wall_timeout,
                tag=mode,
            )

        if not args.skip_ecbs:
            arms["ecbs"] = run_arm_ecbs(
                task_csv,
                args.position,
                tag=mode,
                hierarchical=False,
                max_tasks=args.n_tasks,
                max_active=args.max_active,
                time_limit=args.ecbs_time_limit,
            )
            arms["ecbs+hier"] = run_arm_ecbs(
                task_csv,
                args.position,
                tag=mode,
                hierarchical=True,
                max_tasks=args.n_tasks,
                max_active=args.max_active,
                time_limit=args.ecbs_time_limit,
            )

        # Compact deltas
        deltas = {}
        if "ecbs" in arms and "ecbs+hier" in arms:
            a, b = arms["ecbs"], arms["ecbs+hier"]
            if a.get("sim_time") and b.get("sim_time"):
                deltas["sim_time_hier_minus_ecbs"] = b["sim_time"] - a["sim_time"]
                deltas["sim_speedup_pct"] = round(
                    100.0 * (a["sim_time"] - b["sim_time"]) / max(1, a["sim_time"]), 2
                )
            if a.get("wall_seconds") and b.get("wall_seconds"):
                deltas["wall_hier_minus_ecbs"] = round(
                    float(b["wall_seconds"]) - float(a["wall_seconds"]), 2
                )
            ha = (b.get("hierarchical") or {}) if isinstance(b.get("hierarchical"), dict) else {}
            deltas["hier_runtime_triggers"] = ha.get("runtime_triggers")
            deltas["hier_deescalations"] = ha.get("deescalations")
            deltas["hier_waves_escalated"] = ha.get("waves_escalated")

        summary["modes"][mode] = {
            "task_meta": meta_t,
            "arms": {
                k: {
                    "sim_time": v.get("sim_time"),
                    "wall_seconds": v.get("wall_seconds"),
                    "completion_ratio": v.get("completion_ratio"),
                    "tasks_completed": v.get("tasks_completed"),
                    "tasks_total": v.get("tasks_total"),
                    "validate_ok": v.get("validate_ok"),
                    "conflict_free": v.get("conflict_free"),
                    "forced_stop": v.get("forced_stop"),
                    "n_conflicts": v.get("n_conflicts"),
                    "hierarchical": v.get("hierarchical"),
                }
                for k, v in arms.items()
            },
            "deltas_ecbs_vs_hier": deltas,
        }
        print(_fmt_row(mode, arms), flush=True)
        if deltas:
            print(f"  deltas ecbs→hier: {deltas}", flush=True)

    out_json = OUT / f"hotspot_wave_ab_n{args.n_tasks}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[HOTSPOT-AB] wrote {out_json}", flush=True)

    # Verdict helpers
    print("\n========== VERDICT ==========", flush=True)
    for mode, block in summary["modes"].items():
        arms = block["arms"]
        base = arms.get("baseline_astar") or {}
        ecbs = arms.get("ecbs") or {}
        hier = arms.get("ecbs+hier") or {}
        print(f"\n[{mode}]", flush=True)
        if base:
            print(
                f"  baseline: sim={base.get('sim_time')} "
                f"cr={base.get('completion_ratio')} "
                f"forced={base.get('forced_stop')} "
                f"conflicts={base.get('n_conflicts')}",
                flush=True,
            )
        if ecbs and hier:
            ds = block.get("deltas_ecbs_vs_hier") or {}
            faster = ds.get("sim_speedup_pct")
            print(
                f"  ecbs sim={ecbs.get('sim_time')} cr={ecbs.get('completion_ratio')} "
                f"| hier sim={hier.get('sim_time')} cr={hier.get('completion_ratio')} "
                f"| hier_sim_speedup%={faster} "
                f"(>0 => hierarchical finished fewer sim ticks)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
