"""Batch-test hierarchical MAPD on 1000+ unique maps.

Generates connected warehouse maps (plus SH01-20 exports), runs
``solve_ecbs(..., use_hierarchical=True, use_wavenet=True)``, and reports
completion / validate_ok. Supports resume + multiprocessing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.hier_coord.scene_difficulty.map_gen import gen_map
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import (
    gen_stress_tasks,
    write_scene_bundle,
)
from ml_research.common.paths import RESULTS

OUT = RESULTS / "hier_coord" / "bench_maps_1k"
STYLES = (
    "open",
    "sprinkle",
    "corridor_mild",
    "corridor_hard",
    "hutong",
    "maze",
)
PATTERNS = (
    "uniform",
    "same_pickup_10",
    "same_dropoff_10",
    "cross_flow",
    "fifo_deep",
)


def _result_path(out_root: Path, scene_id: str) -> Path:
    return out_root / "results" / f"{scene_id}.json"


def _pass_row(row: dict) -> bool:
    return (
        str(row.get("status")) == "PASS"
        and float(row.get("completion_ratio") or 0) >= 0.999
        and bool(row.get("validate_ok"))
        and not (row.get("tasks_failed") or [])
    )


def _run_one(job: dict) -> dict:
    out_root = Path(job["out_root"])
    scene_id = str(job["scene_id"])
    rp = _result_path(out_root, scene_id)
    if rp.exists() and not job.get("overwrite"):
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    t0 = time.perf_counter()
    row: Dict[str, Any] = {
        "id": scene_id,
        "source": job.get("source"),
        "style": job.get("style"),
        "n_agvs": job.get("n_agvs"),
        "n_tasks": job.get("n_tasks"),
        "pattern": job.get("pattern"),
        "seed": job.get("seed"),
    }
    try:
        if job.get("source") == "SH":
            slot = int(job["slot"])
            meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=float(job.get("wall_timeout") or 90.0))
            if not meta:
                row.update({"status": "MISSING", "error": "no_meta"})
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(json.dumps(row, indent=2), encoding="utf-8")
                return row
            meta["wall_timeout"] = float(job.get("wall_timeout") or 90.0)
            rep = solve_ecbs(
                slot=slot,
                meta=meta,
                max_tasks=int(job["n_tasks"]),
                max_active=0,
                weight=1.5,
                time_limit=float(job["time_limit"]),
                plan="joint",
                pipeline=True,
                turn_aware=True,
                initial_park=False,
                use_hierarchical=True,
                use_wavenet=True,
                hierarchical_force=False,
                wave_hard_cap=4,
                use_traffic_accel=False,
                use_lane_rules=False,
                plan_horizon=int(job.get("plan_horizon") or 24),
                exec_horizon=int(job.get("exec_horizon") or 12),
            )
        else:
            obstacles = job["obstacles"]
            bundle = out_root / "scenes" / scene_id
            meta_b = write_scene_bundle(
                bundle,
                scene_id=scene_id,
                obstacles=obstacles,
                n_agvs=int(job["n_agvs"]),
                tasks=gen_stress_tasks(
                    str(job["pattern"]), int(job["n_tasks"]), int(job["seed"])
                ),
                meta_extra={
                    "style": job.get("style"),
                    "source": "R",
                    "seed": job.get("seed"),
                },
            )
            meta = {
                "id": scene_id,
                "slot": 0,
                "task_csv": meta_b["task_csv"],
                "position_csv": meta_b["position_csv"],
                "extra_obstacles": obstacles,
                "n_tasks": int(job["n_tasks"]),
                "n_agvs": int(job["n_agvs"]),
                "max_sim_time": 300000,
                "wall_timeout": float(job.get("wall_timeout") or 90.0),
            }
            rep = solve_ecbs(
                slot=0,
                meta=meta,
                max_tasks=0,
                max_active=0,
                weight=1.5,
                time_limit=float(job["time_limit"]),
                plan="joint",
                pipeline=True,
                turn_aware=True,
                initial_park=False,
                use_hierarchical=True,
                use_wavenet=True,
                hierarchical_force=False,
                wave_hard_cap=4,
                use_traffic_accel=False,
                use_lane_rules=False,
                plan_horizon=int(job.get("plan_horizon") or 24),
                exec_horizon=int(job.get("exec_horizon") or 12),
            )

        cr = float(rep.get("completion_ratio") or 0.0)
        ok = bool(rep.get("validate_ok"))
        failed = list(rep.get("tasks_failed") or [])
        hier = rep.get("hierarchical") or {}
        status = (
            "PASS"
            if ok and cr >= 0.999 and not failed
            else "FAIL"
        )
        row.update(
            {
                "status": status,
                "completion_ratio": cr,
                "validate_ok": ok,
                "tasks_completed": rep.get("tasks_completed"),
                "tasks_total": rep.get("tasks_total"),
                "tasks_failed": failed,
                "sim_time": rep.get("sim_time"),
                "wall_seconds": round(time.perf_counter() - t0, 2),
                "waves_astar": hier.get("waves_astar"),
                "waves_ecbs": hier.get("waves_ecbs"),
                "waves_escalated": hier.get("waves_escalated"),
                "last_planner": hier.get("last_planner"),
                "last_reason": hier.get("last_reason"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "status": "ERROR",
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc()[-1200:],
                "wall_seconds": round(time.perf_counter() - t0, 2),
            }
        )

    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[{row.get('status')}] {scene_id} cr={row.get('completion_ratio')} "
        f"wall={row.get('wall_seconds')}s",
        flush=True,
    )
    return row


def schedule_jobs(
    *,
    out_root: Path,
    n_maps: int,
    seed: int,
    n_agvs: int,
    n_tasks: int,
    time_limit: float,
    include_sh: bool,
    styles: Sequence[str],
    patterns: Sequence[str],
) -> List[dict]:
    rng = random.Random(seed)
    jobs: List[dict] = []
    seen_obs = set()

    if include_sh:
        for slot in range(1, 21):
            sid = f"SH{slot:02d}_t{n_tasks}"
            jobs.append(
                {
                    "out_root": str(out_root),
                    "scene_id": sid,
                    "source": "SH",
                    "slot": slot,
                    "style": f"SH{slot:02d}",
                    "n_agvs": None,
                    "n_tasks": int(n_tasks),
                    "pattern": "export",
                    "seed": seed + slot,
                    "time_limit": float(time_limit),
                    "wall_timeout": float(time_limit) * 5.0,
                    "overwrite": False,
                }
            )

    i = 0
    attempts = 0
    while len([j for j in jobs if j.get("source") != "SH"]) < int(n_maps):
        attempts += 1
        if attempts > int(n_maps) * 40:
            break
        style = styles[i % len(styles)]
        s = seed + 10_000 + i
        i += 1
        g = gen_map(style, s)
        if not g.get("connected"):
            continue
        obs = g["obstacles"]
        key = json.dumps(obs)
        if key in seen_obs:
            continue
        seen_obs.add(key)
        pattern = rng.choice(list(patterns))
        sid = f"R_{style}{s}_a{n_agvs}_t{n_tasks}_{pattern}"
        jobs.append(
            {
                "out_root": str(out_root),
                "scene_id": sid,
                "source": "R",
                "style": style,
                "obstacles": obs,
                "n_agvs": int(n_agvs),
                "n_tasks": int(n_tasks),
                "pattern": pattern,
                "seed": int(s),
                "time_limit": float(time_limit),
                "wall_timeout": float(time_limit) * 5.0,
                "overwrite": False,
            }
        )
    return jobs


def rebuild_summary(out_root: Path) -> dict:
    rows = []
    rdir = out_root / "results"
    if rdir.exists():
        for p in sorted(rdir.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
    n = len(rows)
    n_pass = sum(1 for r in rows if _pass_row(r))
    n_fail = sum(1 for r in rows if r.get("status") == "FAIL")
    n_err = sum(1 for r in rows if r.get("status") == "ERROR")
    n_miss = sum(1 for r in rows if r.get("status") == "MISSING")
    by_style: Dict[str, Dict[str, int]] = {}
    fails = []
    for r in rows:
        st = str(r.get("style") or "?")
        bucket = by_style.setdefault(st, {"n": 0, "pass": 0, "fail": 0, "error": 0})
        bucket["n"] += 1
        if _pass_row(r):
            bucket["pass"] += 1
        elif r.get("status") == "ERROR":
            bucket["error"] += 1
        else:
            bucket["fail"] += 1
        if not _pass_row(r):
            fails.append(
                {
                    "id": r.get("id"),
                    "status": r.get("status"),
                    "cr": r.get("completion_ratio"),
                    "failed": r.get("tasks_failed"),
                    "error": r.get("error"),
                    "style": r.get("style"),
                }
            )
    summary = {
        "n": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_err,
        "n_missing": n_miss,
        "pass_rate": round(n_pass / max(1, n), 4),
        "by_style": by_style,
        "fails": fails[:200],
        "n_fails_listed": min(200, len(fails)),
        "n_fails_total": len(fails),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Hierarchical MAPD bench on 1000+ maps")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--n-maps", type=int, default=1000, help="random unique maps")
    ap.add_argument("--n-agvs", type=int, default=8)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--no-sh", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        s = rebuild_summary(out_root)
        print(
            f"[summary] n={s['n']} pass={s['n_pass']} fail={s['n_fail']} "
            f"error={s['n_error']} rate={s['pass_rate']}",
            flush=True,
        )
        return 0 if s["n_fail"] == 0 and s["n_error"] == 0 else 1

    jobs = schedule_jobs(
        out_root=out_root,
        n_maps=int(args.n_maps),
        seed=int(args.seed),
        n_agvs=int(args.n_agvs),
        n_tasks=int(args.n_tasks),
        time_limit=float(args.time_limit),
        include_sh=not bool(args.no_sh),
        styles=STYLES,
        patterns=PATTERNS,
    )
    # skip already done
    todo = [j for j in jobs if not _result_path(out_root, j["scene_id"]).exists()]
    print(
        f"[bench] total_jobs={len(jobs)} todo={len(todo)} workers={args.workers} "
        f"agv={args.n_agvs} tasks={args.n_tasks} tl={args.time_limit}",
        flush=True,
    )
    (out_root / "jobs_manifest.json").write_text(
        json.dumps(
            {
                "n_jobs": len(jobs),
                "n_todo": len(todo),
                "n_agvs": args.n_agvs,
                "n_tasks": args.n_tasks,
                "time_limit": args.time_limit,
                "seed": args.seed,
                "ids": [j["scene_id"] for j in jobs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    done = 0
    if int(args.workers) <= 1:
        for job in todo:
            _run_one(job)
            done += 1
            if done % 25 == 0:
                rebuild_summary(out_root)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(_run_one, job) for job in todo]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] worker {exc}", flush=True)
                done += 1
                if done % 25 == 0:
                    s = rebuild_summary(out_root)
                    print(
                        f"[progress] {done}/{len(todo)} "
                        f"pass_so_far={s['n_pass']}/{s['n']} rate={s['pass_rate']}",
                        flush=True,
                    )

    s = rebuild_summary(out_root)
    print(
        f"[done] n={s['n']} PASS={s['n_pass']} FAIL={s['n_fail']} "
        f"ERROR={s['n_error']} pass_rate={s['pass_rate']}",
        flush=True,
    )
    print(f"[done] summary={out_root / 'summary.json'}", flush=True)
    return 0 if s["n_fail"] == 0 and s["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
