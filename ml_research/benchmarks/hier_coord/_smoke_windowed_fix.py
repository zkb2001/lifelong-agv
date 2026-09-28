"""Smoke re-run a few FAIL scenes after windowed-ECBS timeline sync fix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ml_research.benchmarks.hier_coord.bench_hier_maps_1k import _run_one

OUT = Path(__file__).resolve().parents[2] / "results" / "hier_coord" / "bench_maps_1k"

IDS = [
    "R_maze20270916_a8_t12_same_pickup_10",
    "R_maze20270922_a8_t12_same_pickup_10",
    "R_maze20270988_a8_t12_cross_flow",
    "R_corridor_hard20271052_a8_t12_same_dropoff_10",
    "R_maze20270928_a8_t12_cross_flow",
]


def _job_from_result(scene_id: str) -> dict:
    row = json.loads((OUT / "results" / f"{scene_id}.json").read_text(encoding="utf-8"))
    scene = OUT / "scenes" / scene_id
    obstacles = None
    meta_path = scene / f"{scene_id}.json"
    if meta_path.exists():
        blob = json.loads(meta_path.read_text(encoding="utf-8"))
        obstacles = blob.get("extra_obstacles") or blob.get("obstacles")
    return {
        "out_root": str(OUT),
        "scene_id": scene_id,
        "source": row.get("source") or "R",
        "style": row.get("style") or "maze",
        "n_agvs": int(row.get("n_agvs") or 8),
        "n_tasks": int(row.get("n_tasks") or 12),
        "pattern": row.get("pattern") or "cross_flow",
        "seed": int(row.get("seed") or 0),
        "obstacles": obstacles,
        "time_limit": 18.0,
        "wall_timeout": 90.0,
        "plan_horizon": 24,
        "exec_horizon": 12,
        "overwrite": True,
    }


def main() -> int:
    ids = sys.argv[1:] or IDS
    for sid in ids:
        rp = OUT / "results" / f"{sid}.json"
        if not rp.exists():
            print(f"SKIP missing {sid}", flush=True)
            continue
        job = _job_from_result(sid)
        if job["obstacles"] is None and job["source"] != "SH":
            print(f"SKIP no obstacles {sid}", flush=True)
            continue
        print(f"=== RUN {sid} ===", flush=True)
        row = _run_one(job)
        print(
            f">>> {sid} status={row.get('status')} cr={row.get('completion_ratio')} "
            f"val={row.get('validate_ok')} failed={row.get('tasks_failed')} "
            f"wall={row.get('wall_seconds')} done={row.get('tasks_completed')}/"
            f"{row.get('tasks_total')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
