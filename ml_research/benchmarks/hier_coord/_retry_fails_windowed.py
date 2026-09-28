"""Delete FAIL result JSONs and re-run with overwrite (windowed ECBS sync fix)."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ml_research.benchmarks.hier_coord.bench_hier_maps_1k import (
    _run_one,
    rebuild_summary,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "hier_coord" / "bench_maps_1k"


def _job_from_fail(row: dict) -> dict | None:
    scene_id = str(row["id"])
    scene = OUT / "scenes" / scene_id
    meta_path = scene / f"{scene_id}.json"
    if not meta_path.exists():
        return None
    blob = json.loads(meta_path.read_text(encoding="utf-8"))
    obstacles = blob.get("extra_obstacles") or blob.get("obstacles")
    if obstacles is None and row.get("source") != "SH":
        return None
    return {
        "out_root": str(OUT),
        "scene_id": scene_id,
        "source": row.get("source") or "R",
        "style": row.get("style"),
        "n_agvs": int(row.get("n_agvs") or 8),
        "n_tasks": int(row.get("n_tasks") or 12),
        "pattern": row.get("pattern"),
        "seed": int(row.get("seed") or 0),
        "slot": row.get("slot"),
        "obstacles": obstacles,
        "time_limit": 18.0,
        "wall_timeout": 90.0,
        "plan_horizon": 24,
        "exec_horizon": 12,
        "overwrite": True,
    }


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 4) // 2)
    rows = []
    for p in (OUT / "results").glob("*.json"):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    fails = [r for r in rows if r.get("status") != "PASS"]
    jobs = []
    for r in fails:
        j = _job_from_fail(r)
        if j is None:
            print(f"skip no scene {r.get('id')}", flush=True)
            continue
        jobs.append(j)
    print(f"[retry] fails={len(fails)} jobs={len(jobs)} workers={workers}", flush=True)
    t0 = time.perf_counter()
    n_pass = n_fail = n_err = 0
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            try:
                row = _run_one(job)
                if row.get("status") == "PASS":
                    n_pass += 1
                else:
                    n_fail += 1
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                print(f"[error] {job['scene_id']}: {exc}", flush=True)
            if i % 10 == 0 or i == len(jobs):
                print(
                    f"progress {i}/{len(jobs)} pass={n_pass} fail={n_fail} err={n_err}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    row = fut.result()
                    if row.get("status") == "PASS":
                        n_pass += 1
                    else:
                        n_fail += 1
                except Exception as exc:  # noqa: BLE001
                    n_err += 1
                    print(f"[error] {exc}", flush=True)
                if done % 10 == 0 or done == len(jobs):
                    print(
                        f"progress {done}/{len(jobs)} pass={n_pass} fail={n_fail} err={n_err}",
                        flush=True,
                    )
    s = rebuild_summary(OUT)
    print(
        f"RETRY pass {n_pass} fail {n_fail} err {n_err} "
        f"recover_rate {n_pass / max(1, len(jobs)):.3f} "
        f"elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    print(
        f"FULL {{'n': {s['n']}, 'n_pass': {s['n_pass']}, 'n_fail': {s['n_fail']}, "
        f"'n_error': {s['n_error']}, 'pass_rate': {s['pass_rate']}}}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
