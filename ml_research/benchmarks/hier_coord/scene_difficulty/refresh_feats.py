"""Recompute obs.npz feats from task_csv (fixed normalization + queue burst)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
    burst_from_task_rows,
    build_scene_feats,
)
from ml_research.common.paths import RESULTS

DEFAULT_DATA = RESULTS / "hier_coord" / "scene_difficulty"


def _load_tasks(task_csv: Path) -> list:
    rows = []
    with open(task_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def refresh_one(sample_dir: Path) -> bool:
    meta_p = sample_dir / "meta.json"
    obs_p = sample_dir / "obs.npz"
    if not meta_p.exists() or not obs_p.exists():
        return False
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    task_csv = Path(meta.get("task_csv") or "")
    if not task_csv.exists():
        return False
    tasks = _load_tasks(task_csv)
    pk_n, dr_n, pk_sh, dr_sh = burst_from_task_rows(tasks)
    n_tasks = int(meta.get("n_tasks") or len(tasks))
    feats = build_scene_feats(
        n_tasks=n_tasks,
        n_agvs=int(meta.get("n_agvs") or 8),
        n_obstacles=int(meta.get("n_obstacles") or 0),
        queue_depth=n_tasks,
        same_pickup_burst=pk_n,
        same_dropoff_burst=dr_n,
        pickup_share=pk_sh,
        dropoff_share=dr_sh,
    )
    blob = np.load(obs_p)
    maps = blob["maps"]
    np.savez_compressed(obs_p, maps=maps, feats=feats)
    meta["feat_pk_burst"] = pk_n
    meta["feat_dr_burst"] = dr_n
    meta["feat_refresh"] = "v2_log_burst"
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--test-data", type=str, default=str(RESULTS / "hier_coord" / "scene_difficulty_test"))
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    for name, root in (("train", Path(args.data)), ("test", Path(args.test_data))):
        if name == "test" and args.skip_test:
            continue
        samples = root / "samples"
        if not samples.exists():
            print(f"[refresh] skip {samples}", flush=True)
            continue
        n = ok = 0
        for d in sorted(samples.iterdir()):
            if not d.is_dir():
                continue
            n += 1
            if refresh_one(d):
                ok += 1
        print(f"[refresh] {name}: updated {ok}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
