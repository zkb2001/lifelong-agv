"""Rebuild near-horizon obs/feats and wave-ST labels (train == deploy).

For each sample:
1. Load scene obstacles + task/position CSVs
2. Relabel with first-wave prioritized spacetime A* (not full-episode M0 CR)
3. Rebuild map obs + 8-d feats from near-horizon task window (~2 * n_agvs)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ml_research.benchmarks.hier_coord.scene_difficulty.label_m0 import (
    build_label_obs,
    label_first_wave_st,
)
from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
    SCENE_HORIZON_MULT,
    burst_from_task_rows,
    build_scene_feats,
    scene_horizon_n,
    take_horizon_task_rows,
)
from ml_research.common.paths import RESULTS

DEFAULT_TRAIN = RESULTS / "hier_coord" / "scene_difficulty"
DEFAULT_TEST = RESULTS / "hier_coord" / "scene_difficulty_test"


def _load_tasks(task_csv: Path) -> list:
    rows = []
    with open(task_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _load_obstacles(meta: dict) -> list:
    pos = Path(meta.get("position_csv") or "")
    sid = str(meta.get("id") or "")
    candidates = []
    if pos.parent.exists():
        candidates.append(pos.parent / f"{sid}.json")
        candidates.append(pos.parent / "scene.json")
    for c in candidates:
        if c.exists():
            blob = json.loads(c.read_text(encoding="utf-8"))
            obs = blob.get("extra_obstacles") or blob.get("obstacles") or []
            if obs:
                return list(obs)
    # fallback: empty (open map) — rare
    return []


def refresh_one(sample_dir: str) -> dict:
    sample_dir_p = Path(sample_dir)
    meta_p = sample_dir_p / "meta.json"
    obs_p = sample_dir_p / "obs.npz"
    if not meta_p.exists():
        return {"ok": False, "reason": "no_meta"}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    task_csv = Path(meta.get("task_csv") or "")
    pos_csv = Path(meta.get("position_csv") or "")
    if not task_csv.exists() or not pos_csv.exists():
        return {"ok": False, "id": meta.get("id"), "reason": "missing_csv"}

    tasks = _load_tasks(task_csv)
    obstacles = _load_obstacles(meta)
    n_agvs = int(meta.get("n_agvs") or 8)
    n_obs = int(meta.get("n_obstacles") or len(obstacles))

    # Preserve episode labels
    if "label_episode" not in meta:
        meta["label_episode"] = meta.get("label")
        meta["label_int_episode"] = meta.get("label_int")
        meta["completion_ratio_episode"] = meta.get("completion_ratio")

    wave = label_first_wave_st(
        obstacles=obstacles,
        position_csv=pos_csv,
        task_rows=tasks,
        n_agvs=n_agvs,
    )
    meta["label"] = wave["label"]
    meta["label_int"] = int(wave["label_int"])
    meta["label_rule"] = wave["label_rule"]
    meta["wave_ok"] = bool(wave.get("wave_ok"))
    meta["wave_k"] = int(wave.get("wave_k") or 0)
    meta["label_policy"] = "first_wave_st"

    hz_rows = take_horizon_task_rows(tasks, n_agvs)
    pk_n, dr_n, pk_sh, dr_sh = burst_from_task_rows(hz_rows)
    hz = scene_horizon_n(n_agvs, len(tasks))
    feats = build_scene_feats(
        n_tasks=hz,
        n_agvs=n_agvs,
        n_obstacles=n_obs,
        queue_depth=hz,
        same_pickup_burst=pk_n,
        same_dropoff_burst=dr_n,
        pickup_share=pk_sh,
        dropoff_share=dr_sh,
    )
    maps = build_label_obs(
        obstacles=obstacles,
        position_csv=pos_csv,
        task_rows=tasks,
        n_agvs=n_agvs,
    )
    np.savez_compressed(obs_p, maps=maps, feats=feats)
    meta["feat_pk_burst"] = pk_n
    meta["feat_dr_burst"] = dr_n
    meta["feat_horizon"] = hz
    meta["feat_refresh"] = f"v3_horizon_x{SCENE_HORIZON_MULT}_wave_st"
    meta["n_obstacles"] = n_obs
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "ok": True,
        "id": meta.get("id"),
        "label": meta["label"],
        "wave_ok": meta["wave_ok"],
        "flipped": int(meta.get("label_int_episode", meta["label_int"])) != int(meta["label_int"]),
    }


def refresh_root(root: Path, *, workers: int) -> dict:
    samples = root / "samples"
    if not samples.exists():
        return {"n": 0, "ok": 0, "easy": 0, "hard": 0, "flip": 0}
    dirs = [d for d in sorted(samples.iterdir()) if d.is_dir()]
    ok = easy = hard = flip = 0
    if workers <= 1:
        results = [refresh_one(str(d)) for d in dirs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(refresh_one, str(d)) for d in dirs]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 200 == 0:
                    print(f"[refresh] {root.name} progress {i}/{len(dirs)}", flush=True)
    for r in results:
        if not r.get("ok"):
            continue
        ok += 1
        if r.get("label") == "easy":
            easy += 1
        else:
            hard += 1
        flip += int(bool(r.get("flipped")))
    return {"n": len(dirs), "ok": ok, "easy": easy, "hard": hard, "flip": flip}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Near-horizon feats + wave-ST relabel")
    ap.add_argument("--data", type=str, default=str(DEFAULT_TRAIN))
    ap.add_argument("--test-data", type=str, default=str(DEFAULT_TEST))
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args(list(argv) if argv is not None else None)

    for name, root in (("train", Path(args.data)), ("test", Path(args.test_data))):
        if name == "test" and args.skip_test:
            continue
        print(f"[refresh] {name} workers={args.workers} …", flush=True)
        st = refresh_root(root, workers=int(args.workers))
        print(
            f"[refresh] {name}: ok={st['ok']}/{st['n']} "
            f"easy={st['easy']} hard={st['hard']} flip_vs_episode={st['flip']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
