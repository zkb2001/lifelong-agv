"""Build large labeled scene-difficulty dataset (map × task × AGV combos).

Labels come from M0 spacetime-A*: finish => easy, else hard.
Designed for thousands of samples with resume + optional multiprocessing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml_research.benchmarks.hier_coord.scene_difficulty.label_m0 import (
    build_label_obs,
    label_first_wave_st,
    label_with_m0,
)
from ml_research.benchmarks.hier_coord.scene_difficulty.map_gen import (
    gen_map,
    harvest_existing_maps,
)
from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
    burst_from_task_rows,
    build_scene_feats,
    scene_horizon_n,
    take_horizon_task_rows,
)
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import (
    TASK_PATTERNS,
    gen_stress_tasks,
    write_scene_bundle,
)
from ml_research.common.paths import RESULTS

OUT = RESULTS / "hier_coord" / "scene_difficulty"

DEFAULT_AGVS = (4, 8, 12, 16, 20, 24, 30, 40, 50)
DEFAULT_TASKS = (8, 12, 20)
DEFAULT_PATTERNS = (
    "uniform",
    "same_pickup_10",
    "same_pickup_20",
    "same_dropoff_10",
    "same_dropoff_30",
    "pickup10_dropoff30",
    "fifo_deep",
    "cross_flow",
)
DEFAULT_STYLES = (
    "open",
    "sprinkle",
    "corridor_mild",
    "corridor_hard",
    "hutong",
    "maze",
)


def _burst_stats(pattern: str) -> Tuple[int, int]:
    p = pattern.lower()
    if p.startswith("same_pickup_"):
        return int(p.split("_")[-1]), 0
    if p.startswith("same_dropoff_"):
        return 0, int(p.split("_")[-1])
    if p in ("pickup10_dropoff30", "dual_stress"):
        return 10, 30
    return 0, 0


def _save_sample(sample_dir: Path, sample: dict, maps: np.ndarray, feats: np.ndarray) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_dir / "obs.npz", maps=maps, feats=feats)
    (sample_dir / "meta.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _scene_id(
    *,
    source: str,
    map_key: str,
    pattern: str,
    n_tasks: int,
    n_agvs: int,
    seed: int,
) -> str:
    # compact stable id
    return f"{source}_{map_key}_a{n_agvs}_t{n_tasks}_{pattern}_s{seed}"


def build_one_job(job: dict) -> dict:
    """Worker entry: label one combo. Always writes its own position CSV for n_agvs."""
    out_root = Path(job["out_root"])
    scene_id = job["scene_id"]
    sample_meta = out_root / "samples" / scene_id / "meta.json"
    if sample_meta.exists() and not job.get("overwrite"):
        return json.loads(sample_meta.read_text(encoding="utf-8"))

    obstacles = job["obstacles"]
    pattern = job["pattern"]
    n_tasks = int(job["n_tasks"])
    n_agvs = int(job["n_agvs"])
    seed = int(job["seed"])
    tasks = gen_stress_tasks(pattern, n_tasks, seed)
    bundle_dir = out_root / "scenes" / scene_id
    meta = write_scene_bundle(
        bundle_dir,
        scene_id=scene_id,
        obstacles=obstacles,
        n_agvs=n_agvs,
        tasks=tasks,
        meta_extra={
            "style": job.get("style"),
            "source": job.get("source"),
            "task_pattern": pattern,
            "seed": seed,
            "map_key": job.get("map_key"),
        },
    )
    # IMPORTANT: use the bundle position CSV so AGV count matches the job
    pos = Path(meta["position_csv"])
    task_csv = Path(meta["task_csv"])
    max_time = int(job["max_time"])
    # Scale budget lightly with fleet/tasks (hard maps still fail fast via wall)
    max_time = min(int(job.get("max_time_cap", 1500)), max_time + 15 * max(0, n_agvs - 8) + 10 * max(0, n_tasks - 12))
    wall = float(job["wall"])

    t0 = time.perf_counter()
    # Episode M0 kept for diagnostics; primary label = first-wave ST-A*
    lab_ep = label_with_m0(
        task_csv=task_csv,
        position_csv=pos,
        obstacles=obstacles,
        scenario_tag=scene_id,
        max_time=max_time,
        wall_timeout=wall,
        save_trajectory=False,
    )
    lab = label_first_wave_st(
        obstacles=obstacles,
        position_csv=pos,
        task_rows=tasks,
        n_agvs=n_agvs,
    )
    maps = build_label_obs(
        obstacles=obstacles,
        position_csv=pos,
        task_rows=tasks,
        n_agvs=n_agvs,
    )
    hz_rows = take_horizon_task_rows(tasks, n_agvs)
    pk_b, dr_b, pk_sh, dr_sh = burst_from_task_rows(hz_rows)
    hz = scene_horizon_n(n_agvs, n_tasks)
    feats = build_scene_feats(
        n_tasks=hz,
        n_agvs=n_agvs,
        n_obstacles=len(obstacles),
        queue_depth=hz,
        same_pickup_burst=pk_b,
        same_dropoff_burst=dr_b,
        pickup_share=pk_sh,
        dropoff_share=dr_sh,
    )
    sample = {
        "id": scene_id,
        "style": job.get("style"),
        "source": job.get("source"),
        "map_key": job.get("map_key"),
        "task_pattern": pattern,
        "n_tasks": n_tasks,
        "n_agvs": n_agvs,
        "n_obstacles": len(obstacles),
        "label": lab["label"],
        "label_int": lab["label_int"],
        "label_rule": lab.get("label_rule"),
        "label_policy": "first_wave_st",
        "wave_ok": lab.get("wave_ok"),
        "wave_k": lab.get("wave_k"),
        "label_episode": lab_ep["label"],
        "label_int_episode": lab_ep["label_int"],
        "completion_ratio": lab_ep["completion_ratio"],
        "completion_ratio_episode": lab_ep["completion_ratio"],
        "forced_stop": lab_ep["forced_stop"],
        "sim_time": lab_ep["sim_time"],
        "wall_label_seconds": round(time.perf_counter() - t0, 2),
        "position_csv": str(pos),
        "task_csv": str(task_csv),
        "seed": seed,
        "feat_horizon": hz,
    }
    _save_sample(out_root / "samples" / scene_id, sample, maps, feats)
    print(
        f"[label] {scene_id} wave={lab['label']} ep={lab_ep['label']} "
        f"cr={lab_ep['completion_ratio']:.3f} agv={n_agvs} tasks={n_tasks} "
        f"wave_ok={lab.get('wave_ok')}",
        flush=True,
    )
    return sample


def _exports_only(harvested: List[dict]) -> List[dict]:
    return [m for m in harvested if m.get("source") == "exports"]


def schedule_jobs(
    *,
    out_root: Path,
    target: int,
    seed: int,
    patterns: Sequence[str],
    styles: Sequence[str],
    agv_list: Sequence[int],
    task_list: Sequence[int],
    n_random_maps: int,
    max_time: int,
    wall: float,
    include_harvest: bool,
    include_curriculum: bool,
) -> List[dict]:
    rng = random.Random(seed)
    maps: List[dict] = []

    if include_harvest:
        harvested = harvest_existing_maps()
        exports = _exports_only(harvested)
        for m in exports:
            maps.append(
                {
                    "map_key": m["id"],
                    "obstacles": m["obstacles"],
                    "style": str(m.get("style") or m["id"]),
                    "source": "H",
                }
            )
        if include_curriculum:
            # subsample curriculum to avoid explosion
            curr = [m for m in harvested if m.get("source") == "curriculum_shape"]
            rng.shuffle(curr)
            for m in curr[:40]:
                maps.append(
                    {
                        "map_key": m["id"][:48],
                        "obstacles": m["obstacles"],
                        "style": str(m.get("style") or m["id"]),
                        "source": "C",
                    }
                )

    for i in range(int(n_random_maps)):
        style = styles[i % len(styles)]
        s = seed + 10_000 + i
        g = gen_map(style, s)
        maps.append(
            {
                "map_key": f"{style}{s}",
                "obstacles": g["obstacles"],
                "style": style,
                "source": "R",
            }
        )

    if not maps:
        raise RuntimeError("no maps scheduled")

    # Stratified coverage then fill
    jobs: List[dict] = []
    seen_ids = set()

    def add_job(m: dict, pattern: str, n_agvs: int, n_tasks: int, s: int) -> None:
        nonlocal jobs
        if len(jobs) >= target:
            return
        sid = _scene_id(
            source=m["source"],
            map_key=m["map_key"],
            pattern=pattern,
            n_tasks=n_tasks,
            n_agvs=n_agvs,
            seed=s,
        )
        if sid in seen_ids:
            return
        if (out_root / "samples" / sid / "meta.json").exists():
            # still count toward target via later index rebuild; skip queue
            seen_ids.add(sid)
            return
        seen_ids.add(sid)
        jobs.append(
            {
                "out_root": str(out_root),
                "scene_id": sid,
                "obstacles": m["obstacles"],
                "style": m["style"],
                "source": m["source"],
                "map_key": m["map_key"],
                "pattern": pattern,
                "n_tasks": int(n_tasks),
                "n_agvs": int(n_agvs),
                "seed": int(s),
                "max_time": int(max_time),
                "wall": float(wall),
                "overwrite": False,
            }
        )

    # Pass 1: full factorial on a compact core (ensures every AGV count appears)
    core_maps = maps[: max(12, min(30, len(maps)))]
    s = seed
    for m in core_maps:
        for n_agvs in agv_list:
            for n_tasks in task_list:
                for pattern in patterns:
                    s += 1
                    add_job(m, pattern, n_agvs, n_tasks, s)
                    if len(jobs) >= target:
                        return jobs

    # Pass 2: random combos until target
    while len(jobs) < target:
        m = rng.choice(maps)
        pattern = rng.choice(list(patterns))
        n_agvs = int(rng.choice(list(agv_list)))
        n_tasks = int(rng.choice(list(task_list)))
        s += 1
        before = len(jobs)
        add_job(m, pattern, n_agvs, n_tasks, s)
        if len(jobs) == before:
            # likely id collision; mutate seed
            s += 97
            add_job(m, pattern, n_agvs, n_tasks, s)
        if s > seed + target * 50:
            break
    return jobs


def rebuild_index(out_root: Path) -> dict:
    samples_dir = out_root / "samples"
    rows: List[dict] = []
    if samples_dir.exists():
        for d in sorted(samples_dir.iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                try:
                    rows.append(json.loads(meta.read_text(encoding="utf-8")))
                except Exception:
                    continue
    index = {
        "n": len(rows),
        "n_easy": sum(1 for r in rows if r.get("label") == "easy"),
        "n_hard": sum(1 for r in rows if r.get("label") == "hard"),
        "agv_hist": {},
        "pattern_hist": {},
        "rows": rows,
    }
    for r in rows:
        a = str(r.get("n_agvs"))
        p = str(r.get("task_pattern"))
        index["agv_hist"][a] = int(index["agv_hist"].get(a, 0)) + 1
        index["pattern_hist"][p] = int(index["pattern_hist"].get(p, 0)) + 1
    (out_root / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return index


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Large map×task×AGV difficulty dataset")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--target", type=int, default=3000, help="desired labeled samples")
    ap.add_argument("--n-random-maps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-time", type=int, default=500)
    ap.add_argument("--wall", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--skip-harvest", action="store_true")
    ap.add_argument("--include-curriculum", action="store_true")
    ap.add_argument(
        "--agvs",
        type=str,
        default=",".join(str(x) for x in DEFAULT_AGVS),
        help="AGV counts up to 50",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TASKS),
    )
    ap.add_argument(
        "--patterns",
        type=str,
        default=",".join(DEFAULT_PATTERNS),
    )
    ap.add_argument(
        "--styles",
        type=str,
        default=",".join(DEFAULT_STYLES),
    )
    ap.add_argument("--rebuild-index-only", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.rebuild_index_only:
        idx = rebuild_index(out_root)
        print(f"[index] n={idx['n']} easy={idx['n_easy']} hard={idx['n_hard']}", flush=True)
        return 0

    agv_list = [int(x) for x in args.agvs.split(",") if x.strip()]
    for a in agv_list:
        if a < 1 or a > 50:
            raise SystemExit(f"AGV count out of range 1..50: {a}")
    task_list = [int(x) for x in args.tasks.split(",") if x.strip()]
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    existing = rebuild_index(out_root)
    print(
        f"[resume] existing samples={existing['n']} easy={existing['n_easy']} "
        f"hard={existing['n_hard']}",
        flush=True,
    )
    need = max(0, int(args.target) - int(existing["n"]))
    if need <= 0:
        print(f"[done] already have {existing['n']} >= target {args.target}", flush=True)
        return 0

    jobs = schedule_jobs(
        out_root=out_root,
        target=need,
        seed=int(args.seed),
        patterns=patterns,
        styles=styles,
        agv_list=agv_list,
        task_list=task_list,
        n_random_maps=int(args.n_random_maps),
        max_time=int(args.max_time),
        wall=float(args.wall),
        include_harvest=not bool(args.skip_harvest),
        include_curriculum=bool(args.include_curriculum),
    )
    print(
        f"[schedule] new_jobs={len(jobs)} workers={args.workers} "
        f"agvs={agv_list} tasks={task_list} patterns={len(patterns)}",
        flush=True,
    )

    done = 0
    if int(args.workers) <= 1:
        for job in jobs:
            build_one_job(job)
            done += 1
            if done % 20 == 0:
                rebuild_index(out_root)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(build_one_job, job) for job in jobs]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[error] {exc}", flush=True)
                done += 1
                if done % 20 == 0:
                    idx = rebuild_index(out_root)
                    print(
                        f"[progress] labeled_batch={done}/{len(jobs)} "
                        f"total_on_disk={idx['n']}",
                        flush=True,
                    )

    idx = rebuild_index(out_root)
    print(
        f"[done] samples={idx['n']} easy={idx['n_easy']} hard={idx['n_hard']} "
        f"agv_hist={idx['agv_hist']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
