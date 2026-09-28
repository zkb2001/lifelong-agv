"""Verify FIFO pickup order vs display queue on trajectory CSVs."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ml_research.common.paths import ROOT


def _nkey(tid: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"([0-9]+)", str(tid))]


def load_task_order(task_csv: Path) -> Dict[str, List[str]]:
    by_station: Dict[str, List[str]] = defaultdict(list)
    with task_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = row.get("start_point") or row.get("pickup_name") or row.get("start")
            tid = row.get("task_id") or row.get("task-id") or row.get("id")
            if sp and tid:
                by_station[str(sp)].append(str(tid))
    for sp in by_station:
        by_station[sp].sort(key=_nkey)
    return dict(by_station)


def analyze_fifo(
    task_csv: Path,
    traj_csv: Path,
    *,
    skip_task_ids: Optional[Set[str]] = None,
) -> dict:
    order_all = load_task_order(task_csv)
    skip_requested = {str(x) for x in (skip_task_ids or set()) if str(x).strip()}
    rows: List[dict] = []
    with traj_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pickups: List[Tuple[int, str, str]] = []  # t, station, task_id
    prev_loaded: Dict[str, bool] = {}
    task_info: Dict[str, dict] = {}
    with task_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row.get("task_id") or row.get("task-id")
            sp = row.get("start_point") or row.get("pickup_name")
            if tid and sp:
                task_info[str(tid)] = row

    station_for: Dict[str, str] = {}
    for sp, tids in order_all.items():
        for tid in tids:
            station_for[tid] = sp

    for r in sorted(rows, key=lambda x: (int(x["timestamp"]), x["name"])):
        name = r["name"]
        loaded = str(r.get("loaded", "")).lower() in ("true", "1", "yes")
        pl = prev_loaded.get(name, False)
        tid = str(r.get("task-id") or "").strip()
        if loaded and not pl and tid and tid.lower() != "nan":
            sp = station_for.get(tid) or task_info.get(tid, {}).get("start_point", "?")
            pickups.append((int(r["timestamp"]), str(sp), tid))
        prev_loaded[name] = loaded

    # A task_id may only legally rise once; duplicate edges are solver flicker.
    seen_tid: set = set()
    deduped: List[Tuple[int, str, str]] = []
    for t, sp, tid in sorted(pickups, key=lambda x: x[0]):
        if tid in seen_tid:
            continue
        seen_tid.add(tid)
        deduped.append((t, sp, tid))
    pickups = deduped

    # Only never-picked abandoned ids are removed from the legal FIFO sequence.
    picked = {tid for _, _, tid in pickups}
    skip = skip_requested - picked
    order = {
        sp: [tid for tid in seq if tid not in skip]
        for sp, seq in order_all.items()
    }
    order = {sp: seq for sp, seq in order.items() if seq}

    # Per-station: pickups must follow CSV order (after removing abandoned)
    fifo_violations: List[dict] = []
    next_idx: Dict[str, int] = {sp: 0 for sp in order}
    for t, sp, tid in sorted(pickups, key=lambda x: x[0]):
        seq = order.get(sp, [])
        idx = next_idx.get(sp, 0)
        if idx >= len(seq):
            continue
        expected = seq[idx]
        if tid != expected:
            fifo_violations.append(
                {"t": t, "station": sp, "expected": expected, "actual": tid}
            )
        while idx < len(seq) and seq[idx] != tid:
            idx += 1
        if idx < len(seq) and seq[idx] == tid:
            idx += 1
        next_idx[sp] = idx

    # Display: abandoned never-picked tasks are not shown as queue head
    completed: set = set(skip)
    display_mismatch: List[dict] = []
    for t, sp, tid in pickups:
        seq = order.get(sp, [])
        head = None
        for cand in seq:
            if cand not in completed:
                head = cand
                break
        if head and head != tid:
            display_mismatch.append(
                {"t": t, "station": sp, "display": head, "picked": tid}
            )
        completed.add(tid)

    return {
        "task_csv": str(task_csv),
        "traj_csv": str(traj_csv),
        "n_pickups": len(pickups),
        "fifo_violations": fifo_violations,
        "display_mismatch": display_mismatch,
        "ok": not fifo_violations and not display_mismatch,
        "skip_task_ids": sorted(skip),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", help="e.g. SH_custom_05 or path stem")
    ap.add_argument("--task-dir", type=Path, default=ROOT / "ml_research" / "map_editor" / "exports")
    ap.add_argument(
        "--traj-dir",
        type=Path,
        default=ROOT / "ml_research" / "results" / "curriculum_shape" / "paper_baseline_astar_custom" / "trajectories",
    )
    args = ap.parse_args()
    tag = args.scenario.replace(".csv", "")
    task_csv = args.task_dir / f"{tag}_task.csv"
    traj_csv = args.traj_dir / f"{tag}.csv"
    rep = analyze_fifo(task_csv, traj_csv)
    print(f"scenario={tag} pickups={rep['n_pickups']}")
    print(f"fifo_violations={len(rep['fifo_violations'])} display_mismatch={len(rep['display_mismatch'])} ok={rep['ok']}")
    for v in rep["fifo_violations"][:10]:
        print(f"  FIFO t={v['t']} {v['station']}: expected {v['expected']} got {v['actual']}")
    for v in rep["display_mismatch"][:10]:
        print(f"  DISP t={v['t']} {v['station']}: show {v['display']} picked {v['picked']}")


if __name__ == "__main__":
    main()
