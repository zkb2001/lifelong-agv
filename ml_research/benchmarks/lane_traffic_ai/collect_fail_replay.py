"""Collect M0 path/assign-failure snapshots for BC replay."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from .config import RESULTS_DIR, ensure_dirs
from .features import build_park_artery_channels


def collect_from_sim_hook(sim, samples: List[dict], *, max_samples: int = 500) -> None:
    """Append one sample if assign_fail_streak > 0."""
    if len(samples) >= max_samples:
        return
    streak = int(getattr(sim, "_assign_fail_streak", 0) or 0)
    if streak <= 0:
        return
    failed_pos = []
    unloads = []
    for agv in getattr(sim, "agvs", []) or []:
        tid = getattr(agv, "task_id", None)
        if tid and not str(tid).startswith(("escape_", "park_", "recover_")):
            continue
        if agv.state is not None:
            failed_pos.append((int(agv.state[0]), int(agv.state[1])))
    for t in (getattr(sim, "surface_tasks", None) or {}).values():
        for e in t.get("end_points") or []:
            unloads.append((int(e[0]), int(e[1])))
    ch = build_park_artery_channels(
        sim, failed_positions=failed_pos, failed_unloads=unloads
    )
    samples.append(
        {
            "channels": ch,
            "goals": unloads[:4],
            "sem_pos": failed_pos[0] if failed_pos else None,
            "sem_goal": unloads[0] if unloads else None,
            "pad_free": 0.0,
            "path_clear": 0.0,
            "t": int(getattr(sim, "time", 0) or 0),
            "streak": streak,
        }
    )


def save_replay(samples: List[dict], path: Path) -> Path:
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, samples=np.asarray(samples, dtype=object))
    print(f"[REPLAY] saved {len(samples)} samples -> {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write empty replay scaffold (fill via sim hook in M0 runs)."
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "replay" / "fail_replay.npz"),
    )
    args = ap.parse_args()
    ensure_dirs()
    # synthetic seed samples so train_bc --replay always has something
    from .train_bc import _empty_walk_channels

    samples = []
    for i in range(32):
        unload = (3 + (i % 15), 3 + ((i * 3) % 15))
        fpos = (5 + (i % 12), 8 + ((i * 2) % 10))
        ch = _empty_walk_channels(failed_pos=[fpos], unloads=[unload], seed=i)
        # block some pads
        ch[8, unload[1], unload[0]] = 1.0
        ch[3, fpos[1], :] = 0.4 * ch[1, fpos[1], :]
        samples.append(
            {
                "channels": ch,
                "goals": [unload],
                "sem_pos": fpos,
                "sem_goal": unload,
                "pad_free": 0.0,
                "path_clear": 0.0,
            }
        )
    save_replay(samples, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
