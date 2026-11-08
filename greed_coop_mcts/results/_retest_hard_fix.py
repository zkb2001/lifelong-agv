"""Retest previously-failing hard synth maps + SH17 after dense-maze fix."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "greed_coop_mcts" / "checkpoints" / "facn_k_v1.pt"


def run(item: tuple[str, int]) -> dict:
    kind, idx = item
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["GREED_SKIP_TRAJ"] = "1"
    os.environ["GREED_LIVELOCK_MODE"] = "full"
    os.environ["GREED_FACN_CKPT"] = str(CKPT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from greed_coop_mcts.solve import solve

    t0 = time.perf_counter()
    if kind == "SYN":
        from greed_coop_mcts.synth_maps.generate import load_synth_meta

        meta = dict(load_synth_meta(idx))
        meta["n_tasks"] = 100
        r = solve(
            idx,
            meta=meta,
            policy="rrsc",
            use_facn=True,
            wall_timeout=300.0,
            max_sim_time=40000,
            device="cpu",
        )
        tag = f"SYN_{idx:05d}"
    else:
        r = solve(
            idx,
            policy="rrsc",
            use_facn=True,
            wall_timeout=300.0,
            max_sim_time=20000,
            device="cpu",
        )
        tag = f"SH{idx:02d}"
    return {
        "id": tag,
        "done": r.get("tasks_completed"),
        "total": r.get("tasks_total"),
        "sim": r.get("sim_time"),
        "valid": r.get("validate_ok"),
        "wall": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    jobs = [
        ("SYN", 10003),
        ("SYN", 10007),
        ("SYN", 10030),
        ("SYN", 10054),
        ("SYN", 10124),
        ("SYN", 10138),
        ("SH", 17),
    ]
    fails = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            row = f.result()
            ok = row["done"] == row["total"] and bool(row["valid"])
            if not ok:
                fails += 1
            print(("OK" if ok else "FAIL"), json.dumps(row), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
