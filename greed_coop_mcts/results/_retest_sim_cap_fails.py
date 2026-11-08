"""Retest hard maps that hit the probe sim-cap, with map-native limits."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "greed_coop_mcts" / "checkpoints" / "facn_k_v1.pt"
IDS = [10463, 10652, 10912, 11352, 11442, 11608, 12411]


def run(idx: int) -> dict:
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["GREED_SKIP_TRAJ"] = "1"
    os.environ["GREED_LIVELOCK_MODE"] = "full"
    os.environ["GREED_FACN_CKPT"] = str(CKPT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from greed_coop_mcts.solve import solve
    from greed_coop_mcts.synth_maps.generate import load_synth_meta

    meta = dict(load_synth_meta(idx))
    meta["n_tasks"] = 100
    t0 = time.perf_counter()
    r = solve(
        idx,
        meta=meta,
        policy="rrsc",
        use_facn=True,
        wall_timeout=600.0,
        max_sim_time=200000,
        device="cpu",
    )
    return {
        "id": f"SYN_{idx:05d}",
        "done": r.get("tasks_completed"),
        "total": r.get("tasks_total"),
        "sim": r.get("sim_time"),
        "valid": r.get("validate_ok"),
        "wall": round(time.perf_counter() - t0, 2),
        "timed_out": bool(r.get("timed_out") or r.get("wall_timeout")),
    }


def main() -> int:
    fails = 0
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(run, i) for i in IDS]
        for f in as_completed(futs):
            row = f.result()
            ok = row["done"] == row["total"] and bool(row["valid"])
            if not ok:
                fails += 1
            print(("OK" if ok else "FAIL"), json.dumps(row), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
