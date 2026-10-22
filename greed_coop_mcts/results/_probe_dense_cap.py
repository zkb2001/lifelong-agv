"""Probe dense-fleet throat cap on SH11/12 + sanity gates."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

BASE = {11: 2138, 12: 1558, 2: 2021, 8: 1285, 13: 832, 20: 883}
SLOTS = [11, 12, 2, 8, 13, 20]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_dc"
        meta["n_tasks"] = 100
        rep = solve(
            slot,
            meta=meta,
            policy="rrsc",
            use_stage_net=True,
            device="cpu",
            wall_timeout=600.0,
            max_sim_time=200000,
        )
        sim = int(rep.get("sim_time") or -1)
        print(
            f"SH{slot:02d} sim={sim} base={BASE[slot]} dlt={sim - BASE[slot]:+d} "
            f"fleet={rep.get('max_fleet_busy')} ok={rep.get('validate_ok')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
