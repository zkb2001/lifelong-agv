"""Probe open-dense capacity (n>=20, near<40 → cap7)."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

BASE = {12: 1558, 13: 832, 20: 883, 2: 2021, 11: 1707, 9: 1337}
SLOTS = [13, 12, 20, 2, 11, 9]


def main() -> int:
    for slot in SLOTS:
        meta = dict(
            load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        )
        meta["id"] = f"SH{slot:02d}_od"
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
