"""Probe CEBI n>=9 near>=95 on SH14 + gates + risk maps."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

BASE = {2: 2021, 8: 1285, 12: 1558, 14: 1216, 17: 2395, 19: 1437, 4: 1053, 10: 2070}
SLOTS = [2, 8, 14, 12, 17, 19, 4, 10]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_c95"
        meta["n_tasks"] = 100
        rep = solve(
            slot,
            meta=meta,
            policy="rrsc",
            use_stage_net=True,
            use_cgpn=False,
            device="cpu",
            wall_timeout=600.0,
            max_sim_time=200000,
        )
        sim = int(rep.get("sim_time") or -1)
        done = int(rep.get("tasks_completed") or 0)
        ok = bool(rep.get("validate_ok")) and done == 100
        print(
            f"SH{slot:02d} sim={sim} base={BASE[slot]} dlt={sim - BASE[slot]:+d} "
            f"ok={ok} lane={rep.get('n_lane_locks')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
