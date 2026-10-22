"""Probe SMPP + dense region admit on critical maps."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

BASE = {
    1: 919, 2: 2021, 3: 3743, 8: 1285, 11: 1707, 12: 1558,
    14: 1216, 15: 2358, 17: 2395, 19: 1437,
}
SLOTS = [1, 2, 3, 8, 17, 12, 11, 14, 15, 19]


def main() -> int:
    for slot in SLOTS:
        meta = dict(
            load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        )
        meta["id"] = f"SH{slot:02d}_smpp"
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
        done = int(rep.get("tasks_completed") or 0)
        ok = bool(rep.get("validate_ok")) and done == 100
        print(
            f"SH{slot:02d} sim={sim} done={done} base={BASE[slot]} "
            f"dlt={sim - BASE[slot]:+d} ok={ok} "
            f"sticky={rep.get('n_sticky_retreat')} lane={rep.get('n_lane_locks')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
