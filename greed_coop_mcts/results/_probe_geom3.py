"""Probe mid-TBC LO45 + dense-TBC n20/near75 + tiny CEBI near110."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

FULL = {
    1: 965,
    2: 2021,
    3: 4440,
    8: 1320,
    10: 2215,
    11: 2384,
    12: 1558,
    14: 1216,
    15: 2523,
    17: 2395,
    19: 1446,
}
BASE = {
    1: 919,
    2: 2021,
    3: 3743,
    8: 1285,
    10: 2070,
    11: 1707,
    12: 1558,
    14: 1216,
    15: 2358,
    17: 2395,
    19: 1437,
}
# Gates + targets + regression risks
SLOTS = [1, 2, 3, 8, 12, 17, 15, 11, 10, 14, 19]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_g3"
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
        ok = bool(rep.get("validate_ok")) and int(rep.get("tasks_completed") or 0) == 100
        print(
            f"SH{slot:02d} sim={sim} FULL={FULL[slot]} BASE={BASE[slot]} "
            f"dFULL={sim - FULL[slot]:+d} dBASE={sim - BASE[slot]:+d} "
            f"ok={ok} cap={rep.get('max_fleet_busy')} "
            f"lane={rep.get('n_lane_locks')} egress={rep.get('n_pad_egress')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
