"""Probe open-floor SoftGuidance (n 12-24, near<55, λ=0.10)."""
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
    9: 1337,
    11: 2384,
    12: 1558,
    13: 832,
    15: 2523,
    17: 2395,
}
BASE = {
    1: 919,
    2: 2021,
    3: 3743,
    8: 1285,
    9: 1337,
    11: 1707,
    12: 1558,
    13: 832,
    15: 2358,
    17: 2395,
}
SLOTS = [2, 9, 13, 1, 3, 8, 12, 11, 15, 17]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_sg"
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
        ok = bool(rep.get("validate_ok")) and int(rep.get("tasks_completed") or 0) == 100
        print(
            f"SH{slot:02d} sim={sim} FULL={FULL[slot]} BASE={BASE[slot]} "
            f"dFULL={sim - FULL[slot]:+d} dBASE={sim - BASE[slot]:+d} "
            f"ok={ok} soft={rep.get('lambda_soft')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
