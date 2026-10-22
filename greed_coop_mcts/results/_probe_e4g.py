"""Probe geometry-gated E4 (n>=19, near 70-85) — expect SH12 only."""
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
    11: 2384,
    12: 1558,
    15: 2523,
    17: 2395,
    18: 1711,
}
BASE = {
    1: 919,
    2: 2021,
    3: 3743,
    8: 1285,
    11: 1707,
    12: 1558,
    15: 2358,
    17: 2395,
    18: 1618,
}
SLOTS = [1, 2, 3, 8, 12, 11, 15, 17, 18]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_e4g"
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
            f"ok={ok} e4={rep.get('n_e4_sticky')} egress={rep.get('n_pad_egress')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
