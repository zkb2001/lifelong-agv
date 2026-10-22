"""Probe PGFA off on remaining flat maps (monkeypatch should_pgfa)."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import greed_coop_mcts.global_altruism as ga
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

ga.should_pgfa = lambda **kw: False  # type: ignore

BASE = {2: 2021, 4: 1053, 9: 1337, 12: 1558, 13: 832, 14: 1216, 17: 2395, 1: 919, 8: 1285}
SLOTS = [4, 9, 14, 12, 13, 2, 17, 1, 8]


def main() -> int:
    for slot in SLOTS:
        meta = dict(
            load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        )
        meta["id"] = f"SH{slot:02d}_npg"
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
        print(
            f"SH{slot:02d} sim={sim} done={done} base={BASE[slot]} "
            f"dlt={sim - BASE[slot]:+d} ok={rep.get('validate_ok')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
