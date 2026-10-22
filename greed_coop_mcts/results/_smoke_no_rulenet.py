"""Smoke: default (no RuleNet) vs prior sims on gate slots."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

# Prior stack (with RuleNet) from SH01_40_RETEST
PRIOR = {1: 919, 10: 2070, 12: 1468, 17: 2395}


def main() -> None:
    for slot in (1, 10, 12, 17):
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        assert meta
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_norule"
        rep = solve(
            slot,
            meta=meta,
            policy="rrsc",
            use_stage_net=True,
            use_rule_net=False,
            device="cpu",
            wall_timeout=600.0,
            max_sim_time=200000,
        )
        sim = int(rep["sim_time"])
        ok = bool(rep["validate_ok"]) and int(rep["tasks_completed"]) == 100
        print(
            f"SH{slot:02d} sim={sim} prior={PRIOR[slot]} dlt={sim - PRIOR[slot]:+d} "
            f"ok={ok} use_rule_net={rep.get('use_rule_net')} "
            f"asa={rep.get('asa_stage_net')} wall={rep.get('wall_seconds')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
