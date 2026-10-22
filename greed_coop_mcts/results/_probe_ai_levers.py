import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

CASES = [
    # slot, use_stage_net, use_cgpn
    (16, True, False),
    (16, False, False),
    (14, True, False),
    (14, False, False),
    (12, True, False),
    (12, True, True),
    (12, False, False),
    (5, True, False),
    (5, False, False),
    (6, True, False),
    (6, False, False),
    (18, True, False),
    (18, False, False),
    (11, True, True),
    (20, True, True),
    (3, True, True),
    (8, True, True),
    (15, True, True),
]


def main() -> int:
    for slot, sn, cg in CASES:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=1800.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_lev"
        meta["n_tasks"] = 100
        rep = solve(
            slot,
            meta=meta,
            policy="rrsc",
            use_stage_net=bool(sn),
            use_cgpn=bool(cg),
            device="cpu",
            wall_timeout=1800.0,
            max_sim_time=200000,
        )
        print(
            f"SH{slot:02d} stage={int(sn)} cgpn={int(cg)} "
            f"sim={rep.get('sim_time')} fleet={rep.get('max_fleet_busy')} "
            f"ok={rep.get('validate_ok')} n_cgpn={rep.get('n_cgpn_priority')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
