"""Probe Adaptive Stage Authority candidates (stage on/off)."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

# Critical: gates + maps where OFF helped + risk maps
SLOTS = [1, 2, 3, 8, 15, 5, 6, 7, 16, 18, 4, 13, 19, 20]


def main() -> int:
    for slot in SLOTS:
        for sn in (True, False):
            meta = load_slot_meta(
                slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100
            )
            meta = dict(meta)
            meta["id"] = f"SH{slot:02d}_asa{int(sn)}"
            meta["n_tasks"] = 100
            rep = solve(
                slot,
                meta=meta,
                policy="rrsc",
                use_stage_net=bool(sn),
                use_cgpn=False,
                device="cpu",
                wall_timeout=600.0,
                max_sim_time=200000,
            )
            print(
                f"SH{slot:02d} stage={int(sn)} sim={rep.get('sim_time')} "
                f"done={rep.get('tasks_completed')} ok={rep.get('validate_ok')} "
                f"fleet={rep.get('max_fleet_busy')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
