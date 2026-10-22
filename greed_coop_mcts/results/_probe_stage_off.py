import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

# Gates + maps where stage-off might help or hurt
SLOTS = [1, 2, 3, 8, 9, 11, 5, 6, 16, 18, 7, 10, 17, 19]


def main() -> int:
    for slot in SLOTS:
        for sn in (True, False):
            meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=1800.0, n_tasks=100)
            meta = dict(meta)
            meta["id"] = f"SH{slot:02d}_s{int(sn)}"
            meta["n_tasks"] = 100
            rep = solve(
                slot,
                meta=meta,
                policy="rrsc",
                use_stage_net=bool(sn),
                use_cgpn=False,
                device="cpu",
                wall_timeout=1800.0,
                max_sim_time=200000,
            )
            print(
                f"SH{slot:02d} stage={int(sn)} sim={rep.get('sim_time')} "
                f"done={rep.get('tasks_completed')} ok={rep.get('validate_ok')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
