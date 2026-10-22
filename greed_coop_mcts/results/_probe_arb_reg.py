"""Check if contested×egress arb regresses SH11 / flats."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

BASE = {
    2: 2021,
    4: 1053,
    9: 1337,
    11: 1707,
    12: 1558,
    13: 832,
    14: 1216,
    17: 2395,
}
SLOTS = [11, 12, 17, 2, 4]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_ar"
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
            f"SH{slot:02d} sim={sim} BASE={BASE[slot]} dlt={sim - BASE[slot]:+d} "
            f"ok={ok} gates={rep.get('n_dest_gates')} egress={rep.get('n_pad_egress')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
