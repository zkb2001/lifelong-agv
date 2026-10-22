"""Repro check on ASA-hit maps (seeded subprocess-equivalent)."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

EXPECT = {1: 919, 3: 3743, 6: 1756, 7: 3311, 16: 1195, 18: 1618}


def main() -> int:
    bad = 0
    for slot, exp in EXPECT.items():
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_repro"
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
        match = sim == exp
        print(
            f"SH{slot:02d} sim={sim} expect={exp} match={match} ok={ok} "
            f"asa={rep.get('asa_stage_net')}",
            flush=True,
        )
        if not (ok and match):
            bad += 1
    print("REPRO_PASS", bad == 0, "mismatches", bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
