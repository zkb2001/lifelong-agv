import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

EXPECT = {8: 1285, 14: 1216, 19: 1437}


def main() -> int:
    bad = 0
    for s, exp in EXPECT.items():
        m = dict(
            load_slot_meta(s, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        )
        m["id"] = f"SH{s:02d}_v"
        m["n_tasks"] = 100
        r = solve(
            s,
            meta=m,
            policy="rrsc",
            use_stage_net=True,
            device="cpu",
            wall_timeout=600.0,
            max_sim_time=200000,
        )
        sim = int(r.get("sim_time") or -1)
        ok = sim == exp and bool(r.get("validate_ok"))
        print(
            f"SH{s:02d} sim={sim} exp={exp} match={ok} lane={r.get('n_lane_locks')}",
            flush=True,
        )
        if not ok:
            bad += 1
    print("VERIFY_PASS", bad == 0)
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
