"""Probe contested×egress arbitration (tick order + holder yield)."""
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
    10: 2215,
    12: 1558,
    15: 2523,
    17: 2395,
}
BASE = {
    1: 919,
    2: 2021,
    3: 3743,
    8: 1285,
    10: 2070,
    12: 1558,
    15: 2358,
    17: 2395,
}
SLOTS = [1, 2, 3, 8, 12, 17, 15, 10]


def run(slot: int, *, use_cgpn: bool) -> None:
    meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
    meta = dict(meta)
    tag = "cgpn" if use_cgpn else "arb"
    meta["id"] = f"SH{slot:02d}_{tag}"
    meta["n_tasks"] = 100
    rep = solve(
        slot,
        meta=meta,
        policy="rrsc",
        use_stage_net=True,
        use_cgpn=use_cgpn,
        device="cpu",
        wall_timeout=600.0,
        max_sim_time=200000,
    )
    sim = int(rep.get("sim_time") or -1)
    ok = bool(rep.get("validate_ok")) and int(rep.get("tasks_completed") or 0) == 100
    print(
        f"SH{slot:02d}[{tag}] sim={sim} FULL={FULL[slot]} BASE={BASE[slot]} "
        f"dFULL={sim - FULL[slot]:+d} dBASE={sim - BASE[slot]:+d} "
        f"ok={ok} egress={rep.get('n_pad_egress')} "
        f"gates={rep.get('n_dest_gates')} lane={rep.get('n_lane_locks')}",
        flush=True,
    )


def main() -> int:
    for slot in SLOTS:
        run(slot, use_cgpn=False)
    print("--- CGPN@SH12 ---", flush=True)
    run(12, use_cgpn=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
