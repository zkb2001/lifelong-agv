"""Probe progress-gated CEBI (stale+between only) on gates + SH08/17/12."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

FULL = {1: 965, 2: 2021, 3: 4440, 8: 1320, 12: 1558, 15: 2523, 17: 2395}
ASA = {1: 919, 2: 2021, 3: 3743, 8: 1320, 12: 1558, 15: 2358, 17: 2395}
SLOTS = [1, 2, 3, 8, 17, 12, 15]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_pgcebi"
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
        done = int(rep.get("tasks_completed") or 0)
        ok = bool(rep.get("validate_ok")) and done == 100
        print(
            f"SH{slot:02d} sim={sim} done={done} FULL={FULL[slot]} ASA={ASA[slot]} "
            f"dASA={sim - ASA[slot]:+d} ok={ok} lane={rep.get('n_lane_locks')} "
            f"egress={rep.get('n_pad_egress')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
