"""Probe DPSM-only (no CEBI lane-lock) on gates + flats."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

FULL = {
    1: 965, 2: 2021, 3: 4440, 4: 1053, 8: 1320, 9: 1337,
    12: 1558, 14: 1216, 15: 2523, 17: 2395, 19: 1446,
}
ASA = {
    1: 919, 2: 2021, 3: 3743, 4: 1053, 8: 1320, 9: 1337,
    12: 1558, 14: 1216, 15: 2358, 17: 2395, 19: 1446,
}
SLOTS = [1, 2, 3, 8, 17, 15, 12, 4, 9, 14, 19]


def main() -> int:
    for slot in SLOTS:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_dpsm"
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
        dual = 0
        # dual overrides live on StageSelector; expose via report if present
        print(
            f"SH{slot:02d} sim={sim} done={done} FULL={FULL[slot]} ASA={ASA[slot]} "
            f"dASA={sim - ASA[slot]:+d} ok={ok} dual={rep.get('n_dual_override')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
