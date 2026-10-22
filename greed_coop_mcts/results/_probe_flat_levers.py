"""Probe levers on remaining flat maps (force stage / cgpn / teacher)."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

# slot -> FULL
FULL = {2: 2021, 4: 1053, 9: 1337, 12: 1558, 13: 832, 14: 1216, 17: 2395}
CUR = {2: 2021, 4: 1053, 9: 1337, 12: 1558, 13: 832, 14: 1216, 17: 2395}

# (slot, use_stage_net, force_stage_net, use_cgpn, tag)
CASES = [
    # SH04: try teacher vs force net
    (4, False, False, False, "tea"),
    (4, True, True, False, "fnet"),
    (4, True, False, True, "cgpn"),
    # SH09: teacher (ASA currently net because near<60)
    (9, False, False, False, "tea"),
    (9, True, True, False, "fnet"),
    (9, True, False, True, "cgpn"),
    # SH14: teacher (ASA: n=9 near=97 → net)
    (14, False, False, False, "tea"),
    (14, True, False, True, "cgpn"),
    # SH13: teacher / cgpn
    (13, False, False, False, "tea"),
    (13, True, False, True, "cgpn"),
    # SH02 gate: cgpn only
    (2, True, False, True, "cgpn"),
    # SH12: cgpn
    (12, True, False, True, "cgpn"),
    # SH17: cgpn (stage must stay on)
    (17, True, False, True, "cgpn"),
    (17, True, True, False, "fnet"),
]


def main() -> int:
    for slot, sn, force, cg, tag in CASES:
        meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
        meta = dict(meta)
        meta["id"] = f"SH{slot:02d}_{tag}"
        meta["n_tasks"] = 100
        rep = solve(
            slot,
            meta=meta,
            policy="rrsc",
            use_stage_net=bool(sn),
            force_stage_net=bool(force),
            use_cgpn=bool(cg),
            device="cpu",
            wall_timeout=600.0,
            max_sim_time=200000,
        )
        sim = int(rep.get("sim_time") or -1)
        done = int(rep.get("tasks_completed") or 0)
        ok = bool(rep.get("validate_ok")) and done == 100
        print(
            f"SH{slot:02d}/{tag} sim={sim} done={done} CUR={CUR[slot]} "
            f"dlt={sim - CUR[slot]:+d} ok={ok} asa={rep.get('asa_stage_net')} "
            f"cgpn={rep.get('n_cgpn_priority')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
