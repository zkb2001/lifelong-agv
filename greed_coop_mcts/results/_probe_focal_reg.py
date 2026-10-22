import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

BASE = {12: 1558, 11: 1707, 2: 2021, 8: 1285, 17: 2395, 18: 1618, 1: 919, 3: 3743}
for s in [12, 11, 2, 8, 17, 18, 1, 3]:
    m = dict(load_slot_meta(s, max_sim_time=200000, wall_timeout=600, n_tasks=100))
    m["id"] = f"SH{s:02d}_fc"
    m["n_tasks"] = 100
    r = solve(
        s,
        meta=m,
        policy="rrsc",
        use_stage_net=True,
        device="cpu",
        wall_timeout=600,
        max_sim_time=200000,
    )
    sim = int(r.get("sim_time") or -1)
    print(
        f"SH{s:02d} sim={sim} base={BASE[s]} dlt={sim - BASE[s]:+d} "
        f"focal={r.get('asa_focal_clear')} ok={r.get('validate_ok')}",
        flush=True,
    )
