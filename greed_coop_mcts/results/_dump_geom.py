"""Dump fleet×throat geometry and ASA/TBC/CEBI decisions for SH01-20."""
import json
import os
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.pgcp import (
    adaptive_use_stage_net,
    near_pad_deg2_count,
    throat_band_fleet_cap,
)
from greed_coop_mcts.scenario import load_scenario, load_slot_meta, patch_extra_obstacles

FULL = {
    1: 965,
    2: 2021,
    3: 4440,
    4: 1053,
    5: 1616,
    6: 2024,
    7: 3897,
    8: 1320,
    9: 1337,
    10: 2215,
    11: 2384,
    12: 1558,
    13: 832,
    14: 1216,
    15: 2523,
    16: 1643,
    17: 2395,
    18: 1711,
    19: 1446,
    20: 947,
}
BASE = {
    1: 919,
    2: 2021,
    3: 3743,
    4: 1053,
    5: 1554,
    6: 1756,
    7: 3311,
    8: 1285,
    9: 1337,
    10: 2070,
    11: 1707,
    12: 1558,
    13: 832,
    14: 1216,
    15: 2358,
    16: 1195,
    17: 2395,
    18: 1618,
    19: 1437,
    20: 883,
}


def main() -> int:
    for s in range(1, 21):
        meta = load_slot_meta(s)
        _mod, env, _a, _t, _n = load_scenario(
            Path(meta["task_csv"]), Path(meta["position_csv"]), force_reload=True
        )
        if meta.get("extra_obstacles"):
            patch_extra_obstacles(env, list(meta["extra_obstacles"]))
        static = {(int(p[0]), int(p[1])) for p in (env.get_static_obstacles() or [])}
        j = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
        pickups = {(int(p["x"]), int(p["y"])) for p in j.get("pickups") or []}
        dropoffs = {(int(p["x"]), int(p["y"])) for p in j.get("dropoffs") or []}
        stations = pickups | dropoffs
        blocked = (static - stations) | stations
        near = near_pad_deg2_count(blocked, dropoffs)
        n = int(meta.get("n_agvs") or len(_a) or 0)
        asa = adaptive_use_stage_net(n, near)
        cap = throat_band_fleet_cap(n, near, base=6)
        cebi = n >= 10 and near >= 100
        flat = BASE[s] == FULL[s]
        print(
            f"SH{s:02d} n={n:2d} near={near:3d} asa={int(asa)} cap={cap} "
            f"cebi={int(cebi)} flat={int(flat)} base={BASE[s]} full={FULL[s]} "
            f"dlt={BASE[s] - FULL[s]:+d}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
