import json
from pathlib import Path

from greed_coop_mcts.pgcp import near_pad_deg2_count, throat_band_fleet_cap
from greed_coop_mcts.rule_obs import seal_ghost_border
from greed_coop_mcts.scenario import load_slot_meta

for s in [1, 3, 11, 12, 13, 20]:
    m = load_slot_meta(s)
    j = json.loads(Path(m["map_json"]).read_text(encoding="utf-8"))
    blocked = set()
    for p in j.get("extra_obstacles") or []:
        if isinstance(p, dict):
            blocked.add((int(p["x"]), int(p["y"])))
        else:
            blocked.add((int(p[0]), int(p[1])))
    drops = set()
    for p in j.get("dropoffs") or []:
        c = (int(p["x"]), int(p["y"]))
        drops.add(c)
        blocked.add(c)
    for p in j.get("pickups") or []:
        blocked.add((int(p["x"]), int(p["y"])))
    blocked = seal_ghost_border(blocked)
    n = int(m.get("n_agvs") or 0)
    r = near_pad_deg2_count(blocked, drops)
    free = sum(
        1 for x in range(1, 21) for y in range(1, 21) if (x, y) not in blocked
    )
    cap = throat_band_fleet_cap(n, r, base=6)
    print(
        f"SH{s:02d} n={n:2d} free={free:3d} npd2={r:3d} "
        f"TBC={cap} dens={n / max(1, free):.3f} idle_pool≈{max(0, n - cap)}"
    )
