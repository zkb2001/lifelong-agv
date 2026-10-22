import json
from pathlib import Path

from greed_coop_mcts.pgcp import cell_degree
from greed_coop_mcts.scenario import load_scenario, load_slot_meta, patch_extra_obstacles


def score(slot: int) -> None:
    meta = load_slot_meta(slot)
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
    deg2 = 0
    near = 0
    for x in range(1, 21):
        for y in range(1, 21):
            if (x, y) in blocked:
                continue
            if cell_degree((x, y), blocked) <= 2:
                deg2 += 1
                if any(abs(x - px) + abs(y - py) <= 6 for px, py in dropoffs):
                    near += 1
    print(
        f"SH{slot:02d} n={meta.get('n_agvs')} drops={len(dropoffs)} "
        f"deg2={deg2} near_drop_deg2={near}"
    )


if __name__ == "__main__":
    for s in range(1, 21):
        score(s)
