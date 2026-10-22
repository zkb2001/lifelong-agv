"""Scan SH01-40 geometry vs FACN AFA v2 (idle attach + raise caps)."""
from __future__ import annotations

import json
from pathlib import Path

from greed_coop_mcts.facn_net import (
    FLOOD_NEAR_HI,
    GATE_N_HI,
    MID_NEAR_HI,
    OPEN_NEAR_HI,
    afa_allow_attach,
    afa_allow_raise,
    afa_raise_cap,
)
from greed_coop_mcts.pgcp import near_pad_deg2_count, throat_band_fleet_cap
from greed_coop_mcts.rule_obs import GRID, seal_ghost_border
from greed_coop_mcts.scenario import load_slot_meta

ROOT = Path(__file__).resolve().parent


def geom(slot: int) -> dict:
    meta = load_slot_meta(slot)
    jpath = Path((meta or {}).get("map_json") or "")
    blocked, drops = set(), set()
    if jpath.is_file():
        data = json.loads(jpath.read_text(encoding="utf-8"))
        for p in data.get("extra_obstacles") or []:
            blocked.add(
                (int(p["x"]), int(p["y"]))
                if isinstance(p, dict)
                else (int(p[0]), int(p[1]))
            )
        for p in data.get("dropoffs") or []:
            c = (int(p["x"]), int(p["y"]))
            drops.add(c)
            blocked.add(c)
        for p in data.get("pickups") or []:
            c = (int(p["x"]), int(p["y"]))
            blocked.add(c)
    blocked = seal_ghost_border(blocked)
    free = sum(1 for x in range(1, GRID) for y in range(1, GRID) if (x, y) not in blocked)
    n = int((meta or {}).get("n_agvs") or 0)
    near = near_pad_deg2_count(blocked, drops)
    k = throat_band_fleet_cap(n, near, base=6)
    dens = n / max(1, free)
    idle = (n - k) / max(1, n)
    return {
        "n": n,
        "free": free,
        "near": near,
        "k_tbc": k,
        "dens": round(dens, 4),
        "idle": round(idle, 3),
        "raise_cap": afa_raise_cap(
            n_agents=n, free=free, near_pad_deg2=near, k_tbc=k
        ),
        "raise": afa_allow_raise(
            n_agents=n, free=free, near_pad_deg2=near, k_tbc=k
        ),
        "attach": afa_allow_attach(
            n_agents=n, free=free, near_pad_deg2=near, k_tbc=k
        ),
    }


def main() -> None:
    nof = {}
    p = ROOT / "summary_nofacn_40.json"
    if p.is_file():
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows") or []:
            nof[int(r["slot"])] = r.get("sim_time")
    print(
        f"AFA v2: OPEN≤{OPEN_NEAR_HI}→+4 MID≤{MID_NEAR_HI}→+2 else+1 "
        f"GATE_N≤{GATE_N_HI}→0 FLOOD>{FLOOD_NEAR_HI} ban−2"
    )
    print("slot n free near dens idle k raise_cap attach nofacn_sim")
    n_attach = n_raise = 0
    for s in range(1, 41):
        g = geom(s)
        n_attach += int(g["attach"])
        n_raise += int(g["raise"])
        print(
            f"SH{s:02d} {g['n']:2d} {g['free']:3d} {g['near']:3d} "
            f"{g['dens']:.3f} {g['idle']:.2f} {g['k_tbc']} "
            f"{g['raise_cap']} {int(g['attach'])} {nof.get(s)}"
        )
    print(f"attach={n_attach}/40 raise_cap>0={n_raise}/40")


if __name__ == "__main__":
    main()
