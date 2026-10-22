"""Helpers for standalone greed–coop MAPD (no Hier / ECBS)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from greed_coop_mcts.mcts import LocalAgent
from greed_coop_mcts.obs import manhattan

Cell = Tuple[int, int]


def extract_agents_from_sim(sim, *, max_agents: int = 12) -> List[LocalAgent]:
    out: List[LocalAgent] = []
    for agv in list(getattr(sim, "agvs", None) or [])[: max(1, int(max_agents))]:
        name = str(getattr(agv, "name", "") or getattr(agv, "id", "") or "")
        if not name:
            continue
        x = int(getattr(agv, "x", 0))
        y = int(getattr(agv, "y", 0))
        goal = None
        tx = getattr(agv, "target_x", None)
        ty = getattr(agv, "target_y", None)
        if tx is not None and ty is not None:
            goal = (int(tx), int(ty))
        else:
            path = getattr(agv, "path", None) or []
            if path:
                last = path[-1]
                if isinstance(last, (tuple, list)) and len(last) >= 2:
                    goal = (int(last[0]), int(last[1]))
        out.append(LocalAgent(name=name, pos=(x, y), goal=goal))
    return out


def pick_ego_crowded(agents: Sequence[LocalAgent]) -> List[LocalAgent]:
    if not agents:
        return []

    def crowd(a: LocalAgent) -> float:
        return sum(manhattan(a.pos, b.pos) for b in agents if b.name != a.name)

    return sorted(agents, key=crowd)
