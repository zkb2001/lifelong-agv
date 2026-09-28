"""Connected random warehouse maps + harvest of existing hard/hutong assets."""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ml_research.benchmarks.scenarios.generator import (
    END_POINTS,
    START_POINTS,
    _reserved_cells,
    build_corridor_obstacles,
)

Cell = Tuple[int, int]
ROOT = Path(__file__).resolve().parents[3]  # ml_research/
EXPORTS = ROOT / "map_editor" / "exports"
CURRICULUM = ROOT / "results" / "curriculum_shape" / "scenarios"


def _neighbors(c: Cell) -> List[Cell]:
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _in_grid(c: Cell) -> bool:
    return 1 <= c[0] <= 20 and 1 <= c[1] <= 20


def free_cells(static: Set[Cell]) -> Set[Cell]:
    """Cells not occupied by static obstacles (stations/AGVs may sit here)."""
    out: Set[Cell] = set()
    for x in range(1, 21):
        for y in range(1, 21):
            c = (x, y)
            if c not in static:
                out.add(c)
    return out


def is_connected(static: Set[Cell], seeds: Optional[Sequence[Cell]] = None) -> bool:
    """True if every non-obstacle cell is in one 4-connected component.

    Station/AGV reserved cells are walkable for this check (they are not walls);
    only ``static`` obstacles block. Also require each station approach cell
    that is non-static to be reachable from an AGV seed.
    """
    walk = free_cells(static)
    if not walk:
        return False
    start = None
    for s in seeds or ((3, 1), (6, 1), (10, 10), (2, 6)):
        if s in walk:
            start = s
            break
    if start is None:
        start = next(iter(walk))
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur):
            if nxt in walk and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    if len(seen) < len(walk):
        return False
    for _, x, y in START_POINTS + END_POINTS:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a = (x + dx, y + dy)
            if a in walk and a not in seen:
                return False
    return True


def punch_to_connect(static: Set[Cell], rng: random.Random) -> Set[Cell]:
    """Remove obstacles until the free graph is connected."""
    obs = set(static)
    for _ in range(400):
        if is_connected(obs):
            return obs
        # carve a random obstacle toward a free neighbor
        blockers = [c for c in obs if c not in _reserved_cells()]
        if not blockers:
            break
        c = rng.choice(blockers)
        obs.discard(c)
    return obs


def gen_hutong(seed: int, *, density: float = 0.72) -> List[Cell]:
    """Parallel vertical walls with staggered 1-cell gaps (snake hutong style)."""
    rng = random.Random(seed)
    reserved = _reserved_cells()
    obs: Set[Cell] = set()
    xs = [4, 7, 10, 13, 16]
    for i, x in enumerate(xs):
        gap = 3 + (i * 3 + seed) % 15
        gap2 = (gap + 7) % 18 + 2
        for y in range(2, 20):
            if y in (gap, gap2):
                continue
            if rng.random() > density:
                continue
            c = (x, y)
            if c not in reserved:
                obs.add(c)
    # a few horizontal stubs
    for _ in range(8):
        x = rng.randint(3, 18)
        y = rng.randint(3, 18)
        for dx in range(3):
            c = (x + dx, y)
            if _in_grid(c) and c not in reserved:
                obs.add(c)
    obs = punch_to_connect(obs, rng)
    return sorted(obs)


def gen_maze_dfs(seed: int, *, carve_prob: float = 0.55) -> List[Cell]:
    """Fill then carve a DFS spanning tree of odd cells; keep connectivity."""
    rng = random.Random(seed)
    reserved = _reserved_cells()
    obs: Set[Cell] = {
        (x, y)
        for x in range(2, 20)
        for y in range(2, 20)
        if (x, y) not in reserved
    }
    # carve from AGV hub
    start = (3, 2) if (3, 2) not in reserved else (4, 2)
    stack = [start]
    visited = {start}
    obs.discard(start)
    while stack:
        cur = stack[-1]
        nbrs = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nxt = (cur[0] + dx, cur[1] + dy)
            mid = (cur[0] + dx // 2, cur[1] + dy // 2)
            if not _in_grid(nxt) or nxt in reserved or nxt in visited:
                continue
            if mid in reserved:
                continue
            nbrs.append((nxt, mid))
        if not nbrs:
            stack.pop()
            continue
        nxt, mid = rng.choice(nbrs)
        visited.add(nxt)
        obs.discard(nxt)
        obs.discard(mid)
        stack.append(nxt)
        if rng.random() > carve_prob and len(visited) > 30:
            # early stop → denser residual walls
            break
    # open approaches
    for _, x, y in START_POINTS + END_POINTS:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            obs.discard((x + dx, y + dy))
    obs = punch_to_connect(obs, rng)
    return sorted(obs)


def gen_sprinkle(seed: int, *, n_extra: int = 40) -> List[Cell]:
    rng = random.Random(seed)
    reserved = _reserved_cells()
    candidates = [
        (x, y)
        for x in range(2, 20)
        for y in range(2, 20)
        if (x, y) not in reserved
    ]
    rng.shuffle(candidates)
    obs = set(candidates[:n_extra])
    obs = punch_to_connect(obs, rng)
    return sorted(obs)


def gen_map(style: str, seed: int) -> Dict:
    style = style.lower().strip()
    if style in ("corridor_mild", "corridor"):
        obs = build_corridor_obstacles(seed, density="mild")
    elif style == "corridor_hard":
        obs = build_corridor_obstacles(seed, density="hard")
    elif style in ("hutong", "snake"):
        obs = gen_hutong(seed)
    elif style in ("maze", "maze_dfs"):
        obs = gen_maze_dfs(seed)
    elif style in ("open", "sprinkle_mild"):
        obs = gen_sprinkle(seed, n_extra=18)
    elif style in ("sprinkle", "sprinkle_hard"):
        obs = gen_sprinkle(seed, n_extra=55)
    else:
        raise ValueError(f"unknown map style: {style}")
    obs_set = punch_to_connect(set(map(tuple, obs)), random.Random(seed))
    return {
        "style": style,
        "seed": int(seed),
        "obstacles": [[int(x), int(y)] for x, y in sorted(obs_set)],
        "n_obstacles": len(obs_set),
        "connected": bool(is_connected(obs_set)),
    }


def harvest_existing_maps() -> List[Dict]:
    """Reuse SH_custom exports + curriculum_shape scenarios when present."""
    out: List[Dict] = []
    if EXPORTS.exists():
        for p in sorted(EXPORTS.glob("SH_custom_*.json")):
            # only SH_custom_01.json .. SH_custom_20.json (skip *_obstacles.json)
            stem = p.stem
            parts = stem.split("_")
            if len(parts) != 3 or not parts[2].isdigit():
                continue
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            obs = j.get("extra_obstacles") or j.get("obstacles") or []
            if not obs and "cells" in j:
                obs = [
                    [c["x"], c["y"]]
                    for c in j["cells"]
                    if c.get("type") == "obstacle"
                ]
            cells = []
            for o in obs:
                if isinstance(o, dict):
                    cells.append([int(o["x"]), int(o["y"])])
                elif isinstance(o, (list, tuple)) and len(o) >= 2:
                    cells.append([int(o[0]), int(o[1])])
            notes = str(j.get("notes") or j.get("map_shape") or p.stem)
            out.append(
                {
                    "id": p.stem,
                    "source": "exports",
                    "path": str(p),
                    "style": notes,
                    "obstacles": cells,
                    "n_obstacles": len(cells),
                    "position_csv": str(p.with_name(p.stem + "_position.csv")),
                    "task_csv_default": str(p.with_name(p.stem + "_task.csv")),
                }
            )
    if CURRICULUM.exists():
        for p in sorted(CURRICULUM.glob("*.json")):
            # prefer compact L12/L20 if present; else any
            name = p.name.lower()
            if "l100" in name and "l12" not in name:
                # still include maze/hutong-ish
                if "maze" not in name and "hutong" not in name and "snake" not in name:
                    continue
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            obs = j.get("extra_obstacles") or j.get("obstacles") or []
            cells = []
            for o in obs:
                if isinstance(o, (list, tuple)) and len(o) >= 2:
                    cells.append([int(o[0]), int(o[1])])
            out.append(
                {
                    "id": p.stem,
                    "source": "curriculum_shape",
                    "path": str(p),
                    "style": str(j.get("map_shape") or j.get("obstacle_mode") or p.stem),
                    "obstacles": cells,
                    "n_obstacles": len(cells),
                    "position_csv": str(
                        Path(j["position_csv"])
                        if j.get("position_csv")
                        else p.with_suffix(".position.csv")
                    ),
                    "task_csv_default": str(j.get("task_csv") or ""),
                }
            )
    return out
