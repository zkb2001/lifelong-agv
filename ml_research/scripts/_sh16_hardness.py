from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.hierarchical_gate import map_hardness, narrow_cut_score
import json
from pathlib import Path

m = load_custom_meta(16, n_tasks=100)
j = json.loads(Path(m["map_json"]).read_text(encoding="utf-8"))
# rebuild free/static like solve does roughly
extras = {(int(p[0]), int(p[1])) if not isinstance(p, dict) else (int(p["x"]), int(p["y"])) for p in (m.get("extra_obstacles") or [])}
# map cells 1..20 typically
allc = {(x, y) for x in range(1, 21) for y in range(1, 21)}
# walls from map if present
walls = set()
for w in j.get("walls") or j.get("obstacles") or []:
    if isinstance(w, dict):
        walls.add((int(w["x"]), int(w["y"])))
    else:
        walls.add((int(w[0]), int(w[1])))
static = walls | extras
free = allc - static
print("hardness", map_hardness(extra_obstacles=list(m.get("extra_obstacles") or []), static=static, free=free))
print("narrow", narrow_cut_score(free))
print("n_static", len(static), "n_free", len(free))
