"""Run the 2026-09-02 SH03 champion ECBS bytecode on SH_custom_10."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import unload_stand_guard
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta

PYC = Path(__file__).with_name("ecbs312.pyc")
spec = importlib.util.spec_from_file_location("old_sh03_ecbs", PYC)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
set_stations_from_map = unload_stand_guard.set_stations_from_map
install = unload_stand_guard.install
meta = load_custom_meta(10)
assert meta is not None
set_stations_from_map(Path(meta["map_json"]))
install(mod)

rep = mod.solve_ecbs(
    10,
    max_tasks=0,
    max_active=6,
    weight=1.5,
    plan="joint",
    initial_park=False,
    pipeline=True,
)
out = Path(__file__).resolve().parents[3] / "results" / "coord_custom_ai" / "compare" / "SH10_unload_keep.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
print("[old-ecbs] wrote", out, flush=True)
