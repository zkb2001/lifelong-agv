"""Run the SH03 champion ECBS on SH10 with the 400-task demo schedule."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import unload_stand_guard
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.hier_coord.run_easy_demo_400_video import gen_demo_400_tasks
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import write_task_csv

PYC = Path(__file__).with_name("ecbs312.pyc")
spec = importlib.util.spec_from_file_location("old_sh03_ecbs", PYC)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
install = unload_stand_guard.install
meta = load_custom_meta(10, n_tasks=400, max_sim_time=500000, wall_timeout=1e9)
assert meta is not None
unload_stand_guard.set_stations_from_map(Path(meta["map_json"]))
install(mod)

out_dir = Path(__file__).resolve().parents[3] / "results" / "coord_custom_ai" / "compare"
out_dir.mkdir(parents=True, exist_ok=True)
task_csv = out_dir / "SH10_demo400_tasks.csv"
write_task_csv(task_csv, gen_demo_400_tasks(seed=42))
meta = dict(meta)
meta["task_csv"] = str(task_csv.resolve())
meta["n_tasks"] = 400
meta["id"] = "SH_custom_10_t400"

traj = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "coord_custom_ai"
    / "ecbs"
    / "trajectories"
    / "SH_custom_10_ecbs_joint_w1.5_k6.csv"
)
backup = traj.with_name("SH_custom_10_ecbs_joint_w1.5_k6_t100.csv")
if traj.exists() and not backup.exists():
    shutil.copy2(traj, backup)

rep = mod.solve_ecbs(
    10,
    meta=meta,
    task_csv=task_csv,
    max_tasks=0,
    max_active=6,
    weight=1.5,
    plan="joint",
    initial_park=False,
    pipeline=True,
)
out = out_dir / "SH10_ecbs_400.json"
out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
print("[old-ecbs] wrote", out, flush=True)
print(
    "[old-ecbs] completed",
    rep.get("tasks_completed"),
    "/",
    rep.get("tasks_total"),
    "failed",
    rep.get("tasks_failed"),
    "validate",
    rep.get("validate_summary"),
    flush=True,
)
