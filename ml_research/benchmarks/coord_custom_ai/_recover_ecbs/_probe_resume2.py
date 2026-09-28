"""Continue SH10 400 from the 302-task trajectory until completion or the wall budget."""
from __future__ import annotations

import os
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs

os.environ["AGV_M0_HANDOFF"] = "1"
os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
os.environ["AGV_ASTAR_WALL_S"] = "0.20"
os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
os.environ["AGV_TRIED_TTL"] = "8"
os.environ["AGV_QUIET_ASSIGN"] = "1"
os.environ["AGV_ASTAR_BOOST_EVERY"] = "10"
os.environ["AGV_ASTAR_BOOST_S"] = "0.45"

root = Path(r"d:\compitition\MioVerse\final_version\ml_research")
task_csv = root / "results" / "coord_custom_ai" / "batch_400_switch_20" / "SH10_demo400_tasks.csv"
resume = (
    root
    / "results"
    / "coord_custom_ai"
    / "ecbs"
    / "trajectories"
    / "SH10_leave400_stickyfix_ecbs_joint_w1.5_k6_hier.csv"
)
meta = load_custom_meta(10, n_tasks=400, max_sim_time=500000, wall_timeout=900)
assert meta is not None
meta = dict(meta)
meta["task_csv"] = str(task_csv.resolve())
meta["n_tasks"] = 400
meta["id"] = "SH10_leave400_collidefix3"
meta["wall_timeout"] = 900.0
meta["progress_every"] = 20
meta["allow_mode_switch"] = True
meta["legacy_wave_ecbs"] = True
meta["wall_first_ecbs"] = True
meta["wall_first_min_done"] = 16
meta["wall_first_min_s"] = 80.0
meta["ecbs_min_wave_k"] = 8
meta["ecbs_prefer_full_k"] = True
meta["traffic_recovery"] = False
meta["resume_traj"] = str(resume.resolve())

rep = solve_ecbs(
    slot=10,
    meta=meta,
    task_csv=task_csv,
    max_tasks=0,
    max_active=6,
    weight=1.5,
    plan="joint",
    pipeline=True,
    turn_aware=True,
    initial_park=False,
    use_hierarchical=True,
    allow_mode_switch=True,
    legacy_wave_ecbs=True,
    wave_hard_cap=6,
    joint_core="prioritized",
    traffic_recovery=False,
)
print(
    "RESULT",
    "done",
    rep.get("tasks_completed"),
    "sim",
    rep.get("sim_time"),
    "ok",
    rep.get("validate_ok"),
    rep.get("validate_summary"),
    flush=True,
)
