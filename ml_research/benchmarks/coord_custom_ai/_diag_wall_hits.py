"""Diagnose hard-wall hits on SH08/SH10 phase-stress trajectories."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ml_research.benchmarks.common import load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    get_hard_wall_cells,
    load_trajectory_rows,
    validate_hybrid_trajectory,
)

ROOT = Path("ml_research/results/coord_custom_ai/ecbs")


def main() -> None:
    for slot in (8, 10):
        meta = load_custom_meta(slot, max_sim_time=500000, wall_timeout=1e9)
        assert meta
        # use same task csv as phase stress if available
        phase_task = Path(
            "ml_research/results/hier_coord/lifelong_phase_stress/full100_drop/"
            "phased_same_dropoff_s2026.csv"
        )
        if phase_task.exists():
            meta = dict(meta)
            meta["task_csv"] = str(phase_task.resolve())
            meta["id"] = f"SH_custom_{slot:02d}_phase_stress"
        traj = ROOT / "trajectories" / f"SH_custom_{slot:02d}_phase_stress_ecbs_joint_w1.5_k6_hier.csv"
        rep = validate_hybrid_trajectory(meta, traj)
        iss = rep["issues"]
        walls = iss.get("hard_wall") or []
        print(f"\n=== SH{slot:02d} ok={rep['ok']} summary={rep and 'x'} n_wall={iss.get('n_hard_wall')} ===")
        cells = Counter((w.get("x"), w.get("y")) for w in walls)
        print("top wall cells:", cells.most_common(10))
        print("sample events:", walls[:8])
        # compare extras vs hard walls
        _, env, _, _, _ = load_scenario(Path(meta["task_csv"]), Path(meta["position_csv"]))
        extras = list(meta.get("extra_obstacles") or [])
        patch_extra_obstacles(env, extras)
        hard = get_hard_wall_cells(env)
        print(f"n_extras={len(extras)} n_hard_walls={len(hard)}")
        # which hit cells are extras?
        hit_extra = 0
        for (x, y), n in cells.items():
            if {"x": x, "y": y} in extras or (x, y) in {(int(o["x"]), int(o["y"])) if isinstance(o, dict) else (int(o[0]), int(o[1])) for o in extras}:
                hit_extra += n
        print(f"hits_on_extra_obstacles≈{hit_extra}/{iss.get('n_hard_wall')}")


if __name__ == "__main__":
    main()
