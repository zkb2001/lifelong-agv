"""Independent lifelong PIBT-MAPD for SH_custom_01..20 (official 100 tasks).

This package is a **second solver stack**, separate from the hierarchical
M0 + ECBS system in ``coord_custom_ai/solve_ecbs.py``.

| | This package (`pibt_mapd`) | Hierarchical (`solve_ecbs`) |
|--|--|--|
| Motion | PIBT one-step each tick | M0 engine / prioritized ST / ECBS waves |
| Mode switch | None | map hardness / narrow-cut / stall handoff |
| A* fallback | **Not used** | spacetime A* + optional ``pibt_fallback`` |
| Task order | Strict FIFO surface (no rotate-head) | FIFO + historical rotate-head (removed) |
| Results dir | ``results/coord_custom_ai/pibt_mapd_100/`` | ``ecbs/`` / ``batch_400_switch_20/`` |

Do **not** import ``solve_ecbs`` from here. Shared read-only helpers only:
``load_custom_meta``, ``load_scenario``, ``validate_hybrid_trajectory``.

## Run

```bash
# single map
python -m ml_research.benchmarks.pibt_mapd.solve_pibt --slot 1

# SH01–SH20 × official 100 tasks
python -m ml_research.benchmarks.pibt_mapd.batch_100 --start 1 --end 20
```

Outputs: ``pibt_mapd_100/results.jsonl``, ``SUMMARY.md``, ``excel/SHXX.xlsx``,
trajectories under ``pibt_mapd_100/trajectories/``.

## Rules enforced

- Pickup on approach door (not station center); 1s same-cell no-turn dwell
- Unload on free 4-neighbor of dropoff; 1s dwell
- In-place turn ≤90°/tick then move (180° split into two ticks)
- One physical approacher per pickup station
- Surface queue order never rearranged
"""
