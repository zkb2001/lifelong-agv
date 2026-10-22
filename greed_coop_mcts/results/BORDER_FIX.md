# Ghost-border OOB fix

## Bug
Planner used `GRID=21` with bounds `0 <= x,y < 21`, so `(x,0)` / `(0,y)` were walkable.
Competition validator only accepts `[1,20]` → `out_of_bounds` INVALID on some maps
(especially generated corridor/maze), while tasks could still complete via the illegal rim.

## Fix
1. `rule_obs.seal_ghost_border` + `in_arena` — seal `x=0`/`y=0` (same as training).
2. `solve.py` — apply `seal_ghost_border` when building `blocked`.
3. `bfs_field` + runtime `_neighbors` (local/pgcp/meeting/rrsc/glocal/rule_system/altruism)
   — reject non-arena cells.

## Retest (`_retest_border_fix.py`)
- Former INVALID gen sample SH23/25/26/28/29/31/33/39: **8/8 VALID, mot=0**
- Prod smoke SH01/12/17: **3/3 VALID** (SH12 sim=1468 unchanged vs prior E4 stack)
