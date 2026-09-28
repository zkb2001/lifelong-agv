def _move_agvs_to_cells(pose, steps_by, names, movers, cell_goals, free, static, now, *, loaded, dest, tid):
    todo = [m for m in movers if pose[m][:2] != cell_goals[m]]; m = None
    if not todo:
        return now
    cg = {n: pose[n][:2] for n in names}; n = None
    for m in todo:
        g = cell_goals[m]
        cg[m] = g
    tl = _plan_turn_aware_joint(pose, cg, set(todo), static, tmax_scale=2.5, prefer_outer_ring=True)
    if tl is None:
        now = _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid)
        return now
    
    for m in todo:
        cells = _bfs_cells(pose[m][:2], cell_goals[m], static)
        if cells and len(cells) < 2:
            continue
        solo = {n: [pose[n][:2]] for n in names}
        n = None
        solo[m] = cells
        now = _apply_paths(steps_by, pose, solo, now, loaded=loaded, dest=dest, tid=tid)
    return now
    
    m = None; n = None; n = None
