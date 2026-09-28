def _send_idles_wave_wait(pose, steps_by, names, agvs, free, static, now, *, loaded, dest, tid, reserved, max_movers):
    reserved = set(reserved or ()); n = reserved; reserved = ##ERROR## |= {pose[n][:2] for n in names}; ring = _wave_wait_ring(free, reserved); n = None; movers = [n for n in agvs if pose[n][:2] not in ring][:max(0, int(max_movers))]
    if not movers:
        return now
    cells = _wave_wait_cells(len(movers), free, reserved)
    if not cells:
        return now
    cg = {n: pose[n][:2] for n in names}; n = None
    for agv, cell in zip(movers, cells):
        cg[agv] = cell
    tl = _plan_turn_aware_joint(pose, cg, set(movers), static, tmax_scale=2.5, prefer_outer_ring=True)
    if tl is not None:
        tl = _plan_turn_aware_joint(pose, cg, set(movers), static, tmax_scale=4.0)
    if tl is not None:
        for agv, cell in zip(movers, cells):
            path_cells = _bfs_cells(pose[agv][:2], cell, static)
            if path_cells:
                pass
            solo[agv] = path_cells
    now = _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid); print(f"[PIPE] wave-wait {len(movers)} cells={cells} t={now}", flush=True)
    {m: [pose[m][:2]] for ##ERROR## in names}
    n = None; n = None; n = None; m = _apply_paths(steps_by, pose, solo, now, loaded=loaded, dest=dest, tid=tid)
