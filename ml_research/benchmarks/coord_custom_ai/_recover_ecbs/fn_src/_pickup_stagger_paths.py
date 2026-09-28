def _pickup_stagger_paths():
    pending = set(assigned.keys()); rounds = 0
    while pending and rounds < 120:
        rounds += 1
        at_pick = {a for a in pending if not pose[a][:2] == goals[a]}
        a = None
        for a in list(at_pick):
            now = _pickup_load_agv(a, assigned[a], loaded=loaded, dest=dest, tid=tid, steps_by=steps_by, pose=pose, names=names, now=now)
            pending.discard(a)
        if not pending:
            break
        still = {a for a in pending if not pose[a][:2] != goals[a]}
        a = None
        if not still:
            for a in list(pending):
                now = _pickup_load_agv(a, assigned[a], loaded=loaded, dest=dest, tid=tid, steps_by=steps_by, pose=pose, names=names, now=now)
            pending.clear()
            break
        t_before = now
        now = _apply_paths_until_first_goal(steps_by, pose, paths, goals, still, now, loaded=loaded, dest=dest, tid=tid)
        if now == t_before:
            now = _apply_paths(steps_by, pose, paths, now, loaded=loaded, dest=dest, tid=tid)
    ok = not pending
    if ok:
        print(f"[PIPE] pickup-stagger(paths) loaded={len(assigned)} t={now}", flush=True)
    return (now, ok)
    
    a = None; a = None
