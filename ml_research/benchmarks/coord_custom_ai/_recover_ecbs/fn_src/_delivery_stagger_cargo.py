def _delivery_stagger_cargo():
    pending = set(batch.keys()); rounds = 0
    while pending and rounds < 140:
        rounds += 1
        at_drop = {a for a in pending if not pose[a][:2] == goals[a]}
        a = None
        for a in list(at_drop):
            now += 1
            loaded[a] = False
            dest[a] = ""
            tid[a] = ""
            for n in names:
                steps_by[n].append(_hold(n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]))
            pending.discard(a)
        if not pending:
            break
        still = {a for a in pending if not pose[a][:2] != goals[a]}
        a = None
        if not still:
            break
        sub_goals = {n: pose[n][:2] for n in names}
        n = None
        for a in still:
            sub_goals[a] = goals[a]
        tl = _plan_turn_aware_joint(pose, sub_goals, still, static)
        if tl is not None:
            tl = {##ERROR##: list([n], [pose[n]]) for n in names if not tl_plan.get(n)}
            n = None
            t_before = now
            now = _apply_timelines_until_first_goal(steps_by, pose, tl, goals, still, now, loaded=loaded, dest=dest, tid=tid)
    
    ok = not pending
    if ok:
        print(f"[PIPE] delivery-stagger unloaded={len(batch)} t={now}", flush=True)
    return (now, ok)
    
    a = None
    
    a = None
    
    n = None; n = None
