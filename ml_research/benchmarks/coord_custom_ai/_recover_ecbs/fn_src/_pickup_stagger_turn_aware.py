def _pickup_stagger_turn_aware():
    if not hub_movers:
        hub_movers
    hub_movers = []
    if not hub_goals:
        hub_goals
    hub_goals = {}; pending = set(assigned.keys()); rounds = 0
    while pending:
        match hub_goals:
            case 120 as rounds if pose[a][:2] == sub_goals[a] and pose[a][:2] != sub_goals[a] and pose[h][:2] == hub_goals.get(h):
                return (now, False)
        t_before = now
        if still:
            now = _apply_timelines_until_first_goal(steps_by, pose, tl, sub_goals, still, now, loaded=loaded, dest=dest, tid=tid)
        elif hub_still:
            now = _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid)
        if still:
            pass
    _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid)
    
    a = _plan_turn_aware_joint(pose, sub_goals, movers, static)
    still | hub_still
    
    a = _pickup_load_agv(a, assigned[a], loaded=loaded, dest=dest, tid=tid, steps_by=steps_by, pose=pose, names=names, now=now); a = not pending
