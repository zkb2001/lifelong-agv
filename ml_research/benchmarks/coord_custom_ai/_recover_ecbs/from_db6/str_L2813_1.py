_WAVE_WAIT_HUB: Cell = (6, 1)


def _wave_wait_cell(free: Set[Cell], forbidden: Set[Cell]) -> Optional[Cell]:
    """Preferred staging for next-wave / spare idles (user hub at (6,1))."""
    if _WAVE_WAIT_HUB in free and _WAVE_WAIT_HUB not in forbidden:
        return _WAVE_WAIT_HUB
    return _nearby_park(_WAVE_WAIT_HUB, free, forbidden)


def _send_idles_wave_wait(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    names: List[str],
    agvs: List[str],
    free: Set[Cell],
    static: Set[Cell],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    reserved: Optional[Set[Cell]] = None,
) -> int:
    """Move spare idles to wave-wait hub so they do not block pickup/delivery."""
    reserved = set(reserved or ())
    reserved |= {pose[n][:2] for n in names}
    hub = _wave_wait_cell(free, reserved)
    if hub is None:
        return now
    movers = [n for n in agvs if n in names and pose[n][:2] != hub]
    if not movers:
        return now
    cg = {n: pose[n][:2] for n in names}
    for n in movers:
        cg[n] = hub
    tl = _plan_turn_aware_joint(
        pose, cg, set(movers), static, tmax_scale=2.5, prefer_outer_ring=True
    )
    if tl is None:
        tl = _plan_turn_aware_joint(
            pose, cg, set(movers), static, tmax_scale=4.0
        )
    if tl is None:
        return now
    now = _apply_pose_timelines(
        steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid
    )
    print(
        f"[PIPE] wave-wait {len(movers)} -> {hub} t={now}",
        flush=True,
    )
    return now


def _pickup_load_agv(
    agv: str,
    task: dict,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    names: List[str],
    now: int,
) -> int:
    loaded[agv] = True
    dest[agv] = str(task.get("destination") or "")
    tid[agv] = str(task["task_id"])
    now += 1
    for n in names:
        steps_by[n].append(
            _hold(
                n,
                pose[n],
                now,
                loaded=loaded[n],
                dest=dest[n],
                tid=tid[n],
            )
        )
    return now


def _pickup_stagger_turn_aware(
    *,
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    names: List[str],
    assigned: Dict[str, dict],
    goals: Dict[str, Cell],
    static: Set[Cell],
    now: int,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    set_pickup_goals,
) -> Tuple[int, bool]:
    """Arrive-at-pick → load immediately; no wait for slow peers (split waves)."""
    pending = set(assigned.keys())
    rounds = 0
    while pending and rounds < 120:
        rounds += 1
        sub_goals = set_pickup_goals({a: assigned[a] for a in pending})
        at_pick = {a for a in pending if pose[a][:2] == sub_goals[a]}
        for a in list(at_pick):
            now = _pickup_load_agv(
                a,
                assigned[a],
                loaded=loaded,
                dest=dest,
                tid=tid,
                steps_by=steps_by,
                pose=pose,
                names=names,
                now=now,
            )
            pending.discard(a)
        if not pending:
            break
        still = {a for a in pending if pose[a][:2] != sub_goals[a]}
        if not still:
            for a in list(pending):
                now = _pickup_load_agv(
                    a,
                    assigned[a],
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    steps_by=steps_by,
                    pose=pose,
                    names=names,
                    now=now,
                )
            pending.clear()
            break
        tl = _plan_turn_aware_joint(pose, sub_goals, still, static)
        if tl is None:
            return now, False
        t_before = now
        now = _apply_timelines_until_first_goal(
            steps_by,
            pose,
            tl,
            sub_goals,
            still,
            now,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        if now == t_before:
            now = _apply_pose_timelines(
                steps_by,
                pose,
                tl,
                now,
                loaded=loaded,
                dest=dest,
                tid=tid,
            )
    ok = not pending
    if ok:
        print(
            f"[PIPE] pickup-stagger loaded={len(assigned)} t={now}",
            flush=True,
        )
    return now, ok


def _pickup_stagger_paths(
    *,
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    names: List[str],
    assigned: Dict[str, dict],
    paths: Dict[str, List[Cell]],
    goals: Dict[str, Cell],
    now: int,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> Tuple[int, bool]:
    pending = set(assigned.keys())
    rounds = 0
    while pending and rounds < 120:
        rounds += 1
        at_pick = {a for a in pending if pose[a][:2] == goals[a]}
        for a in list(at_pick):
            now = _pickup_load_agv(
                a,
                assigned[a],
                loaded=loaded,
                dest=dest,
                tid=tid,
                steps_by=steps_by,
                pose=pose,
                names=names,
                now=now,
            )
            pending.discard(a)
        if not pending:
            break
        still = {a for a in pending if pose[a][:2] != goals[a]}
        if not still:
            for a in list(pending):
                now = _pickup_load_agv(
                    a,
                    assigned[a],
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    steps_by=steps_by,
                    pose=pose,
                    names=names,
                    now=now,
                )
            pending.clear()
            break
        t_before = now
        now = _apply_paths_until_first_goal(
            steps_by,
            pose,
            paths,
            goals,
            still,
            now,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        if now == t_before:
            now = _apply_paths(
                steps_by, pose, paths, now, loaded=loaded, dest=dest, tid=tid
            )
    ok = not pending
    if ok:
        print(
            f"[PIPE] pickup-stagger(paths) loaded={len(assigned)} t={now}",
            flush=True,
        )
    return now, ok


def _nearby_park(