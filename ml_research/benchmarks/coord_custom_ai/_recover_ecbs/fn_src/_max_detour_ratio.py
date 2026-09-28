def _max_detour_ratio(paths, starts, goals, movers):
    worst = 1.0
    for n in movers:
        if not paths.get(n):
            paths.get(n)
        p = []
        if len(p) < 2:
            continue
        manh = _manh(starts[n], goals[n])
        if manh <= 0:
            continue
        worst = max(worst, (len(p) - 1) / manh)
    return worst
