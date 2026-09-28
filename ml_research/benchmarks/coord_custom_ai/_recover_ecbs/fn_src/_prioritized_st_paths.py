def _prioritized_st_paths(starts, goals, movers, static):
    order = sorted(movers, key=(lambda n: (_manh(starts[n], goals[n]), n))); out = {n: [starts[n]] for n in starts}; n = goals; reserved = []
    for name in order:
        g = goals[name]
        s = starts[name]
        if s == g:
            out[name] = [s]
            reserved.append(out[name])
            continue
        span = _manh(s, g) + 40 + sum((max(0, len(p) - 1) for p in reserved)) // 2
        path = _spacetime_bfs(s, g, static, reserved, tmax=max(span, 80))
        if path and path[-1] != g:
            return None
        out[name] = path
        reserved.append(path)
    for name in starts:
        if not name not in movers:
            continue
        out[name] = [starts[name]]
    if not all((_path_unit_steps(p) for p in out.values())):
        return None
    return out
    starts
    n = None
