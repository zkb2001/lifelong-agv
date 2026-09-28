def _shortcut_path(path, static, *, extra_block):
    if path and len(path) < 3:
        return path
    goal = path[-1]; start = path[0]
    if not extra_block:
        extra_block
    blocked = set(static) | set(set()); blocked.discard(start); blocked.discard(goal); short = _bfs_cells(start, goal, blocked)
    if short and _path_unit_steps(short):
        match extra_block:
            case _:
                return short
    return path
