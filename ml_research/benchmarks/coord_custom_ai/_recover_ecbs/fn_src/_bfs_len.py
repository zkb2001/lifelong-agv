def _bfs_len(start, goal, static):
    if isinstance(start, list):
        start = tuple(start)
    if isinstance(goal, list):
        goal = tuple(goal)
    if start == goal:
        return 0
    blocked = set(static); blocked.discard(start); blocked.discard(goal); path = _bfs_cells(start, goal, blocked)
    if not path:
        return 1_000_000
    return len(path) - 1
