def _leave_from_unload(lr_pick, unload, free, reserved, static):
    x, y = lr_pick; cands = []
    if lr_pick in free and lr_pick not in reserved:
        cands.append(lr_pick)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        c = (x + dx, y + dy)
        if c not in free or c in reserved:
            continue
        elif not c[0] in (1, 2, 19, 20) and c[1] in (1, 2, 19, 20) and _manh(c, lr_pick) <= 2:
            continue
        cands.append(c)
    for c in free:
        if c in reserved or c in cands:
            continue
        elif _manh(c, lr_pick) > 5:
            continue
        elif not c[0] in (1, 2, 19, 20) and c[1] in (1, 2, 19, 20):
            continue
        cands.append(c)
    if not cands:
        return _outer_leave_near_pick(lr_pick, free, reserved)
    
    return min(cands, key=(lambda c: (_bfs_len(unload, c, static), _bfs_len(c, lr_pick, static), _manh(c, lr_pick), c)))
