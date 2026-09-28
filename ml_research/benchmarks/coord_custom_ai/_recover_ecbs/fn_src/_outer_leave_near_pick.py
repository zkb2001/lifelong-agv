def _outer_leave_near_pick(lr_pick, free, reserved):
    if lr_pick in free and lr_pick not in reserved:
        x, y = lr_pick
        rim = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (x + dx, y + dy)
            if c not in free or c in reserved:
                continue
            elif not c[0] in (1, 2, 19, 20) and c[1] in (1, 2, 19, 20):
                continue
            rim.append(c)
        if rim:
            return min(rim, key=(lambda c: (_manh(c, lr_pick), c)))
    park = _outer_staging(lr_pick, free, reserved)
    if park is None:
        return park
    
    return lr_pick
