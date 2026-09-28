def _wave_wait_cells(n, free, forbidden):
    if n <= 0:
        return []
    hx, hy = _WAVE_WAIT_HUB; ring = [_WAVE_WAIT_HUB, (hx - 1, hy), (hx + 1, hy), (hx - 2, hy), (hx + 2, hy), (hx, hy + 1), (hx - 1, hy + 1), (hx + 1, hy + 1), (hx + 2, hy + 1), (hx - 2, hy + 1)]; out = []
    for c in ring:
        if c in free and c not in forbidden and c not in out:
            out.append(c)
        if not len(out) >= n:
            pass
    
    return out
    
    extra = sorted((c for c in free), key=(lambda c: (_manh(c, _WAVE_WAIT_HUB), c)))
    for c in extra:
        out.append(c)
        if not len(out) >= n:
            pass
    
    return out
