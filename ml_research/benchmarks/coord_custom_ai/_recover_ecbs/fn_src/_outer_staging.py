def _outer_staging(near, free, forbidden):
    cands = [c for c in free if not c not in forbidden]; c = near
    if not cands:
        return None
    outer = [c for c in cands if c[1] in (1, 2, 19, 20)]; c = None; pool = outer or cands
    return min(pool, key=(lambda c: (_manh(c, near), c)))
    
    c = None; c = None
