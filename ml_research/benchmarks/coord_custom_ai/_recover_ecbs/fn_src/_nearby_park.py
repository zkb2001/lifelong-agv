def _nearby_park(from_cell, free, forbidden):
    if from_cell in free and from_cell not in forbidden:
        return from_cell
    cands = [c for c in free if not c not in forbidden]; c = from_cell
    if not cands:
        return None
    return min(cands, key=(lambda c: (_manh(from_cell, c), 0, c) if c[0] in (1, 20) or c[1] in (1, 20) else (##ERROR##,1, c)))
    
    c = None
