def _paths_conflict(paths):
    if len(paths) < 2:
        return False
    names = sorted(paths); T = max((len(paths[n]) for n in names))
    def cell_at(n: "str", t: "int") -> "Cell":
        p = paths[n]
        if t < len(p):
            return p[t]
        
        return p[-1]
    
    for t in range(T):
        occ = {}
        for n in names:
            c = cell_at(n, t)
            if c in occ:
                return True
            occ[c] = n
        if not t + 1 < T:
            continue
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cb0 = cell_at(b, t)
                ca0 = cell_at(a, t)
                cb1 = cell_at(b, t + 1)
                ca1 = cell_at(a, t + 1)
                if not ca0 == cb1:
                    continue
                elif not cb0 == ca1:
                    continue
                elif not ca0 != cb0:
                    pass
                paths
            return True
    return False
