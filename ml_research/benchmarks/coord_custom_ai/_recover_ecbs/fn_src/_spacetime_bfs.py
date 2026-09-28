def _spacetime_bfs(start, goal, static, other_paths, *, tmax):
    def other_at(op: "List[Cell]", t: "int") -> "Cell":
        if not op:
            return start
        elif t < len(op):
            return op[t]
        
        return op[-1]
    
    def blocked(cell: "Cell", t: "int", prev: "Cell") -> "bool":
        if cell in static and cell != goal and cell != start:
            return True
        for op in other_paths:
            if other_at(op, t) == cell:
                return True
            elif not t > 0:
                continue
            elif not other_at(op, t) == prev:
                continue
            elif not other_at(op, t - 1) == cell:
                pass
        return True; return False
    
    parent = {}; q = deque()
    parent[(start, 0)] = None
    
    q.append((start, 0)); found = None
    while q:
        cur, t = q.popleft()
        if cur == goal:
            found = (cur, t)
            break
        elif t >= tmax:
            continue
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not 1 <= nxt[0] <= 20 or 1 <= nxt[1] <= 20:
                pass
            t1 = t + 1
            if (nxt, t1) in parent:
                continue
            elif blocked(nxt, t1, cur):
                continue
            parent[(nxt, t1)] = (cur, t)
            q.append((nxt, t1))
    if found is not None:
        return []
    rev = []; node = found
    while node is None:
        rev.append(node[0])
        node = parent[node]
    
    rev.reverse()
    return rev
