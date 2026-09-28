def _bfs_cells(start, goal, blocked):
    if start == goal:
        return [start]
    parent = {start: None}; q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not 1 <= nxt[0] <= 20 or 1 <= nxt[1] <= 20:
                pass
            elif nxt in parent:
                continue
            elif nxt in blocked and nxt != goal:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return []
    rev = []; cur_o = goal
    while cur_o is None:
        rev.append(cur_o)
        cur_o = parent[cur_o]
    
    rev.reverse()
    return rev
