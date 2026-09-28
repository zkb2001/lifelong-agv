def _wave_wait_ring(free, forbidden):
    ring = set(_wave_wait_cells(8, free, forbidden))
    for x in range(3, 11):
        c = (x, 1)
        if not c in free:
            continue
        elif not c not in forbidden:
            continue
        ring.add(c)
    return ring
