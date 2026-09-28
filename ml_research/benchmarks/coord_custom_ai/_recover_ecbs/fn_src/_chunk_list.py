def _chunk_list(items, size):
    if size <= 0:
        return [list(items)]
    i = None
    return [list(items[i:i + size]) for i in range(0, len(items), size)]
    
    i = None
