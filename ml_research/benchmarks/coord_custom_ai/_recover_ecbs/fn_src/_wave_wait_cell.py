def _wave_wait_cell(free, forbidden):
    cells = _wave_wait_cells(1, free, forbidden)
    if cells:
        return cells[0]
