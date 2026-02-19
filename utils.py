def extract_random_clicks(result):
    """Return an integer random-clicks count from various possible result shapes.

    Accepts a dict-like `result` and looks for `random_clicks` or legacy
    `random_click`. Falls back to 0 for missing or non-int values.
    """
    if not isinstance(result, dict):
        return 0
    rc = result.get('random_clicks', result.get('random_click', None))
    if rc is None:
        return 0
    try:
        return int(rc)
    except Exception:
        try:
            return int(float(rc))
        except Exception:
            return 0


def get_unrevealed_cells(state, one_based: bool = True):
    """Return list of unrevealed cells from a 2D `state`.

    Each cell with value < 0 is considered unrevealed. By default returns
    1-based coordinates as used throughout the project: (x, y) where x is
    column and y is row. Set `one_based=False` to return 0-based (row, col).
    """
    cells = []
    if not state:
        return cells
    h = len(state)
    w = len(state[0]) if h > 0 else 0
    for r in range(h):
        for c in range(w):
            try:
                v = state[r][c]
            except Exception:
                v = -1
            if isinstance(v, int) and v < 0:
                if one_based:
                    cells.append((c + 1, r + 1))
                else:
                    cells.append((r, c))
    return cells
