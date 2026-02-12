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
