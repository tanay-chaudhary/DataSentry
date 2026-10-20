import json
from pathlib import Path

import numpy as np
import pandas as pd


def _to_serializable(obj):
    """Fallback for objects the json module can't handle on its own."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_json(results: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(results, indent=2, default=_to_serializable, ensure_ascii=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)