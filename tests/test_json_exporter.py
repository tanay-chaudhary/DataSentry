import json

import numpy as np

from datasentry.reporters.json_exporter import export_json


def test_writes_valid_json(tmp_path):
    results = {"score": 72, "penalties": ["Imbalance detected (-10)"]}
    out = tmp_path / "report.json"

    export_json(results, str(out))

    assert json.loads(out.read_text(encoding="utf-8")) == results


def test_handles_numpy_types(tmp_path):
    results = {
        "p_value": np.float64(0.03),
        "drifted": np.bool_(True),
        "count": np.int64(5),
        "values": np.array([1, 2, 3]),
    }
    out = tmp_path / "report.json"

    export_json(results, str(out))

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == {"p_value": 0.03, "drifted": True, "count": 5, "values": [1, 2, 3]}


def test_creates_missing_folders(tmp_path):
    out = tmp_path / "nested" / "folder" / "report.json"

    export_json({"score": 100}, str(out))

    assert out.exists()


def test_keeps_non_ascii_text(tmp_path):
    results = {"tip": "Column 'age': 60% missing — consider dropping"}
    out = tmp_path / "report.json"

    export_json(results, str(out))

    assert "—" in out.read_text(encoding="utf-8")