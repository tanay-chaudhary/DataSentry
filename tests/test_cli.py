import json

import numpy as np
import pandas as pd

from datasentry.cli import main


def make_csv(tmp_path, name="data.csv", n=200, shift=0):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.normal(40 + shift, 10, n),
        "income": rng.normal(50000, 8000, n),
        "plan": ["free"] * int(n * 0.9) + ["paid"] * (n - int(n * 0.9)),
    })
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_basic_run(tmp_path, capsys):
    code = main([make_csv(tmp_path)])

    assert code == 0
    assert "DataSentry Report" in capsys.readouterr().out


def test_json_export(tmp_path):
    out = tmp_path / "report.json"

    code = main([make_csv(tmp_path), "--json", str(out)])

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "readiness" in data
    assert data["drift"] == {}           # skipped: no --reference


def test_html_export(tmp_path):
    out = tmp_path / "report.html"

    code = main([make_csv(tmp_path), "--html", str(out)])

    assert code == 0
    assert "DataSentry Report" in out.read_text(encoding="utf-8")


def test_target_enables_imbalance_and_leakage(tmp_path):
    out = tmp_path / "report.json"

    code = main([make_csv(tmp_path), "--target", "plan", "--json", str(out)])

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["imbalance"]["class_distribution"] != {}
    assert data["leakage"]["columns"] != {}


def test_reference_enables_drift(tmp_path):
    current = make_csv(tmp_path, "current.csv", shift=30)
    reference = make_csv(tmp_path, "reference.csv", shift=0)
    out = tmp_path / "report.json"

    code = main([current, "--reference", reference, "--json", str(out)])

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["drift"]["age"]["drifted"] is True


def test_missing_file_returns_error(tmp_path, capsys):
    code = main([str(tmp_path / "nope.csv")])

    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_unknown_target_returns_error(tmp_path, capsys):
    code = main([make_csv(tmp_path), "--target", "does_not_exist"])

    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_empty_csv_returns_error(tmp_path, capsys):
    empty = tmp_path / "empty.csv"
    empty.write_text("")

    code = main([str(empty)])

    assert code == 1
    assert "empty.csv" in capsys.readouterr().err


def test_directory_instead_of_file_returns_error(tmp_path, capsys):
    code = main([str(tmp_path)])

    assert code == 1
    assert "Error" in capsys.readouterr().err


def test_missing_reference_file_returns_error(tmp_path, capsys):
    code = main([make_csv(tmp_path), "--reference", str(tmp_path / "nope.csv")])

    assert code == 1
    assert "reference" in capsys.readouterr().err