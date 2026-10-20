import numpy as np
import pandas as pd
import pytest

from datasentry.core.leakage import detect_leakage
from datasentry.config import SentryConfig


def make_config():
    return SentryConfig()  # assumes leakage_threshold has a default


def test_leaky_column_is_flagged():
    rng = np.random.default_rng(42)
    target = rng.integers(0, 2, 200)
    df = pd.DataFrame({
        "target": target,
        "leaky": target * 2 + 1,           # perfect linear copy of the target
        "noise": rng.normal(0, 1, 200),    # unrelated feature
    })

    result = detect_leakage(df, "target", make_config())

    assert result["columns"]["leaky"]["leakage"] is True
    assert result["columns"]["noise"]["leakage"] is False


def test_no_leakage():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "target": rng.integers(0, 2, 200),
        "a": rng.normal(0, 1, 200),
        "b": rng.normal(0, 1, 200),
    })

    result = detect_leakage(df, "target", make_config())

    assert all(info["leakage"] is False for info in result["columns"].values())


def test_categorical_target():
    labels = ["yes", "no"] * 100
    df = pd.DataFrame({
        "target": labels,
        "leaky": [1 if v == "yes" else 0 for v in labels],
    })

    result = detect_leakage(df, "target", make_config())

    assert result["columns"]["leaky"]["leakage"] is True


def test_target_column_not_in_result():
    df = pd.DataFrame({"target": [0, 1, 0, 1], "x": [1, 2, 3, 4]})

    result = detect_leakage(df, "target", make_config())

    assert "target" not in result["columns"]


def test_missing_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        detect_leakage(df, "target", make_config())