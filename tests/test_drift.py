import numpy as np
import pandas as pd
from datasentry.core.drift import detect_drift
from datasentry.config import SentryConfig


def make_config():
    return SentryConfig()  # assumes drift_alpha has a default (e.g. 0.05)


def test_no_drift_numeric():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"age": rng.normal(50, 10, 500)})
    cur = pd.DataFrame({"age": rng.normal(50, 10, 500)})

    result = detect_drift(ref, cur, make_config())

    assert result["age"]["drifted"] is False


def test_drift_numeric():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"age": rng.normal(50, 10, 500)})
    cur = pd.DataFrame({"age": rng.normal(70, 10, 500)})

    result = detect_drift(ref, cur, make_config())

    assert result["age"]["drifted"] is True
    assert result["age"]["p_value"] < 0.05


def test_column_only_in_one_dataframe_is_skipped():
    ref = pd.DataFrame({"age": [1, 2, 3], "extra": [1, 2, 3]})
    cur = pd.DataFrame({"age": [1, 2, 3]})

    result = detect_drift(ref, cur, make_config())

    assert "extra" not in result
    assert "age" in result


def test_drift_categorical():
    ref = pd.DataFrame({"plan": ["free"] * 450 + ["paid"] * 50})
    cur = pd.DataFrame({"plan": ["free"] * 250 + ["paid"] * 250})

    result = detect_drift(ref, cur, make_config())

    assert result["plan"]["drifted"] is True


def test_no_drift_categorical():
    ref = pd.DataFrame({"plan": ["free"] * 250 + ["paid"] * 250})
    cur = pd.DataFrame({"plan": ["free"] * 255 + ["paid"] * 245})

    result = detect_drift(ref, cur, make_config())

    assert result["plan"]["drifted"] is False