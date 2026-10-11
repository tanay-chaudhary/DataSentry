import pandas as pd
from datasentry.config import SentryConfig
from datasentry.core.distribution import summarize_distribution
import pytest

def test_numeric_column():
    df = pd.DataFrame({"salary": [10, 20, 30, 40, 50]})
    config = SentryConfig()
    result = summarize_distribution(df, config)
    stats = result["salary"]

    assert set(stats) == {"mean", "median", "std", "min", "max"}
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["std"] == pytest.approx(15.811, rel=1e-3)


def test_categorical_column():
    df = pd.DataFrame({"color": ["red", "red", "blue"]})
    config = SentryConfig()
    result = summarize_distribution(df, config)
    assert result["color"]["value_counts"] == {"red": 2, "blue": 1}


def test_mixed_columns():
    df = pd.DataFrame({"salary": [10, 20, 30],"city": ["Pune", "Pune", "Delhi"],})
    config = SentryConfig()
    result = summarize_distribution(df, config)
    assert "mean" in result["salary"]
    assert "value_counts" in result["city"]
    assert "value_counts" not in result["salary"]