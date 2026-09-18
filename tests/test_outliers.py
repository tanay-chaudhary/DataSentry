import pandas as pd
from datasentry.config import SentryConfig
from datasentry.core.outliers import detect_outliers
import pytest

def test_outlier():
    df = pd.DataFrame({"score":[10,11,12,13,10,100]})
    config = SentryConfig()
    result = detect_outliers(df, config)
    assert result["score"]["outliers"] >=1

def test_skip_non_numeric():
    df = pd.DataFrame({"score": [10, 12, 11, 13], "name": ["Alice", "Bob", "Charlie", "Dave"]})
    config = SentryConfig()
    result = detect_outliers(df, config)
    assert "name" not in result
    assert "score" in result