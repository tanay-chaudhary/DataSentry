import pandas as pd
import pytest
from datasentry.config import SentryConfig
from datasentry.core.schema import infer_schema  

def test_identifier():
    df = pd.DataFrame({"user_id": [1, 2, 3, 4, 5]})
    config = SentryConfig()
    result = infer_schema(df, config)
    assert result["user_id"] == "identifier"

def test_boolean():
    df = pd.DataFrame({"active":["yes","no","yes","no"] })
    config = SentryConfig()
    result = infer_schema(df, config)
    assert result["active"] == "boolean"

def test_datetime():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-02-15", "2024-03-20", "2024-01-01"]})
    config = SentryConfig()
    result = infer_schema(df, config)
    assert result["date"] == "datetime"

def test_categorical():
    df = pd.DataFrame({"color": ["red", "blue", "red", "green", "blue", "red"]})
    config = SentryConfig()
    result = infer_schema(df, config)
    assert result["color"] == "categorical"

def test_numeric():
    df = pd.DataFrame({"salary": list(range(51)) + [0, 1, 2]})
    config = SentryConfig()
    result = infer_schema(df, config)
    assert result["salary"] == "numeric"