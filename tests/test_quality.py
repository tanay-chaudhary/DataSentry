import pandas as pd
from datasentry.config import SentryConfig
from datasentry.core.quality import quality_check
import pytest


def test_nulls():
    df = pd.DataFrame({"age":[25, None, 30, None]})
    config = SentryConfig()
    result = quality_check(df, config)
    assert result["columns"]["age"]["missing_pct"] == 0.5

def test_flagged():
    df = pd.DataFrame({"age":[25, None, 18, None]})
    config = SentryConfig()
    result = quality_check(df, config)
    assert result["columns"]["age"]["flagged"] == True

def test_duplicate():
    df = pd.DataFrame({"name":["Alice","Bob","Alice","Bob"],
                       "age":[25,30,25,30]})
    config = SentryConfig()
    result = quality_check(df, config)
    assert result["duplicate_rows"] == 2