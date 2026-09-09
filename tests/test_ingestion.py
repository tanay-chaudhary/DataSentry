import pandas as pd
from datasentry.config import SentryConfig
from datasentry.core.ingestion import ingest_csv
import pytest

def test_happy_path(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,age\nAlice,25\nBob,30\n")

    config = SentryConfig()
    df = ingest_csv(str(csv_file), config)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["name", "age"]


def test_sentinel_nulls(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,score\nAlice,NA\nBob,?\nCharlie,-\n")

    config = SentryConfig()
    df = ingest_csv(str(csv_file), config)

    assert pd.isnull(df["score"]).all()


def test_file_not_found():
    config = SentryConfig()

    with pytest.raises(FileNotFoundError):
        ingest_csv("nonexistent_file.csv", config)


def test_encoding_fallback(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_bytes("name\néclair\ncafé\n".encode("latin-1"))

    config = SentryConfig()
    df = ingest_csv(str(csv_file), config)

    assert len(df) == 2
    assert df["name"].iloc[0] == "éclair"