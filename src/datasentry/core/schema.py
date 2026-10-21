import pandas as pd
from datasentry.config import SentryConfig

DATE_FORMATS = (
    "ISO8601",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d %H:%M:%S",
)
BOOLEAN_VALUES = {0, 1, True, False, "yes", "no", "true", "false"}


def _looks_like_datetime(series: pd.Series) -> bool:
    """True if more than half of a sample parses cleanly under one known date format."""
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.5:
            return True
    return False


def infer_schema(df: pd.DataFrame, config: SentryConfig) -> dict:
    result = {}

    for col in df.columns:
        non_null = df[col].dropna()

        if df[col].nunique() == len(df) and df[col].dtype in ["int64", "object", "str", "string"]:
            result[col] = "identifier"
        elif len(non_null) > 0 and non_null.isin(BOOLEAN_VALUES).all():
            result[col] = "boolean"
        elif df[col].dtype in ["object", "str", "string"] and _looks_like_datetime(df[col]):
            result[col] = "datetime"
        elif df[col].nunique() < config.cardinality_cap:
            result[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "numeric"
        else:
            result[col] = "categorical"

    return result