import pandas as pd
from datasentry.config import SentryConfig


def infer_schema(df: pd.DataFrame, config: SentryConfig) -> dict:

    result = {}

    # df = pd.DataFrame() already passed in function call
    for col in df.columns:
        unique_count = df[col].nunique()

        if df[col].nunique() == len(df) and df[col].dtype in ["int64", "object", "str", "string"]:
            result[col] = "identifier"
        elif df[col].dropna().isin({0, 1, True, False, "yes", "no", "true", "false"}).all():
            result[col] = "boolean"
        elif df[col].dtype in ["object", "str", "string"] and pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.5:
            result[col] = "datetime"
        elif df[col].nunique() < config.cardinality_cap:
            result[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            result[col] = "numeric"
        else:
            result[col] = "categorical"

    return result