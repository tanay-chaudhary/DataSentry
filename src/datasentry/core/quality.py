import pandas as pd
from datasentry.config import SentryConfig

def quality_check(df: pd.DataFrame, config: SentryConfig) -> dict:

    columns = {}

    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        flagged = missing_pct > config.null_threshold
        columns[col] = {"missing_pct": missing_pct, "flagged": flagged}

    duplicate_rows = df.duplicated().sum()

    return {"columns": columns, "duplicate_rows": duplicate_rows}