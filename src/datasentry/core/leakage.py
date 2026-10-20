import pandas as pd

from datasentry.config import SentryConfig


def detect_leakage(df: pd.DataFrame, target: str, config: SentryConfig) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    y = df[target]

    # Categorical targets are converted to numeric codes so correlation works
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes.where(y.notna())

    columns = {}

    for col in df.select_dtypes(include="number").columns:
        if col == target:
            continue

        corr = df[col].corr(y)

        if pd.isna(corr):  # constant column or no overlapping values
            continue

        columns[col] = {
            "correlation": float(corr),
            "leakage": bool(abs(corr) >= config.leakage_corr_ceil),
        }

    return {"columns": columns}