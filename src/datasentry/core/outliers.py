import pandas as pd
from datasentry.config import SentryConfig


def detect_outliers(df: pd.DataFrame, config: SentryConfig) -> dict:
    result_outliers = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype.kind != "b":
            series = df[col].dropna().astype(float)
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1
            lower = q1 - config.iqr_multiplier * iqr
            upper = q3 + config.iqr_multiplier * iqr

            mask = (series < lower) | (series > upper)

            result_outliers[col] = {
                "outliers": int(mask.sum()),
                "upper": float(upper),
                "lower": float(lower),
            }
    return result_outliers