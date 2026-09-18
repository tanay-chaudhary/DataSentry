import pandas as pd
from datasentry.config import SentryConfig

def detect_outliers(df : pd.DataFrame, config: SentryConfig) -> dict:
    result_outliers = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1= df[col].quantile(0.25)
            q3=df[col].quantile(0.75)

            iqr=q3-q1
            lower = q1 - config.iqr_multiplier * iqr
            upper = q3 + config.iqr_multiplier * iqr

            outliers = df[(df[col]<lower) | (df[col]>upper)]

            result_outliers[col] = {"outliers": len(outliers),
                                    "upper": upper,
                                    "lower": lower}
    return result_outliers