import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

from datasentry.config import SentryConfig


def detect_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    config: SentryConfig,
) -> dict:
    result = {}

    for col in df_reference.columns:
        if col not in df_current.columns:
            continue

        ref = df_reference[col].dropna()
        cur = df_current[col].dropna()

        if ref.empty or cur.empty:
            continue

        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
            _, p_value = ks_2samp(ref, cur)
        else:
            values = pd.concat([ref, cur], ignore_index=True)
            source = ["reference"] * len(ref) + ["current"] * len(cur)
            table = pd.crosstab(values, source)

            if table.shape[0] < 2:
                p_value = 1.0  # only one category overall, so no drift is possible
            else:
                _, p_value, _, _ = chi2_contingency(table)

        result[col] = {
            "p_value": float(p_value),
            "drifted": bool(p_value < config.drift_alpha),
        }

    return result