import pandas as pd
from datasentry.config import SentryConfig  # adjust the import to match your project layout


def detect_imbalance(df: pd.DataFrame, target: str, config: SentryConfig) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    distribution = df[target].value_counts(normalize=True).to_dict()

    minority_classes = [
        label for label, proportion in distribution.items()
        if proportion < config.imbalance_floor
    ]

    return {
        "class_distribution": distribution,
        "imbalanced": len(minority_classes) > 0,
        "minority_classes": minority_classes,
    }