import pandas as pd
from datasentry.config import SentryConfig 

def summarize_distribution( df : pd.DataFrame, config : SentryConfig) -> dict:
    result_distribution = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            result_distribution[col] = {
            "mean" : float(df[col].mean()),
            "median" : float(df[col].median()),
            "std" : float(df[col].std()),
            "min" : float(df[col].min()),
            "max" : float(df[col].max()),
        }
        else:
            result_distribution[col] = {"value_counts" : df[col].value_counts().to_dict()}

           


    return result_distribution