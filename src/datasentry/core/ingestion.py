import pandas as pd
from datasentry.config import SentryConfig
from pathlib import Path

def ingest_csv(path: str, config: SentryConfig) -> pd.DataFrame:                       # The path str specifies that the path given will be a string & ->pd.DataFrame should return a panda DataFrame
    if not(Path(path).exists()):
       raise FileNotFoundError(f"File not found {path}")

    try:
        df = pd.read_csv(path, na_values=config.sentinels)                             # nna_values=config.sentinels specifies whaich all values are to be replaced with NaN
    except UnicodeDecodeError:
        df = pd.read_csv(path, na_values=config.sentinels, encoding="latin-1")         # Checks for encoding

    return df