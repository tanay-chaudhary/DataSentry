from dataclasses import dataclass, field

@dataclass
class SentryConfig:
    null_threshold: float = 0.4                                                                                     # flag a coloumn if more than 40% value is missing
    cardinality_cap: int = 50                                                                                       # If a coloumn has less than 50 unique values, treat it has categorical
    iqr_multiplier: float = 1.5                                                                                     # How far from the middle a value must be to count as an outlier
    drift_alpha: float = 0.05                                                                                       # Significance level for drift tests — below this means "the data has shifted"
    imbalance_floor: float = 0.1                                                                                    # If the rarest class is less than 10% of the data, flag it as imbalanced
    leakage_corr_ceil: float = 0.95                                                                                 # If a feature correlates above 95% with the target, flag it as potential leakage
    sentinels: list[str] = field(default_factory=lambda: ["", "NA", "N/A", "null", "none", "-", "?"])               # Values that should be treated as missing data when reading a CSV