def suggest_remediation(
    quality_result: dict,
    outlier_result: dict,
    drift_result: dict,
    imbalance_result: dict,
    leakage_result: dict,
) -> list:
    suggestions = []

    # 1. Missing data: one suggestion per flagged column
    for col, info in quality_result["columns"].items():
        if info["flagged"]:
            pct = _missing_percent(info)
            detail = f"{pct:.0f}% missing" if pct is not None else "high missing rate"
            suggestions.append(
                f"Column '{col}': {detail} — consider dropping this column or imputing values"
            )

    # 2. Outliers: one suggestion per column with at least one outlier
    for col, info in outlier_result.items():
        count = info.get("count", info.get("outliers", 0))
        if count > 0:
            noun = "outlier" if count == 1 else "outliers"
            suggestions.append(
                f"Column '{col}': {count} {noun} detected — review for data entry errors or apply capping"
            )

    # 3. Drift: one suggestion per drifted column
    for col, info in drift_result.items():
        if info["drifted"]:
            suggestions.append(
                f"Column '{col}': drift detected — retrain model with recent data"
            )

    # 4. Imbalance: once
    if imbalance_result["imbalanced"]:
        suggestions.append(
            "Target is imbalanced — apply oversampling (SMOTE) or use class weights"
        )

    # 5. Leakage: one suggestion per leaked column
    for col, info in leakage_result["columns"].items():
        if info["leakage"]:
            suggestions.append(
                f"Column '{col}': leakage detected (corr={info['correlation']:.2f}) "
                "— remove this feature before training"
            )

    # 6. Duplicates: once
    duplicates = quality_result["duplicate_rows"]
    if duplicates > 0:
        suggestions.append(
            f"{duplicates} duplicate rows found — consider removing duplicates with df.drop_duplicates()"
        )

    return suggestions


def _missing_percent(info: dict):
    """Return the missing percentage (0-100) if the quality result provides it."""
    for key in ("null_pct", "missing_pct", "null_ratio", "missing_ratio"):
        if key in info:
            value = info[key]
            return value * 100 if value <= 1 else value
    return None