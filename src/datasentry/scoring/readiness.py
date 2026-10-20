def _count_where(items: dict, predicate) -> int:
    """Count how many values in a dict satisfy the predicate."""
    return sum(1 for info in items.values() if predicate(info))


def _plural(n: int, word: str = "column") -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def compute_readiness(
    quality_result: dict,
    outlier_result: dict,
    drift_result: dict,
    imbalance_result: dict,
    leakage_result: dict,
) -> dict:
    score = 100
    penalties = []

    # 1. Missing data: -5 per flagged column
    flagged = _count_where(quality_result["columns"], lambda info: info["flagged"])
    if flagged > 0:
        points = 5 * flagged
        score -= points
        penalties.append(f"Missing data: {_plural(flagged)} flagged (-{points})")

    # 2. Duplicate rows: -5 once
    if quality_result["duplicate_rows"] > 0:
        score -= 5
        penalties.append("Duplicate rows found (-5)")

    # 3. Outliers: -3 per column with at least one outlier
    with_outliers = _count_where(
        outlier_result,
        lambda info: info.get("count", info.get("outliers", 0)) > 0,
    )
    if with_outliers > 0:
        points = 3 * with_outliers
        score -= points
        penalties.append(f"Outliers: {_plural(with_outliers)} (-{points})")

    # 4. Drift: -5 per drifted column
    drifted = _count_where(drift_result, lambda info: info["drifted"])
    if drifted > 0:
        points = 5 * drifted
        score -= points
        penalties.append(f"Drift: {_plural(drifted)} (-{points})")

    # 5. Imbalance: -10 once
    if imbalance_result["imbalanced"]:
        score -= 10
        penalties.append("Imbalance detected (-10)")

    # 6. Leakage: -15 per leaked column, one message per column
    for col, info in leakage_result["columns"].items():
        if info["leakage"]:
            score -= 15
            penalties.append(f"Leakage: {col} (-15)")

    score = max(0, score)
    return {"score": score, "penalties": penalties}