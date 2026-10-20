from datasentry.scoring.readiness import compute_readiness


def clean_inputs():
    quality = {
        "columns": {"age": {"flagged": False}, "income": {"flagged": False}},
        "duplicate_rows": 0,
    }
    outliers = {"age": {"count": 0}}
    drift = {"age": {"p_value": 0.5, "drifted": False}}
    imbalance = {"imbalanced": False}
    leakage = {"columns": {"age": {"correlation": 0.1, "leakage": False}}}
    return quality, outliers, drift, imbalance, leakage


def test_clean_data_scores_100():
    result = compute_readiness(*clean_inputs())

    assert result["score"] == 100
    assert result["penalties"] == []


def test_known_deductions():
    quality, outliers, drift, imbalance, leakage = clean_inputs()
    quality["columns"]["age"]["flagged"] = True
    quality["columns"]["income"]["flagged"] = True      # 2 x -5 = -10
    leakage["columns"]["age"]["leakage"] = True         # 1 x -15

    result = compute_readiness(quality, outliers, drift, imbalance, leakage)

    assert result["score"] == 75
    assert "Missing data: 2 columns flagged (-10)" in result["penalties"]
    assert "Leakage: age (-15)" in result["penalties"]
    assert len(result["penalties"]) == 2


def test_every_penalty_type():
    quality, outliers, drift, imbalance, leakage = clean_inputs()
    quality["columns"]["age"]["flagged"] = True         # -5
    quality["duplicate_rows"] = 12                      # -5 (once)
    outliers["age"]["count"] = 3                        # -3
    drift["age"]["drifted"] = True                      # -5
    imbalance["imbalanced"] = True                      # -10
    leakage["columns"]["age"]["leakage"] = True         # -15

    result = compute_readiness(quality, outliers, drift, imbalance, leakage)

    assert result["score"] == 100 - 5 - 5 - 3 - 5 - 10 - 15
    assert len(result["penalties"]) == 6


def test_score_never_goes_below_zero():
    quality, outliers, drift, imbalance, leakage = clean_inputs()
    leakage["columns"] = {
        f"col{i}": {"correlation": 0.99, "leakage": True} for i in range(10)
    }  # 10 x -15 = -150

    result = compute_readiness(quality, outliers, drift, imbalance, leakage)

    assert result["score"] == 0