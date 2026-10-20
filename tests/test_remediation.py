from datasentry.scoring.remediation import suggest_remediation


def clean_inputs():
    quality = {
        "columns": {"age": {"flagged": False}},
        "duplicate_rows": 0,
    }
    outliers = {"age": {"count": 0}}
    drift = {"age": {"p_value": 0.5, "drifted": False}}
    imbalance = {"imbalanced": False}
    leakage = {"columns": {"age": {"correlation": 0.1, "leakage": False}}}
    return quality, outliers, drift, imbalance, leakage


def test_clean_data_has_no_suggestions():
    assert suggest_remediation(*clean_inputs()) == []


def test_missing_data_suggestion():
    quality, outliers, drift, imbalance, leakage = clean_inputs()
    quality["columns"]["age"] = {"flagged": True, "null_pct": 0.6}

    result = suggest_remediation(quality, outliers, drift, imbalance, leakage)

    assert result == [
        "Column 'age': 60% missing — consider dropping this column or imputing values"
    ]


def test_each_problem_type_gets_a_suggestion():
    quality, outliers, drift, imbalance, leakage = clean_inputs()
    quality["columns"]["age"] = {"flagged": True, "null_pct": 0.6}
    quality["duplicate_rows"] = 5
    outliers["age"]["count"] = 3
    drift["age"]["drifted"] = True
    imbalance["imbalanced"] = True
    leakage["columns"]["age"] = {"correlation": 0.99, "leakage": True}

    result = suggest_remediation(quality, outliers, drift, imbalance, leakage)

    assert len(result) == 6
    assert any("3 outliers detected" in s for s in result)
    assert any("drift detected" in s for s in result)
    assert any("SMOTE" in s for s in result)
    assert any("corr=0.99" in s for s in result)
    assert any("5 duplicate rows" in s for s in result)


def test_suggestions_match_penalty_count():
    quality, outliers, drift, imbalance, leakage = clean_inputs()
    leakage["columns"] = {
        "a": {"correlation": 0.98, "leakage": True},
        "b": {"correlation": 0.97, "leakage": True},
    }

    result = suggest_remediation(quality, outliers, drift, imbalance, leakage)

    assert len(result) == 2