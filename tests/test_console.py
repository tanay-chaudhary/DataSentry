import io

from rich.console import Console

from datasentry.reporters.console import print_report


def make_inputs():
    schema = {"age": "numeric", "plan": "categorical"}
    quality = {
        "columns": {
            "age": {"flagged": True, "null_pct": 0.5},
            "plan": {"flagged": False, "null_pct": 0.0},
        },
        "duplicate_rows": 5,
    }
    outliers = {"age": {"lower": 1.5, "upper": 90.0, "count": 3, "indices": [1, 2, 3]}}
    distribution = {"age": {"mean": 42.5, "std": 10.1}}
    imbalance = {
        "class_distribution": {"free": 0.95, "paid": 0.05},
        "imbalanced": True,
        "minority_classes": ["paid"],
    }
    drift = {"age": {"p_value": 0.03, "drifted": True}}
    leakage = {"columns": {"age": {"correlation": 0.99, "leakage": True}}}
    readiness = {"score": 72, "penalties": ["Imbalance detected (-10)"]}
    suggestions = ["Target is imbalanced — apply oversampling (SMOTE) or use class weights"]
    return (schema, quality, outliers, distribution, imbalance,
            drift, leakage, readiness, suggestions)


def render(*args) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=120, force_terminal=False)
    print_report(*args, console=console)
    return buffer.getvalue()


def test_report_contains_every_section():
    output = render(*make_inputs())

    for title in ("DataSentry Report", "Schema", "Quality", "Outliers",
                  "Distribution", "Class Imbalance", "Drift", "Leakage",
                  "ML Readiness", "Remediation"):
        assert title in output


def test_report_shows_key_values():
    output = render(*make_inputs())

    assert "Score: 72/100" in output
    assert "50.0%" in output          # missing percentage for 'age'
    assert "0.99" in output           # leakage correlation
    assert "95.0%" in output          # class proportion
    assert "SMOTE" in output


def test_clean_report_says_nothing_to_fix():
    args = list(make_inputs())
    args[-1] = []                     # no suggestions
    output = render(*args)

    assert "Nothing to fix" in output


def test_square_brackets_in_column_names_do_not_break_output():
    args = list(make_inputs())
    args[0] = {"price [usd]": "numeric"}
    output = render(*args)

    assert "price [usd]" in output