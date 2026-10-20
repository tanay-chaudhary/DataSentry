from datasentry.reporters.html import export_html


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


def render(tmp_path, inputs=None):
    out = tmp_path / "report.html"
    export_html(*(inputs or make_inputs()), str(out))
    return out.read_text(encoding="utf-8")


def test_report_contains_every_section(tmp_path):
    html = render(tmp_path)

    for title in ("DataSentry Report", "ML Readiness", "Schema", "Quality",
                  "Outliers", "Distribution", "Class Imbalance", "Drift",
                  "Leakage", "Remediation"):
        assert title in html


def test_report_shows_key_values(tmp_path):
    html = render(tmp_path)

    assert "72/100" in html
    assert "50.0%" in html        # missing percentage for 'age'
    assert "0.99" in html         # leakage correlation
    assert "95.0%" in html        # class proportion
    assert "SMOTE" in html


def test_column_names_are_html_escaped(tmp_path):
    inputs = list(make_inputs())
    inputs[0] = {"<script>alert(1)</script>": "numeric"}

    html = render(tmp_path, inputs)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_clean_report_says_nothing_to_fix(tmp_path):
    inputs = list(make_inputs())
    inputs[-1] = []

    html = render(tmp_path, inputs)

    assert "Nothing to fix" in html


def test_creates_missing_folders(tmp_path):
    out = tmp_path / "nested" / "report.html"

    export_html(*make_inputs(), str(out))

    assert out.exists()