import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from datasentry.config import SentryConfig
from datasentry.core.distribution import summarize_distribution
from datasentry.core.drift import detect_drift
from datasentry.core.imbalance import detect_imbalance
from datasentry.core.leakage import detect_leakage
from datasentry.core.outliers import detect_outliers
from datasentry.core.quality import quality_check
from datasentry.core.schema import infer_schema
from datasentry.reporters.html import export_html
from datasentry.reporters.json_exporter import export_json
from datasentry.scoring.readiness import compute_readiness
from datasentry.scoring.remediation import suggest_remediation
from datasentry.ui.components import (
    show_distribution,
    show_drift,
    show_imbalance,
    show_leakage,
    show_outliers,
    show_quality,
    show_readiness,
    show_remediation,
    show_schema,
)

EMPTY_IMBALANCE = {"class_distribution": {}, "imbalanced": False, "minority_classes": []}
EMPTY_LEAKAGE = {"columns": {}}


def read_csv(uploaded, config: SentryConfig, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded, na_values=config.sentinels)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the {label} file as a CSV: {exc}")
        st.stop()


def build_downloads(results: dict, html_args: tuple) -> tuple[str, str]:
    """Reuse the file exporters, then read the files back as text for the download buttons."""
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "report.json"
        html_path = Path(tmp) / "report.html"
        export_json(results, str(json_path))
        export_html(*html_args, str(html_path))
        return (
            json_path.read_text(encoding="utf-8"),
            html_path.read_text(encoding="utf-8"),
        )


st.set_page_config(page_title="DataSentry", layout="wide")
st.title("DataSentry")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
target = st.text_input("Target column (optional)").strip()
reference_file = st.file_uploader("Reference CSV for drift (optional)", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV file to start the audit.")
    st.stop()

config = SentryConfig()
df = read_csv(uploaded_file, config, "main")

# Checks that always run
schema = infer_schema(df, config)
quality_result = quality_check(df, config)
outlier_result = detect_outliers(df, config)
distribution_result = summarize_distribution(df, config)

# Optional: imbalance and leakage need a target column
imbalance_result = EMPTY_IMBALANCE
leakage_result = EMPTY_LEAKAGE
if target:
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in the uploaded file.")
    else:
        if df[target].nunique() > config.cardinality_cap:
            st.warning(
                f"'{target}' has more than {config.cardinality_cap} unique values, "
                "so it doesn't look categorical. Skipping the imbalance check."
            )
        else:
            imbalance_result = detect_imbalance(df, target, config)
        leakage_result = detect_leakage(df, target, config)

# Optional: drift needs a reference file
drift_result = {}
if reference_file is not None:
    df_reference = read_csv(reference_file, config, "reference")
    drift_result = detect_drift(df_reference, df, config)

readiness_result = compute_readiness(
    quality_result, outlier_result, drift_result, imbalance_result, leakage_result
)
suggestions = suggest_remediation(
    quality_result, outlier_result, drift_result, imbalance_result, leakage_result
)

# Display
show_readiness(readiness_result)
show_schema(schema)
show_quality(quality_result)
show_outliers(df, outlier_result)
show_distribution(distribution_result, df)
show_imbalance(imbalance_result)
show_drift(drift_result)
show_leakage(leakage_result)
show_remediation(suggestions)

# Downloads
results = {
    "schema": schema,
    "quality": quality_result,
    "outliers": outlier_result,
    "distribution": distribution_result,
    "imbalance": imbalance_result,
    "drift": drift_result,
    "leakage": leakage_result,
    "readiness": readiness_result,
    "suggestions": suggestions,
}
html_args = (
    schema, quality_result, outlier_result, distribution_result,
    imbalance_result, drift_result, leakage_result, readiness_result, suggestions,
)
json_data, html_data = build_downloads(results, html_args)

st.divider()
col1, col2 = st.columns(2)
col1.download_button(
    "Download JSON", json_data, file_name="datasentry_report.json", mime="application/json"
)
col2.download_button(
    "Download HTML", html_data, file_name="datasentry_report.html", mime="text/html"
)