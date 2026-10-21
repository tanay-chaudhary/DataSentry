import pandas as pd
import streamlit as st

from datasentry.ui.plots import (
    missing_percent,
    plot_class_distribution,
    plot_distribution,
    plot_missing_values,
    plot_outlier_boxplot,
)


def _round(value, digits: int = 3):
    return round(value, digits) if isinstance(value, float) else value


def show_schema(schema: dict) -> None:
    st.subheader("Schema")
    columns = schema.get("columns", schema)
    types = [
        str(info.get("type", info.get("dtype")) if isinstance(info, dict) else info)
        for info in columns.values()
    ]
    st.table({"Column": list(columns.keys()), "Type": types})


def show_quality(quality_result: dict) -> None:
    st.subheader("Data Quality")
    rows = []
    for col, info in quality_result["columns"].items():
        pct = missing_percent(info)
        rows.append({
            "Column": col,
            "Missing %": round(pct, 1) if pct is not None else None,
            "Flagged": "Yes" if info["flagged"] else "No",
        })
    st.dataframe(pd.DataFrame(rows))
    st.metric("Duplicate rows", quality_result.get("duplicate_rows", 0))
    st.pyplot(plot_missing_values(quality_result))


def show_outliers(df: pd.DataFrame, outlier_result: dict) -> None:
    st.subheader("Outliers")
    rows = [
        {
            "Column": col,
            "Outliers": info.get("count", info.get("outliers", 0)),
            "Lower bound": _round(info.get("lower")),
            "Upper bound": _round(info.get("upper")),
        }
        for col, info in outlier_result.items()
    ]
    st.dataframe(pd.DataFrame(rows))
    st.pyplot(plot_outlier_boxplot(df, outlier_result))


def show_distribution(distribution_result: dict, df: pd.DataFrame | None = None) -> None:
    st.subheader("Distribution")
    columns = distribution_result.get("columns", distribution_result)
    table = pd.DataFrame.from_dict(columns, orient="index")
    st.dataframe(table.apply(lambda s: s.map(_round)))
    if df is not None:
        st.pyplot(plot_distribution(distribution_result, df))


def show_imbalance(imbalance_result: dict) -> None:
    st.subheader("Class Imbalance")
    if not imbalance_result["class_distribution"]:
        st.info("Enter a target column to check class imbalance.")
        return
    rows = [
        {"Class": label, "Proportion %": round(p * 100, 1)}
        for label, p in imbalance_result["class_distribution"].items()
    ]
    st.dataframe(pd.DataFrame(rows))
    if imbalance_result["imbalanced"]:
        st.warning("Target is imbalanced.")
    else:
        st.success("Target classes are reasonably balanced.")
    st.pyplot(plot_class_distribution(imbalance_result))


def show_drift(drift_result: dict) -> None:
    st.subheader("Drift Detection")
    if not drift_result:
        st.info("Upload a reference CSV to check for drift.")
        return
    rows = [
        {
            "Column": col,
            "p-value": _round(info["p_value"]),
            "Drifted": "Yes" if info["drifted"] else "No",
        }
        for col, info in drift_result.items()
    ]
    st.dataframe(pd.DataFrame(rows))


def show_leakage(leakage_result: dict) -> None:
    st.subheader("Leakage Detection")
    if not leakage_result["columns"]:
        st.info("Enter a target column to check for leakage.")
        return
    rows = [
        {
            "Column": col,
            "Correlation": round(info["correlation"], 3),
            "Leakage": "Yes" if info["leakage"] else "No",
        }
        for col, info in leakage_result["columns"].items()
    ]
    st.dataframe(pd.DataFrame(rows))


def show_readiness(readiness_result: dict) -> None:
    st.subheader("ML Readiness")
    score = readiness_result["score"]
    st.metric("Score", f"{score}/100")
    st.progress(max(0, min(score, 100)) / 100)
    if not readiness_result["penalties"]:
        st.success("No penalties.")
    for p in readiness_result["penalties"]:
        st.warning(p)


def show_remediation(suggestions: list) -> None:
    st.subheader("Recommendations")
    if not suggestions:
        st.success("No issues found. Nothing to fix.")
    for s in suggestions:
        st.info(s)