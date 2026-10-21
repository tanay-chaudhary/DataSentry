import math

import pandas as pd
from matplotlib.figure import Figure

MAX_PANELS = 12


def missing_percent(info: dict):
    """Missing percentage (0-100) from a quality entry, or None if no known key exists."""
    for key in ("null_pct", "missing_pct", "null_ratio", "missing_ratio"):
        if key in info:
            value = info[key]
            return value * 100 if value <= 1 else value
    return None


def _empty_figure(title: str, message: str) -> Figure:
    fig = Figure(figsize=(6, 3))
    ax = fig.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_missing_values(quality_result: dict) -> Figure:
    columns = quality_result["columns"]
    if not columns:
        return _empty_figure("Missing values per column", "No columns to show")

    names = [str(c) for c in columns]
    pcts = [missing_percent(info) or 0 for info in columns.values()]
    colors = ["#c0392b" if info["flagged"] else "#2e86c1" for info in columns.values()]

    fig = Figure(figsize=(max(6, len(names) * 0.6), 4))
    ax = fig.subplots()
    ax.bar(names, pcts, color=colors)
    ax.set_title("Missing values per column (red = flagged)")
    ax.set_ylabel("Missing %")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    return fig


def plot_outlier_boxplot(df: pd.DataFrame, outlier_result: dict) -> Figure:
    cols = [
        col
        for col, info in outlier_result.items()
        if info.get("count", info.get("outliers", 0)) > 0
        and col in df.columns
        and pd.api.types.is_numeric_dtype(df[col])
    ][:MAX_PANELS]

    if not cols:
        return _empty_figure("Outliers", "No columns with outliers")

    # One small panel per column, because columns usually have very different scales
    fig = Figure(figsize=(3 * len(cols), 4))
    axes = fig.subplots(1, len(cols), squeeze=False)[0]
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna())
        ax.set_title(str(col))
        ax.set_xticks([])
    fig.suptitle("Columns with outliers")
    fig.tight_layout()
    return fig


def plot_distribution(distribution_result: dict, df: pd.DataFrame | None = None) -> Figure:
    if df is None:
        return _empty_figure("Distribution", "Pass the DataFrame to draw histograms")

    summarized = distribution_result.get("columns", distribution_result)
    cols = [
        c for c in summarized
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        cols = list(df.select_dtypes(include="number").columns)
    cols = cols[:MAX_PANELS]

    if not cols:
        return _empty_figure("Distribution", "No numeric columns to show")

    ncols = min(3, len(cols))
    nrows = math.ceil(len(cols) / ncols)
    fig = Figure(figsize=(4 * ncols, 3 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False).flatten()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20, color="#2e86c1")
        ax.set_title(str(col))
    for ax in axes[len(cols):]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_class_distribution(imbalance_result: dict) -> Figure:
    distribution = imbalance_result["class_distribution"]
    if not distribution:
        return _empty_figure("Class distribution", "No target column provided")

    minority = set(imbalance_result["minority_classes"])
    labels = [str(label) for label in distribution]
    values = [p * 100 for p in distribution.values()]
    colors = ["#c0392b" if label in minority else "#2e86c1" for label in distribution]

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.bar(labels, values, color=colors)
    ax.set_title("Class distribution (red = minority)")
    ax.set_ylabel("Proportion %")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig