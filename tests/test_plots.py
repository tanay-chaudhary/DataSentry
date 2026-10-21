import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from datasentry.ui.plots import (
    plot_class_distribution,
    plot_distribution,
    plot_missing_values,
    plot_outlier_boxplot,
)


def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({"age": rng.normal(40, 10, 100), "income": rng.normal(5e4, 8e3, 100)})


def test_missing_values_plot():
    quality = {
        "columns": {
            "age": {"flagged": True, "null_pct": 0.5},
            "plan": {"flagged": False, "null_pct": 0.0},
        },
        "duplicate_rows": 0,
    }
    fig = plot_missing_values(quality)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) == 2  # one bar per column


def test_boxplot_only_for_columns_with_outliers():
    outliers = {"age": {"count": 3}, "income": {"count": 0}}
    fig = plot_outlier_boxplot(sample_df(), outliers)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_boxplot_with_no_outliers_returns_message_figure():
    fig = plot_outlier_boxplot(sample_df(), {"age": {"count": 0}})

    assert isinstance(fig, Figure)


def test_distribution_histograms():
    fig = plot_distribution({"age": {}, "income": {}}, sample_df())

    assert isinstance(fig, Figure)
    assert len([ax for ax in fig.axes if ax.axison]) == 2


def test_class_distribution_plot():
    imbalance = {
        "class_distribution": {"free": 0.95, "paid": 0.05},
        "imbalanced": True,
        "minority_classes": ["paid"],
    }
    fig = plot_class_distribution(imbalance)

    assert len(fig.axes[0].patches) == 2


def test_class_distribution_empty():
    empty = {"class_distribution": {}, "imbalanced": False, "minority_classes": []}

    assert isinstance(plot_class_distribution(empty), Figure)