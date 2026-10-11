import pandas as pd
import pytest
from datasentry.core.imbalance import detect_imbalance
from datasentry.config import SentryConfig


def make_config():
    return SentryConfig()  


def test_imbalanced_target_is_flagged():
    df = pd.DataFrame({"label": ["not_spam"] * 95 + ["spam"] * 5})
    result = detect_imbalance(df, "label", make_config())

    assert result["imbalanced"] is True
    assert result["minority_classes"] == ["spam"]
    assert result["class_distribution"]["spam"] == pytest.approx(0.05)


def test_balanced_target_is_not_flagged():
    df = pd.DataFrame({"label": ["a"] * 50 + ["b"] * 50})
    result = detect_imbalance(df, "label", make_config())

    assert result["imbalanced"] is False
    assert result["minority_classes"] == []


def test_missing_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        detect_imbalance(df, "label", make_config())