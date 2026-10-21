from streamlit.testing.v1 import AppTest


def test_components_render_without_errors():
    def script():
        import numpy as np
        import pandas as pd

        from datasentry.ui.components import (
            show_drift, show_imbalance, show_leakage, show_outliers,
            show_quality, show_readiness, show_remediation, show_schema,
        )

        df = pd.DataFrame({"age": np.arange(50.0)})
        show_schema({"age": "numeric"})
        show_quality({"columns": {"age": {"flagged": True, "null_pct": 0.5}}, "duplicate_rows": 2})
        show_outliers(df, {"age": {"count": 2, "lower": 1.0, "upper": 9.0}})
        show_imbalance({"class_distribution": {"a": 0.9, "b": 0.1},
                        "imbalanced": True, "minority_classes": ["b"]})
        show_drift({"age": {"p_value": 0.01, "drifted": True}})
        show_leakage({"columns": {"age": {"correlation": 0.99, "leakage": True}}})
        show_readiness({"score": 72, "penalties": ["Imbalance detected (-10)"]})
        show_remediation(["Target is imbalanced"])

    at = AppTest.from_function(script, default_timeout=60).run()

    assert not at.exception