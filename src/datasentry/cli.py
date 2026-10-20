import argparse
import sys

from datasentry.config import SentryConfig
from datasentry.core.distribution import summarize_distribution
from datasentry.core.drift import detect_drift
from datasentry.core.imbalance import detect_imbalance
from datasentry.core.ingestion import ingest_csv
from datasentry.core.leakage import detect_leakage
from datasentry.core.outliers import detect_outliers
from datasentry.core.quality import quality_check
from datasentry.reporters.console import print_report
from datasentry.reporters.html import export_html
from datasentry.reporters.json_exporter import export_json
from datasentry.core.schema import infer_schema
from datasentry.scoring.readiness import compute_readiness
from datasentry.scoring.remediation import suggest_remediation

# Placeholders passed along when an optional check is skipped
EMPTY_IMBALANCE = {"class_distribution": {}, "imbalanced": False, "minority_classes": []}
EMPTY_LEAKAGE = {"columns": {}}
EMPTY_DRIFT = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasentry",
        description="DataSentry - Dataset Auditing Tool",
    )
    parser.add_argument("file", help="Path to the CSV file")
    parser.add_argument("--target", help="Target column name (for imbalance/leakage checks)")
    parser.add_argument("--reference", help="Reference CSV for drift detection")
    parser.add_argument("--json", help="Export results to JSON file")
    parser.add_argument("--html", help="Export results to HTML file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = SentryConfig()

    # 1. Load the data
    try:
        df = ingest_csv(args.file, config)
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    df_reference = None
    if args.reference:
        try:
            df_reference = ingest_csv(args.reference, config)
        except FileNotFoundError:
            print(f"Error: reference file not found: {args.reference}", file=sys.stderr)
            return 1

    if args.target and args.target not in df.columns:
        print(f"Error: target column '{args.target}' not found in {args.file}", file=sys.stderr)
        return 1

    # 2. Checks that always run
    schema = infer_schema(df, config)
    quality_result = quality_check(df, config)
    outlier_result = detect_outliers(df, config)
    distribution_result = summarize_distribution(df, config)

    # 3. Optional checks
    imbalance_result = EMPTY_IMBALANCE
    leakage_result = EMPTY_LEAKAGE
    if args.target:
        if df[args.target].nunique() > config.cardinality_cap:
            print(
                f"Warning: '{args.target}' has more than {config.cardinality_cap} unique values, "
                "so it doesn't look categorical. Skipping the imbalance check.",
                file=sys.stderr,
            )
        else:
            imbalance_result = detect_imbalance(df, args.target, config)
        leakage_result = detect_leakage(df, args.target, config)

    drift_result = EMPTY_DRIFT
    if df_reference is not None:
        drift_result = detect_drift(df_reference, df, config)

    # 4. Verdict and fixes
    readiness_result = compute_readiness(
        quality_result, outlier_result, drift_result, imbalance_result, leakage_result
    )
    suggestions = suggest_remediation(
        quality_result, outlier_result, drift_result, imbalance_result, leakage_result
    )

    # 5. Output
    print_report(
        schema, quality_result, outlier_result, distribution_result,
        imbalance_result, drift_result, leakage_result, readiness_result, suggestions,
    )

    if args.json:
        export_json(
            {
                "schema": schema,
                "quality": quality_result,
                "outliers": outlier_result,
                "distribution": distribution_result,
                "imbalance": imbalance_result,
                "drift": drift_result,
                "leakage": leakage_result,
                "readiness": readiness_result,
                "suggestions": suggestions,
            },
            args.json,
        )
        print(f"JSON report saved to {args.json}")

    if args.html:
        export_html(
            schema, quality_result, outlier_result, distribution_result,
            imbalance_result, drift_result, leakage_result, readiness_result,
            suggestions, args.html,
        )
        print(f"HTML report saved to {args.html}")

    return 0


if __name__ == "__main__":
    sys.exit(main())