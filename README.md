# DataSentry

Automated dataset quality, statistical drift, and ML-readiness auditing for tabular data.

Point DataSentry at a CSV file and it runs a series of checks, gives the dataset a **0-100 ML-readiness score**, and suggests a fix for every problem it finds. Results can be viewed in the terminal, exported as JSON or HTML, or explored in a Streamlit web app.

<!-- Add a screenshot of the HTML report or the Streamlit app here:
![DataSentry report](docs/report.png) -->

## Features

| Check | What it does |
|-------|--------------|
| **Ingestion** | Reads CSV files, treats common placeholders (`NA`, `null`, `?`, ...) as missing values, and falls back to another encoding if UTF-8 fails |
| **Schema inference** | Labels each column as identifier, boolean, datetime, numeric, categorical, or text |
| **Data quality** | Missing-value percentage per column (flags columns above a threshold) and duplicate-row count |
| **Outliers** | IQR-based outlier detection on numeric columns, with lower and upper bounds |
| **Distribution** | Summary statistics for numeric columns and value counts for categorical ones |
| **Class imbalance** | Class proportions for a target column; flags classes below a minimum share |
| **Drift detection** | Compares a reference CSV with the current CSV: Kolmogorov-Smirnov test for numeric columns, chi-squared test for categorical ones |
| **Leakage detection** | Flags features that are suspiciously strongly related to the target (Pearson correlation for numeric features, Cramér's V for categorical ones, plus a missingness check) |
| **Readiness score** | Starts at 100 and deducts points for each problem found |
| **Remediation** | A plain-language suggestion for every issue |

Reports: colored terminal output (via `rich`), JSON, a self-contained HTML file, and a Streamlit web app with charts.

## Installation

Requires Python 3.12 or newer.

```bash
git clone https://github.com/tanay-fyre/DataSentry.git
cd DataSentry
pip install -e ".[dev]"
```

The `-e` flag installs the project in editable mode, and `[dev]` adds the test tools (`pytest`, `pytest-cov`). Installing also registers the `datasentry` command.

## Usage

### Command line

```bash
# Basic audit
datasentry data.csv

# Add imbalance and leakage checks by naming the target column
datasentry data.csv --target Survived

# Add drift detection by providing the older (reference) dataset
datasentry data.csv --reference old_data.csv

# Everything, with exported reports
datasentry data.csv --target Survived --reference old_data.csv --json report.json --html report.html
```

| Argument | Description |
|----------|-------------|
| `file` | Path to the CSV file to audit |
| `--target` | Target column name. Enables the imbalance and leakage checks |
| `--reference` | Reference CSV. Enables drift detection (the main file is treated as the newer data) |
| `--json` | Save all results to a JSON file |
| `--html` | Save a self-contained HTML report |

If a file can't be read or the target column doesn't exist, DataSentry prints a short error message and exits with code 1.

If the target column has more unique values than `cardinality_cap` (50 by default), it doesn't look like a classification target, so the imbalance check is skipped with a warning.

### Web app

```bash
streamlit run src/datasentry/ui/app.py
```

Upload a CSV, optionally enter a target column and upload a reference CSV, and the app shows every section with tables and charts. The JSON and HTML reports can be downloaded from the bottom of the page.

### As a library

```python
from datasentry.config import SentryConfig
from datasentry.core.ingestion import ingest_csv
from datasentry.core.schema import infer_schema
from datasentry.core.quality import quality_check
from datasentry.core.outliers import detect_outliers
from datasentry.core.imbalance import detect_imbalance
from datasentry.core.leakage import detect_leakage
from datasentry.scoring.readiness import compute_readiness
from datasentry.scoring.remediation import suggest_remediation

config = SentryConfig()
df = ingest_csv("data.csv", config)

schema = infer_schema(df, config)
quality = quality_check(df, config)
outliers = detect_outliers(df, config, schema)
imbalance = detect_imbalance(df, "Survived", config)
leakage = detect_leakage(df, "Survived", config)

readiness = compute_readiness(quality, outliers, {}, imbalance, leakage)
print(readiness["score"], readiness["penalties"])
print(suggest_remediation(quality, outliers, {}, imbalance, leakage))
```

Every check is a plain function that takes a DataFrame (and a config) and returns a plain dictionary, so the results are easy to inspect, test, and serialize.

## How the readiness score works

The score starts at 100 and never goes below 0.

| Problem | Penalty |
|---------|---------|
| Each column flagged for high missing values | -5 |
| Each column with outliers | -3 |
| Each column with drift | -5 |
| Class imbalance detected | -10 |
| Each column flagged for leakage | -15 |
| Duplicate rows found | -5 |

Leakage has the largest penalty because it makes a model look excellent during training and fail on real data.

## Configuration

All thresholds live in `SentryConfig` (`src/datasentry/config.py`):

| Field | Default | Meaning |
|-------|---------|---------|
| `null_threshold` | `0.4` | Flag a column if more than 40% of its values are missing |
| `cardinality_cap` | `50` | Columns with fewer unique values than this can be treated as categorical |
| `iqr_multiplier` | `1.5` | How far beyond the quartiles a value must be to count as an outlier |
| `drift_alpha` | `0.05` | Significance level for the drift tests |
| `imbalance_floor` | `0.1` | A class below this share of the data counts as a minority class |
| `leakage_corr_ceil` | `0.95` | Association with the target above this is flagged as leakage |
| `sentinels` | `["", "NA", "N/A", "null", "none", "-", "?"]` | Values treated as missing when reading a CSV |

## Project structure

```
DataSentry/
├── pyproject.toml
├── README.md
├── src/datasentry/
│   ├── cli.py                 # Command-line entry point
│   ├── config.py              # SentryConfig thresholds
│   ├── core/
│   │   ├── ingestion.py       # CSV loading
│   │   ├── schema.py          # Column type inference
│   │   ├── quality.py         # Missing values and duplicates
│   │   ├── outliers.py        # IQR outlier detection
│   │   ├── distribution.py    # Summary statistics
│   │   ├── imbalance.py       # Class imbalance
│   │   ├── drift.py           # KS and chi-squared drift tests
│   │   └── leakage.py         # Target leakage
│   ├── scoring/
│   │   ├── readiness.py       # 0-100 score
│   │   └── remediation.py     # Suggested fixes
│   ├── reporters/
│   │   ├── console.py         # Terminal report (rich)
│   │   ├── json_exporter.py   # JSON export
│   │   ├── html.py            # HTML report (Jinja2)
│   │   └── formatting.py      # Shared display helpers
│   └── ui/
│       ├── app.py             # Streamlit app
│       ├── components.py      # Streamlit display sections
│       └── plots.py           # Matplotlib charts
└── tests/
```

## Development

```bash
pytest                              # run all tests
pytest -v                           # one line per test
pytest tests/test_drift.py          # one file
pytest --cov=datasentry             # with coverage
```

## Limitations

DataSentry is a quick audit tool, not a replacement for looking at your data.

- **IQR outliers** work best on roughly symmetric numeric data. Heavily skewed columns can flag many normal values, and columns where the IQR is zero are skipped.
- **Drift tests** report p-values only. On very large datasets, tiny and unimportant shifts can still look "significant". The chi-squared test is unreliable for columns with very many categories.
- **Leakage detection** uses correlation and association statistics. It catches columns that are near copies of the target, but not complicated relationships, and a high score is a warning sign, not proof.
- **Schema inference** is heuristic. Integer columns with fewer than `cardinality_cap` distinct values are treated as categorical, and date detection only recognizes a few common formats.
- **Everything runs in memory** with pandas, so very large files may be slow or exceed available memory.

## Tech stack

Python, pandas, NumPy, SciPy, rich, Jinja2, Streamlit, Matplotlib, pytest.

## License

Add a license before publishing (for example [MIT](https://choosealicense.com/licenses/mit/)), save its text in a `LICENSE` file, and reference it in `pyproject.toml`.
