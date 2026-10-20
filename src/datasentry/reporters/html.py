from pathlib import Path

from jinja2 import Environment

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DataSentry Report</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { color: #0b6e4f; }
  h2 { margin-top: 2rem; border-bottom: 2px solid #ddd; padding-bottom: .3rem; }
  table { border-collapse: collapse; width: 100%; margin: .5rem 0; }
  th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }
  th { background: #f3f3f3; }
  td.num { text-align: right; }
  .bad { color: #c0392b; font-weight: bold; }
  .ok { color: #1e8449; }
  .score { font-size: 2rem; font-weight: bold; padding: 1rem; border-radius: 8px; display: inline-block; }
  .score.green { background: #e8f6ee; color: #1e8449; }
  .score.yellow { background: #fef9e7; color: #b7950b; }
  .score.red { background: #fdedec; color: #c0392b; }
</style>
</head>
<body>
<h1>DataSentry Report</h1>

<h2>ML Readiness</h2>
<div class="score {{ score_color }}">{{ readiness.score }}/100</div>
{% if readiness.penalties %}
<ul>
  {% for item in readiness.penalties %}<li>{{ item }}</li>{% endfor %}
</ul>
{% else %}
<p class="ok">No penalties.</p>
{% endif %}

<h2>Schema</h2>
<table>
  <tr><th>Column</th><th>Type</th></tr>
  {% for row in schema_rows %}
  <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td></tr>
  {% endfor %}
</table>

<h2>Quality</h2>
<table>
  <tr><th>Column</th><th>Missing %</th><th>Flagged</th></tr>
  {% for row in quality_rows %}
  <tr>
    <td>{{ row.col }}</td>
    <td class="num">{{ row.pct }}</td>
    <td class="{{ 'bad' if row.flagged else 'ok' }}">{{ 'Yes' if row.flagged else 'No' }}</td>
  </tr>
  {% endfor %}
</table>
<p>Duplicate rows: <span class="{{ 'bad' if duplicates > 0 else 'ok' }}">{{ duplicates }}</span></p>

<h2>Outliers</h2>
<table>
  <tr><th>Column</th><th>Outliers</th><th>Lower bound</th><th>Upper bound</th></tr>
  {% for row in outlier_rows %}
  <tr>
    <td>{{ row.col }}</td>
    <td class="num {{ 'bad' if row.count > 0 else 'ok' }}">{{ row.count }}</td>
    <td class="num">{{ row.lower }}</td>
    <td class="num">{{ row.upper }}</td>
  </tr>
  {% endfor %}
</table>

<h2>Distribution</h2>
<table>
  <tr><th>Column</th>{% for stat in dist_stats %}<th>{{ stat }}</th>{% endfor %}</tr>
  {% for col, values in dist_rows %}
  <tr><td>{{ col }}</td>{% for v in values %}<td class="num">{{ v }}</td>{% endfor %}</tr>
  {% endfor %}
</table>

<h2>Class Imbalance</h2>
<table>
  <tr><th>Class</th><th>Proportion</th></tr>
  {% for row in imbalance_rows %}
  <tr>
    <td class="{{ 'bad' if row.minority else 'ok' }}">{{ row.label }}</td>
    <td class="num">{{ row.pct }}</td>
  </tr>
  {% endfor %}
</table>
<p>Imbalanced:
  <span class="{{ 'bad' if imbalance.imbalanced else 'ok' }}">{{ 'Yes' if imbalance.imbalanced else 'No' }}</span>
</p>

<h2>Drift</h2>
<table>
  <tr><th>Column</th><th>p-value</th><th>Drifted</th></tr>
  {% for row in drift_rows %}
  <tr>
    <td>{{ row.col }}</td>
    <td class="num">{{ row.p_value }}</td>
    <td class="{{ 'bad' if row.drifted else 'ok' }}">{{ 'Yes' if row.drifted else 'No' }}</td>
  </tr>
  {% endfor %}
</table>

<h2>Leakage</h2>
<table>
  <tr><th>Column</th><th>Correlation</th><th>Leakage</th></tr>
  {% for row in leakage_rows %}
  <tr>
    <td>{{ row.col }}</td>
    <td class="num">{{ row.corr }}</td>
    <td class="{{ 'bad' if row.leakage else 'ok' }}">{{ 'Yes' if row.leakage else 'No' }}</td>
  </tr>
  {% endfor %}
</table>

<h2>Remediation</h2>
{% if suggestions %}
<ul>
  {% for s in suggestions %}<li>{{ s }}</li>{% endfor %}
</ul>
{% else %}
<p class="ok">No issues found. Nothing to fix.</p>
{% endif %}
</body>
</html>
"""

# autoescape=True turns characters like < and & in column names into safe HTML
_env = Environment(autoescape=True)
_template = _env.from_string(TEMPLATE)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _missing_pct(info: dict) -> str:
    for key in ("null_pct", "missing_pct", "null_ratio", "missing_ratio"):
        if key in info:
            value = info[key]
            pct = value * 100 if value <= 1 else value
            return f"{pct:.1f}%"
    return "-"


def export_html(
    schema: dict,
    quality_result: dict,
    outlier_result: dict,
    distribution_result: dict,
    imbalance_result: dict,
    drift_result: dict,
    leakage_result: dict,
    readiness_result: dict,
    suggestions: list,
    output_path: str,
) -> None:
    # Schema: accept {col: "type"} or {col: {"type": ...}}, optionally under "columns"
    schema_cols = schema.get("columns", schema)
    schema_rows = []
    for col, info in schema_cols.items():
        dtype = info.get("type", info.get("dtype")) if isinstance(info, dict) else info
        schema_rows.append((col, _fmt(dtype)))

    quality_rows = [
        {"col": col, "pct": _missing_pct(info), "flagged": info["flagged"]}
        for col, info in quality_result["columns"].items()
    ]

    outlier_rows = [
        {
            "col": col,
            "count": info.get("count", info.get("outliers", 0)),
            "lower": _fmt(info.get("lower")),
            "upper": _fmt(info.get("upper")),
        }
        for col, info in outlier_result.items()
    ]

    # Distribution: one table column per statistic, in order of first appearance
    dist_cols = distribution_result.get("columns", distribution_result)
    dist_stats = []
    for info in dist_cols.values():
        if isinstance(info, dict):
            for key in info:
                if key not in dist_stats:
                    dist_stats.append(key)
    dist_rows = []
    for col, info in dist_cols.items():
        info = info if isinstance(info, dict) else {}
        dist_rows.append((col, [_fmt(info.get(s)) for s in dist_stats]))

    minority = set(imbalance_result["minority_classes"])
    imbalance_rows = [
        {"label": label, "pct": f"{p:.1%}", "minority": label in minority}
        for label, p in imbalance_result["class_distribution"].items()
    ]

    drift_rows = [
        {"col": col, "p_value": _fmt(info["p_value"]), "drifted": info["drifted"]}
        for col, info in drift_result.items()
    ]

    leakage_rows = [
        {"col": col, "corr": f"{info['correlation']:.2f}", "leakage": info["leakage"]}
        for col, info in leakage_result["columns"].items()
    ]

    score = readiness_result["score"]
    score_color = "green" if score >= 80 else "yellow" if score >= 50 else "red"

    html_output = _template.render(
        readiness=readiness_result,
        score_color=score_color,
        schema_rows=schema_rows,
        quality_rows=quality_rows,
        duplicates=quality_result.get("duplicate_rows", 0),
        outlier_rows=outlier_rows,
        dist_stats=dist_stats,
        dist_rows=dist_rows,
        imbalance=imbalance_result,
        imbalance_rows=imbalance_rows,
        drift_rows=drift_rows,
        leakage_rows=leakage_rows,
        suggestions=suggestions,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_output)