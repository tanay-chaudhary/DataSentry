from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table


def _fmt(value, digits: int = 3) -> str:
    """Turn any value into a short, safe string for a table cell."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return escape(str(value))


def _yes_no(flag: bool, bad_when_true: bool = True) -> str:
    """Colored Yes/No: red for a problem, green for OK."""
    is_problem = flag if bad_when_true else not flag
    color = "red" if is_problem else "green"
    return f"[{color}]{'Yes' if flag else 'No'}[/{color}]"


def _section(console: Console, title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def _print_schema(console: Console, schema: dict) -> None:
    _section(console, "Schema")
    columns = schema.get("columns", schema)
    table = Table()
    table.add_column("Column")
    table.add_column("Type")
    for col, info in columns.items():
        dtype = info.get("type", info.get("dtype")) if isinstance(info, dict) else info
        table.add_row(escape(str(col)), _fmt(dtype))
    console.print(table)


def _print_quality(console: Console, quality_result: dict) -> None:
    _section(console, "Quality")
    table = Table()
    table.add_column("Column")
    table.add_column("Missing %", justify="right")
    table.add_column("Flagged")
    for col, info in quality_result["columns"].items():
        pct = None
        for key in ("null_pct", "missing_pct", "null_ratio", "missing_ratio"):
            if key in info:
                value = info[key]
                pct = value * 100 if value <= 1 else value
                break
        table.add_row(
            escape(str(col)),
            f"{pct:.1f}%" if pct is not None else "-",
            _yes_no(info["flagged"]),
        )
    console.print(table)

    duplicates = quality_result.get("duplicate_rows", 0)
    color = "red" if duplicates > 0 else "green"
    console.print(f"Duplicate rows: [{color}]{duplicates}[/{color}]")


def _print_outliers(console: Console, outlier_result: dict) -> None:
    _section(console, "Outliers")
    table = Table()
    table.add_column("Column")
    table.add_column("Outliers", justify="right")
    table.add_column("Lower bound", justify="right")
    table.add_column("Upper bound", justify="right")
    for col, info in outlier_result.items():
        count = info.get("count", info.get("outliers", 0))
        color = "red" if count > 0 else "green"
        table.add_row(
            escape(str(col)),
            f"[{color}]{count}[/{color}]",
            _fmt(info.get("lower")),
            _fmt(info.get("upper")),
        )
    console.print(table)


def _print_distribution(console: Console, distribution_result: dict) -> None:
    _section(console, "Distribution")
    columns = distribution_result.get("columns", distribution_result)

    # One column per statistic, in the order they first appear
    stats = []
    for info in columns.values():
        if isinstance(info, dict):
            for key in info:
                if key not in stats:
                    stats.append(key)

    table = Table()
    table.add_column("Column")
    for stat in stats:
        table.add_column(escape(str(stat)), justify="right")
    for col, info in columns.items():
        info = info if isinstance(info, dict) else {}
        table.add_row(escape(str(col)), *[_fmt(info.get(s)) for s in stats])
    console.print(table)


def _print_imbalance(console: Console, imbalance_result: dict) -> None:
    _section(console, "Class Imbalance")
    table = Table()
    table.add_column("Class")
    table.add_column("Proportion", justify="right")
    minority = set(imbalance_result["minority_classes"])
    for label, proportion in imbalance_result["class_distribution"].items():
        style = "red" if label in minority else "green"
        table.add_row(f"[{style}]{escape(str(label))}[/{style}]", f"{proportion:.1%}")
    console.print(table)

    status = _yes_no(imbalance_result["imbalanced"])
    console.print(f"Imbalanced: {status}")


def _print_drift(console: Console, drift_result: dict) -> None:
    _section(console, "Drift")
    table = Table()
    table.add_column("Column")
    table.add_column("p-value", justify="right")
    table.add_column("Drifted")
    for col, info in drift_result.items():
        table.add_row(
            escape(str(col)),
            _fmt(info["p_value"]),
            _yes_no(info["drifted"]),
        )
    console.print(table)


def _print_leakage(console: Console, leakage_result: dict) -> None:
    _section(console, "Leakage")
    table = Table()
    table.add_column("Column")
    table.add_column("Correlation", justify="right")
    table.add_column("Leakage")
    for col, info in leakage_result["columns"].items():
        table.add_row(
            escape(str(col)),
            f"{info['correlation']:.2f}",
            _yes_no(info["leakage"]),
        )
    console.print(table)


def _print_readiness(console: Console, readiness_result: dict) -> None:
    score = readiness_result["score"]
    if score >= 80:
        color = "green"
    elif score >= 50:
        color = "yellow"
    else:
        color = "red"

    lines = [f"[bold {color}]Score: {score}/100[/bold {color}]"]
    for penalty in readiness_result["penalties"]:
        lines.append(f"  • {escape(penalty)}")

    console.print()
    console.print(Panel("\n".join(lines), title="ML Readiness", border_style=color))


def _print_suggestions(console: Console, suggestions: list) -> None:
    _section(console, "Remediation")
    if not suggestions:
        console.print("[green]No issues found. Nothing to fix.[/green]")
        return
    for suggestion in suggestions:
        console.print(f"  • {escape(suggestion)}")


def print_report(
    schema: dict,
    quality_result: dict,
    outlier_result: dict,
    distribution_result: dict,
    imbalance_result: dict,
    drift_result: dict,
    leakage_result: dict,
    readiness_result: dict,
    suggestions: list,
    console: Console | None = None,
) -> None:
    console = console or Console()

    console.print("[bold green]DataSentry Report[/bold green]")

    _print_schema(console, schema)
    _print_quality(console, quality_result)
    _print_outliers(console, outlier_result)
    _print_distribution(console, distribution_result)
    _print_imbalance(console, imbalance_result)
    _print_drift(console, drift_result)
    _print_leakage(console, leakage_result)
    _print_readiness(console, readiness_result)
    _print_suggestions(console, suggestions)