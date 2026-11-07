
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "automated_reporting"

# These folders contain outputs from the completed quantitative stages.
SOURCE_FOLDERS = [
    "walk_forward_comparison",
    "performance_analytics",
    "stress_testing",
    "robustness_analysis",
    "black_litterman_optimization",
]

EXCLUDED_DIRS = {
    "automated_reporting",
    "__pycache__",
}

MAX_EXCEL_ROWS = 1_048_575  # One row reserved for the header.
MAX_SHEET_NAME_LENGTH = 31


def log(message: str) -> None:
    print(f"[REPORTING] {message}")


def safe_sheet_name(name: str, used: set[str]) -> str:
    """Return a unique Excel-compatible worksheet name."""
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", name).strip("'")
    cleaned = cleaned or "Report"
    cleaned = cleaned[:MAX_SHEET_NAME_LENGTH]

    candidate = cleaned
    counter = 1

    while candidate.lower() in used:
        suffix = f"_{counter}"
        candidate = cleaned[: MAX_SHEET_NAME_LENGTH - len(suffix)] + suffix
        counter += 1

    used.add(candidate.lower())
    return candidate


def discover_csv_files() -> list[Path]:
    """Discover CSV reports in the configured report folders."""
    discovered: list[Path] = []

    for folder_name in SOURCE_FOLDERS:
        folder = REPORTS_DIR / folder_name

        if not folder.exists():
            log(f"Source folder not found: {folder_name}")
            continue

        for path in sorted(folder.rglob("*.csv")):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            discovered.append(path)

    # Also include CSV reports directly inside reports/.
    for path in sorted(REPORTS_DIR.glob("*.csv")):
        if path.parent == REPORTS_DIR:
            discovered.append(path)

    # Avoid processing the same file twice.
    return sorted(set(discovered))


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while preserving a useful error message."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def json_safe(value: Any) -> Any:
    """Convert NumPy and pandas objects to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None

    return value


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def classify_report(path: Path) -> str:
    name = path.name.lower()
    folder = path.parent.name.lower()

    if "performance" in name or "benchmark" in name:
        return "Performance Analytics"
    if "drawdown" in name:
        return "Drawdown Analysis"
    if "calendar" in name:
        return "Calendar Returns"
    if "rolling" in name:
        return "Rolling Metrics"
    if "stress" in name or "scenario" in name or "historical" in name:
        return "Stress Testing"
    if "regime" in name:
        return "Market Regime Analysis"
    if "robustness" in name or "sensitivity" in name:
        return "Robustness Analysis"
    if "failed" in name:
        return "Optimization Diagnostics"
    if "strategy" in name or "walk_forward" in name:
        return "Strategy Comparison"
    if "black_litterman" in folder or "black_litterman" in name:
        return "Black-Litterman Analysis"

    return folder.replace("_", " ").title()


def build_catalog(
    paths: list[Path],
    datasets: dict[str, pd.DataFrame],
    errors: list[dict[str, str]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for path in paths:
        key = relative_path(path)
        frame = datasets.get(key)

        records.append(
            {
                "report_file": path.name,
                "relative_path": key,
                "report_category": classify_report(path),
                "rows": len(frame) if frame is not None else None,
                "columns": len(frame.columns) if frame is not None else None,
                "status": "loaded" if frame is not None else "not_loaded",
                "file_size_kb": round(path.stat().st_size / 1024, 2),
            }
        )

    for error in errors:
        records.append(
            {
                "report_file": Path(error["path"]).name,
                "relative_path": error["path"],
                "report_category": "Input Error",
                "rows": None,
                "columns": None,
                "status": "error",
                "file_size_kb": None,
            }
        )

    return pd.DataFrame(records)


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return str(lookup[candidate.lower()])

    return None


def make_dashboard(
    workbook: Any,
    datasets: dict[str, pd.DataFrame],
    sheet_lookup: dict[str, str],
) -> None:
    """Create an executive dashboard using available source reports."""
    sheet = workbook.add_worksheet("Executive Dashboard")

    title_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 20,
            "font_color": "#FFFFFF",
            "bg_color": "#17365D",
            "align": "left",
            "valign": "vcenter",
        }
    )
    section_format = workbook.add_format(
        {
            "bold": True,
            "font_size": 12,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
        }
    )
    label_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "#17365D",
            "bg_color": "#D9EAF7",
            "border": 1,
        }
    )
    value_format = workbook.add_format(
        {"font_size": 12, "border": 1, "num_format": "0.000"}
    )
    note_format = workbook.add_format(
        {"font_color": "#666666", "italic": True, "text_wrap": True}
    )

    sheet.set_column("A:A", 31)
    sheet.set_column("B:B", 20)
    sheet.set_column("C:H", 15)
    sheet.set_row(0, 34)

    sheet.merge_range("A1:H1", "QUANTITATIVE PORTFOLIO INTELLIGENCE", title_format)
    sheet.merge_range(
        "A2:H2",
        "Automated research, portfolio performance, risk and robustness report",
        note_format,
    )
    sheet.write("A4", "REPORTING SNAPSHOT", section_format)

    sheet.write("A5", "Generated (UTC)", label_format)
    sheet.write(
        "B5",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        value_format,
    )
    sheet.write("A6", "Source datasets loaded", label_format)
    sheet.write("B6", len(datasets), value_format)

    # Find the main performance and robustness tables.
    performance_key = next(
        (
            key
            for key in datasets
            if Path(key).name.lower() == "performance_summary.csv"
        ),
        None,
    )
    robustness_key = next(
        (
            key
            for key in datasets
            if Path(key).name.lower() == "strategy_robustness_scorecard.csv"
        ),
        None,
    )

    row = 8

    if performance_key:
        frame = datasets[performance_key]
        sheet.merge_range(
            row, 0, row, 5, "OUT-OF-SAMPLE PERFORMANCE SUMMARY", section_format
        )
        row += 1

        strategy_col = find_column(
            frame, ["strategy", "strategy_name", "portfolio", "name"]
        )
        selected_metrics = [
            "annualized_return",
            "annual_return",
            "annualized_volatility",
            "volatility",
            "sharpe_ratio",
            "sharpe",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "information_ratio",
            "beta",
            "total_return",
            "final_value",
            "total_costs",
        ]

        metric_columns = [
            column
            for column in frame.columns
            if str(column).lower() in selected_metrics
        ]

        if strategy_col and metric_columns:
            headers = [strategy_col] + metric_columns
            for col_index, header in enumerate(headers):
                sheet.write(row, col_index, str(header), label_format)

            for record_index, (_, record) in enumerate(frame.iterrows(), start=1):
                for col_index, header in enumerate(headers):
                    value = record[header]
                    if pd.isna(value):
                        sheet.write_blank(row + record_index, col_index, None)
                    elif isinstance(value, (int, float, np.number)):
                        sheet.write_number(
                            row + record_index, col_index, float(value), value_format
                        )
                    else:
                        sheet.write(
                            row + record_index, col_index, str(value)
                        )

            row += len(frame) + 3

            # Chart annualized return against annualized volatility, if available.
            return_col = find_column(
                frame, ["annualized_return", "annual_return"]
            )
            volatility_col = find_column(
                frame, ["annualized_volatility", "volatility"]
            )

            if return_col and volatility_col:
                return_index = headers.index(return_col)
                volatility_index = headers.index(volatility_col)

                chart = workbook.add_chart({"type": "scatter"})
                chart.add_series(
                    {
                        "name": "Risk vs Return",
                        "categories": [
                            "Executive Dashboard",
                            10,
                            volatility_index + 1,
                            9 + len(frame),
                            volatility_index + 1,
                        ],
                        "values": [
                            "Executive Dashboard",
                            10,
                            return_index + 1,
                            9 + len(frame),
                            return_index + 1,
                        ],
                        "marker": {"type": "circle", "size": 7},
                    }
                )
                chart.set_title({"name": "Strategy Risk vs Return"})
                chart.set_x_axis({"name": "Annualized Volatility"})
                chart.set_y_axis({"name": "Annualized Return"})
                chart.set_legend({"none": True})
                sheet.insert_chart("A22", chart, {"x_scale": 1.35, "y_scale": 1.2})
        else:
            sheet.write(
                row,
                0,
                ( 
                    "Performance summary found, but expected strategy/metric" 
                    "columns were not detected.",
                ),

                note_format,
            )
            row += 2
    else:
        sheet.write(row, 0, "Performance summary CSV was not found.", note_format)
        row += 2

    if robustness_key:
        frame = datasets[robustness_key]
        row = max(row, 35)
        sheet.merge_range(
            row, 0, row, 5, "ROBUSTNESS SCORECARD", section_format
        )
        row += 1

        columns = list(frame.columns[:8])
        for col_index, header in enumerate(columns):
            sheet.write(row, col_index, str(header), label_format)

        for record_index, (_, record) in enumerate(frame.iterrows(), start=1):
            for col_index, header in enumerate(columns):
                value = record[header]
                if pd.isna(value):
                    sheet.write_blank(row + record_index, col_index, None)
                elif isinstance(value, (int, float, np.number)):
                    sheet.write_number(
                        row + record_index, col_index, float(value), value_format
                    )
                else:
                    sheet.write(row + record_index, col_index, str(value))

    sheet.freeze_panes(4, 0)
    sheet.hide_gridlines(2)


def main() -> int:
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Reports directory: {REPORTS_DIR}")

    if not REPORTS_DIR.exists():
        log("ERROR: reports directory does not exist.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = discover_csv_files()
    log(f"Discovered {len(paths)} CSV report files.")

    datasets: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    for path in paths:
        key = relative_path(path)

        try:
            frame = load_csv(path)
            datasets[key] = frame
            log(f"Loaded {key}: {len(frame):,} rows x {len(frame.columns)} columns")
        except Exception as exc:
            errors.append({"path": key, "error": str(exc)})
            log(f"FAILED to load {key}: {exc}")

    if not datasets:
        log("ERROR: No CSV reports could be loaded.")
        log("Run the earlier analysis stages before running Stage 1.16.")
        return 1

    catalog = build_catalog(paths, datasets, errors)

    catalog_csv = OUTPUT_DIR / "report_catalog.csv"
    catalog_parquet = OUTPUT_DIR / "report_catalog.parquet"
    catalog.to_csv(catalog_csv, index=False)
    catalog.to_parquet(catalog_parquet, index=False)

    # Consolidated report inventory and run metadata.
    metadata = {
        "project": "Quantitative Portfolio Intelligence & Risk Optimization Platform",
        "stage": "1.16 - Automated Quantitative Reporting",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "source_folders": SOURCE_FOLDERS,
        "discovered_csv_files": len(paths),
        "loaded_datasets": len(datasets),
        "failed_files": errors,
        "output_directory": str(OUTPUT_DIR),
        "datasets": [
            {
                "path": key,
                "rows": len(frame),
                "columns": list(map(str, frame.columns)),
            }
            for key, frame in datasets.items()
        ],
    }

    metadata_path = OUTPUT_DIR / "reporting_manifest.json"
    metadata_path.write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Build the workbook.
    workbook_path = OUTPUT_DIR / "quantitative_portfolio_analytics.xlsx"
    used_sheet_names: set[str] = set()
    sheet_lookup: dict[str, str] = {}

    with pd.ExcelWriter(
        workbook_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd hh:mm:ss",
        date_format="yyyy-mm-dd",
    ) as writer:
        workbook = writer.book

        # Executive dashboard first.
        make_dashboard(workbook, datasets, sheet_lookup)

        # Add a catalog sheet.
        catalog_sheet = safe_sheet_name("Report Catalog", used_sheet_names)
        catalog.to_excel(writer, sheet_name=catalog_sheet, index=False)
        sheet_lookup["report_catalog"] = catalog_sheet

        # Add each discovered dataset as its own worksheet.
        for key, frame in datasets.items():
            if len(frame) > MAX_EXCEL_ROWS:
                log(
                    f"Workbook row limit exceeded for {key}; "
                    "dataset remains available in CSV source."
                )
                continue

            stem = Path(key).stem
            sheet_name = safe_sheet_name(stem, used_sheet_names)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_lookup[key] = sheet_name

            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(
                0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0)
            )
            worksheet.hide_gridlines(2)

            header_format = workbook.add_format(
                {
                    "bold": True,
                    "font_color": "#FFFFFF",
                    "bg_color": "#1F4E78",
                    "text_wrap": True,
                    "valign": "vcenter",
                    "border": 0,
                }
            )
            number_format = workbook.add_format(
                {"num_format": "#,##0.0000"}
            )

            for column_index, column in enumerate(frame.columns):
                worksheet.write(0, column_index, str(column), header_format)

                sample = frame[column].head(300).map(
                    lambda value: "" if pd.isna(value) else str(value))
                max_length = max(
                    [len(str(column))] + [len(value) for value in sample]
                )
                worksheet.set_column(
                    column_index,
                    column_index,
                    min(max(max_length + 2, 12), 28),
                    number_format if pd.api.types.is_numeric_dtype(frame[column]) else None,
                )

                if pd.api.types.is_numeric_dtype(frame[column]) and len(frame) > 1:
                    worksheet.conditional_format(
                        1,
                        column_index,
                        len(frame),
                        column_index,
                        {
                            "type": "3_color_scale",
                            "min_color": "#F8696B",
                            "mid_color": "#FFEB84",
                            "max_color": "#63BE7B",
                        },
                    )

            worksheet.set_row(0, 30)

        # Add an errors sheet if any source could not be read.
        if errors:
            error_frame = pd.DataFrame(errors)
            error_sheet = safe_sheet_name("Input Errors", used_sheet_names)
            error_frame.to_excel(writer, sheet_name=error_sheet, index=False)
            writer.sheets[error_sheet].freeze_panes(1, 0)

    # Save a concise machine-readable execution summary.
    summary = {
        "status": "completed_with_warnings" if errors else "completed",
        "discovered_files": len(paths),
        "loaded_datasets": len(datasets),
        "failed_files": len(errors),
        "workbook": str(workbook_path),
        "catalog_csv": str(catalog_csv),
        "catalog_parquet": str(catalog_parquet),
        "manifest": str(metadata_path),
    }

    (OUTPUT_DIR / "reporting_run_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    log("Stage 1.16 reporting completed.")
    log(f"Excel workbook: {workbook_path}")
    log(f"Report catalog: {catalog_csv}")
    log(f"JSON manifest: {metadata_path}")

    if errors:
        log(f"WARNING: {len(errors)} input file(s) could not be loaded.")
        for error in errors:
            log(f"  {error['path']}: {error['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
