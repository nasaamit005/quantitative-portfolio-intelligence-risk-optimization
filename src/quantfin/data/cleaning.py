
"""Validate and clean downloaded market datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports" / "tables"

logger = logging.getLogger(__name__)

DATASETS = {
    "us_equities": {
        "file": "us_equities_raw.parquet",
        "allow_complete_missing_rows": False,
    },
    "us_benchmarks": {
        "file": "us_benchmarks_raw.parquet",
        "allow_complete_missing_rows": False,
    },
    "market_indicators": {
        "file": "market_indicators_raw.parquet",
        "allow_complete_missing_rows": True,
    },
}

OHLC_FIELDS = ["Open", "High", "Low", "Close"]
REQUIRED_FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def validate_dataset(
    data: pd.DataFrame,
    dataset_name: str,
    allow_complete_missing_rows: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a multi-ticker OHLCV dataset without filling missing data."""

    if data.empty:
        raise ValueError(f"{dataset_name}: dataset is empty.")

    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError(f"{dataset_name}: expected MultiIndex columns.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{dataset_name}: expected a DatetimeIndex.")

    if data.index.has_duplicates:
        raise ValueError(f"{dataset_name}: duplicate dates detected.")

    data = data.sort_index().copy()

    tickers = data.columns.get_level_values(0).unique().tolist()
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "rows": len(data),
        "columns": len(data.columns),
        "first_date": str(data.index.min()),
        "last_date": str(data.index.max()),
        "duplicate_dates": int(data.index.duplicated().sum()),
        "tickers": {},
    }

    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            raise ValueError(f"{dataset_name}: missing ticker {ticker}.")

        ticker_data = data[ticker]

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in ticker_data.columns
        ]

        if missing_fields:
            raise ValueError(
                f"{ticker}: missing required fields {missing_fields}."
            )

        missing_by_field = ticker_data[REQUIRED_FIELDS].isna().sum()
        missing_rows = ticker_data[REQUIRED_FIELDS].isna().all(axis=1)
        partial_missing_rows = (
            ticker_data[REQUIRED_FIELDS].isna().any(axis=1)
            & ~missing_rows
        )

        if partial_missing_rows.any():
            dates = ticker_data.index[partial_missing_rows].astype(str).tolist()
            raise ValueError(
                f"{ticker}: partially missing OHLCV rows detected: {dates[:10]}"
            )

        if missing_rows.any() and not allow_complete_missing_rows:
            dates = ticker_data.index[missing_rows].astype(str).tolist()
            raise ValueError(
                f"{ticker}: missing OHLCV observations detected: {dates[:10]}"
            )

        numeric = ticker_data[REQUIRED_FIELDS].select_dtypes(
            include=[np.number]
        )

        if numeric.shape[1] != len(REQUIRED_FIELDS):
            raise ValueError(f"{ticker}: non-numeric OHLCV values detected.")

        if np.isinf(numeric.to_numpy()).any():
            raise ValueError(f"{ticker}: infinite values detected.")

        valid = ticker_data.loc[~missing_rows]

        if (valid[OHLC_FIELDS] <= 0).any().any():
            raise ValueError(f"{ticker}: non-positive OHLC prices detected.")

        # Allow negligible floating-point precision differences.
        OHLC_TOLERANCE = 1e-10

        invalid_high = (
            valid["High"] + OHLC_TOLERANCE
            < valid[["Open", "Close", "Low"]].max(axis=1)
        )

        invalid_low = (
            valid["Low"] - OHLC_TOLERANCE
            > valid[["Open", "Close", "High"]].min(axis=1)
        )

        if invalid_high.any() or invalid_low.any():
            raise ValueError(f"{ticker}: inconsistent OHLC values detected.")

        if (valid["Volume"] < 0).any():
            raise ValueError(f"{ticker}: negative volume detected.")

        report["tickers"][ticker] = {
            "missing_cells": int(missing_by_field.sum()),
            "missing_rows": int(missing_rows.sum()),
            "missing_by_field": {
                field: int(value)
                for field, value in missing_by_field.items()
            },
            "valid_rows": int((~missing_rows).sum()),
            "status": "PASS",
        }

    report["total_missing_cells"] = int(data.isna().sum().sum())
    report["status"] = "PASS"

    return data, report


def save_clean_dataset(data: pd.DataFrame, name: str) -> None:
    """Save validated data in Parquet and CSV formats."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data.to_parquet(PROCESSED_DIR / f"{name}_clean.parquet")
    data.to_csv(PROCESSED_DIR / f"{name}_clean.csv")

    logger.info("Saved cleaned dataset: %s", name)


def main() -> None:
    """Validate all raw datasets and generate quality reports."""

    all_reports = []

    for name, settings in DATASETS.items():
        raw_path = RAW_DIR / settings["file"]

        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset not found: {raw_path}")

        logger.info("Validating %s", name)

        raw_data = pd.read_parquet(raw_path)

        clean_data, report = validate_dataset(
            data=raw_data,
            dataset_name=name,
            allow_complete_missing_rows=settings[
                "allow_complete_missing_rows"
            ],
        )

        save_clean_dataset(clean_data, name)
        all_reports.append(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "data_quality_report.json"

    report_path.write_text(
        json.dumps(all_reports, indent=4),
        encoding="utf-8",
    )

    summary = []

    for report in all_reports:
        summary.append({
            "dataset": report["dataset"],
            "rows": report["rows"],
            "columns": report["columns"],
            "instruments": len(report["tickers"]),
            "missing_cells": report["total_missing_cells"],
            "status": report["status"],
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        REPORT_DIR / "data_quality_summary.csv",
        index=False,
    )

    print("\nDATA VALIDATION SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("\nAll datasets passed validation.")
    print("Raw files were not modified.")
    print(f"Detailed report: {report_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
