
"""Download historical US equity, benchmark and market indicator data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from quantfin.data.provider import download_market_data

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "config" / "assets.yaml"
RAW_DIR = ROOT_DIR / "data" / "raw"
METADATA_DIR = ROOT_DIR / "data" / "reference"

logger = logging.getLogger(__name__)


def load_asset_config() -> dict:
    """Load market instruments from the YAML configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_dataset(data: pd.DataFrame, filename: str) -> Path:
    """Save a downloaded dataset in Parquet and CSV formats."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = RAW_DIR / f"{filename}.parquet"
    csv_path = RAW_DIR / f"{filename}.csv"

    data.to_parquet(parquet_path)
    data.to_csv(csv_path)

    logger.info("Saved %s", parquet_path)
    logger.info("Saved %s", csv_path)

    return parquet_path


def download_group(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    filename: str,
) -> pd.DataFrame:
    """Download one instrument group and save the raw result."""

    if not tickers:
        raise ValueError(f"No tickers configured for {filename}.")

    data = download_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        auto_adjust=True,
    )

    save_dataset(data, filename)
    return data


def main() -> None:
    """Download all configured market instruments."""

    config = load_asset_config()

    start_date = config["historical_data"]["start_date"]
    end_date = config["historical_data"].get("end_date")

    equities = list(config["assets"].keys())
    benchmarks = config.get("benchmarks", [])
    indicators = config.get("market_indicators", [])

    results = {
        "equities": download_group(
            equities, start_date, end_date, "us_equities_raw"
        ),
        "benchmarks": download_group(
            benchmarks, start_date, end_date, "us_benchmarks_raw"
        ),
        "market_indicators": download_group(
            indicators, start_date, end_date, "market_indicators_raw"
        ),
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "provider": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date_requested": start_date,
        "end_date_requested": end_date,
        "groups": {
            name: {
                "tickers": list(data.columns.get_level_values(0).unique())
                if isinstance(data.columns, pd.MultiIndex)
                else list(data.columns),
                "rows": len(data),
                "columns": len(data.columns),
                "first_date": str(data.index.min()),
                "last_date": str(data.index.max()),
            }
            for name, data in results.items()
        },
    }

    metadata_path = METADATA_DIR / "download_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nMARKET DATA DOWNLOAD SUMMARY")
    print("=" * 55)

    for name, details in metadata["groups"].items():
        print(f"\n{name.upper()}")
        print(f"Tickers: {details['tickers']}")
        print(f"Rows: {details['rows']}")
        print(f"First date: {details['first_date']}")
        print(f"Last date: {details['last_date']}")

    print("\nRaw datasets saved successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
