
"""Inspect downloaded raw market datasets without modifying them."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"


def inspect_dataset(path: Path) -> None:
    """Print structural and data-quality information for one dataset."""

    data = pd.read_parquet(path)

    print("\n" + "=" * 75)
    print(f"DATASET: {path.name}")
    print("=" * 75)

    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Duplicate dates: {data.index.duplicated().sum()}")
    print(f"Missing cells: {data.isna().sum().sum()}")
    print(f"Infinite values: {np.isinf(data.select_dtypes(include=[np.number])).sum().sum()}")

    if isinstance(data.columns, pd.MultiIndex):
        tickers = data.columns.get_level_values(0).unique()

        print(f"\nNumber of instruments: {len(tickers)}")
        print("Instruments:", sorted(tickers.tolist()))

        print("\nMissing values by instrument:")

        missing_by_column = data.isna().sum()
        missing_by_ticker = missing_by_column.groupby(level=0).sum()

        print(missing_by_ticker.to_string())

        print("\nMissing values by instrument and field:")

        for ticker in tickers:
            ticker_data = data[ticker]
            missing = ticker_data.isna().sum()
            missing = missing[missing > 0]

            if not missing.empty:
                print(f"\n{ticker}:")
                print(missing.to_string())

    else:
        print("\nColumns:", data.columns.tolist())
        print("\nMissing values by column:")
        print(data.isna().sum().to_string())


def main() -> None:
    """Inspect all available raw Parquet datasets."""

    files = sorted(RAW_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError("No raw Parquet datasets were found.")

    for path in files:
        inspect_dataset(path)


if __name__ == "__main__":
    main()
