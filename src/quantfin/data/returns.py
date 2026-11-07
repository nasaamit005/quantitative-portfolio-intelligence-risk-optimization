
"""Calculate simple and logarithmic returns from validated prices."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

logger = logging.getLogger(__name__)

PRICE_FIELD = "Close"


def load_clean_prices(filename: str) -> pd.DataFrame:
    """Load validated closing prices from a cleaned dataset."""

    path = PROCESSED_DIR / f"{filename}_clean.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {path}")

    data = pd.read_parquet(path)

    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected multi-level ticker and price columns.")

    prices = data.xs(PRICE_FIELD, axis=1, level=1)

    return prices.sort_index()


def calculate_returns(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate simple and log returns without filling missing prices."""

    if prices.empty:
        raise ValueError("Price dataset is empty.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates found in price index.")

    prices = prices.sort_index().astype(float)

    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")

    simple_returns = prices.pct_change(fill_method=None)

    log_returns = np.log(prices / prices.shift(1))

    simple_returns = simple_returns.replace(
        [np.inf, -np.inf], np.nan
    )

    log_returns = log_returns.replace(
        [np.inf, -np.inf], np.nan
    )

    return simple_returns, log_returns


def save_returns(
    returns: pd.DataFrame,
    filename: str,
) -> None:
    """Save return data in Parquet and CSV formats."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    returns.to_parquet(PROCESSED_DIR / f"{filename}.parquet")
    returns.to_csv(PROCESSED_DIR / f"{filename}.csv")


def main() -> None:
    """Generate returns for equities and benchmarks."""

    datasets = ["us_equities", "us_benchmarks"]

    for dataset in datasets:
        prices = load_clean_prices(dataset)

        simple_returns, log_returns = calculate_returns(prices)

        save_returns(
            simple_returns,
            f"{dataset}_simple_returns",
        )

        save_returns(
            log_returns,
            f"{dataset}_log_returns",
        )

        print(f"\n{dataset.upper()}")
        print("-" * 55)
        print("Price observations:", len(prices))
        print("Number of instruments:", len(prices.columns))
        print("Simple return shape:", simple_returns.shape)
        print("Log return shape:", log_returns.shape)
        print("Missing simple returns:", simple_returns.isna().sum().sum())
        print("Missing log returns:", log_returns.isna().sum().sum())

    logger.info("Return calculation completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
