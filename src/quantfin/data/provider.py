
"""Market data provider configuration and download utilities."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download historical market data for the requested tickers."""

    if not tickers:
        raise ValueError("At least one ticker must be provided.")

    if len(tickers) != len(set(tickers)):
        raise ValueError("Duplicate ticker symbols are not allowed.")

    logger.info("Downloading market data for %d tickers.", len(tickers))

    data: pd.DataFrame = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError(
            "The market data provider returned an empty dataset."
        )

    logger.info(
        "Download completed. Rows: %d | Columns: %d",
        len(data),
        len(data.columns),
    )

    return data


def get_provider_metadata() -> dict[str, Any]:
    """Return basic metadata about the configured provider."""

    return {
        "provider": "Yahoo Finance via yfinance",
        "library_version": yf.__version__,
        "data_interval": "1d",
        "default_adjustment": True,
        "timezone": "America/New_York",
    }
