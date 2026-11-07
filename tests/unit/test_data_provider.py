
"""Unit tests for market data provider utilities."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pandas as pd
import pytest
import yfinance as yf

from quantfin.data.provider import (
    download_market_data,
    get_provider_metadata,
)


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Return representative downloaded market data."""
    index = pd.date_range("2024-01-02", periods=3, freq="B")

    columns = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Price"],
    )

    return pd.DataFrame(
        [
            [
                100, 105, 99, 103, 1_000_000,
                200, 205, 199, 203, 2_000_000,
            ],
            [
                103, 108, 102, 106, 1_100_000,
                203, 208, 202, 206, 2_100_000,
            ],
            [
                106, 110, 105, 109, 1_200_000,
                206, 210, 205, 209, 2_200_000,
            ],
        ],
        index=index,
        columns=columns,
    )


def test_download_market_data_returns_provider_data(
    monkeypatch: pytest.MonkeyPatch,
    sample_market_data: pd.DataFrame,
) -> None:
    """Return the DataFrame supplied by yfinance."""
    download_mock = Mock(return_value=sample_market_data)

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    result = download_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
    )

    pd.testing.assert_frame_equal(result, sample_market_data)
    download_mock.assert_called_once()


def test_download_market_data_forwards_default_parameters(
    monkeypatch: pytest.MonkeyPatch,
    sample_market_data: pd.DataFrame,
) -> None:
    """Forward all default download parameters to yfinance."""
    download_mock = Mock(return_value=sample_market_data)

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    download_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
    )

    download_mock.assert_called_once_with(
        tickers=["AAPL", "MSFT"],
        start="2024-01-01",
        end=None,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )


def test_download_market_data_forwards_custom_parameters(
    monkeypatch: pytest.MonkeyPatch,
    sample_market_data: pd.DataFrame,
) -> None:
    """Forward custom date, interval, and adjustment settings."""
    download_mock = Mock(return_value=sample_market_data)

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    download_market_data(
        tickers=["AAPL"],
        start_date="2020-01-01",
        end_date="2024-01-01",
        interval="1wk",
        auto_adjust=False,
    )

    download_mock.assert_called_once_with(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2024-01-01",
        interval="1wk",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )


def test_download_market_data_rejects_empty_ticker_list() -> None:
    """Reject an empty ticker list."""
    with pytest.raises(
        ValueError,
        match="At least one ticker must be provided",
    ):
        download_market_data(
            tickers=[],
            start_date="2024-01-01",
        )


@pytest.mark.parametrize(
    "tickers",
    [
        ["AAPL", "AAPL"],
        ["AAPL", "MSFT", "AAPL"],
        ["MSFT", "GOOGL", "MSFT", "GOOGL"],
    ],
)
def test_download_market_data_rejects_duplicate_tickers(
    tickers: list[str],
) -> None:
    """Reject ticker lists containing duplicate symbols."""
    with pytest.raises(
        ValueError,
        match="Duplicate ticker symbols are not allowed",
    ):
        download_market_data(
            tickers=tickers,
            start_date="2024-01-01",
        )


def test_download_market_data_does_not_call_provider_for_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not call yfinance when ticker validation fails."""
    download_mock = Mock()

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    with pytest.raises(ValueError):
        download_market_data(
            tickers=["AAPL", "AAPL"],
            start_date="2024-01-01",
        )

    download_mock.assert_not_called()


def test_download_market_data_rejects_empty_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise an error when yfinance returns an empty DataFrame."""
    download_mock = Mock(return_value=pd.DataFrame())

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    with pytest.raises(
        RuntimeError,
        match="The market data provider returned an empty dataset",
    ):
        download_market_data(
            tickers=["AAPL"],
            start_date="2024-01-01",
        )

    download_mock.assert_called_once()


def test_download_market_data_propagates_provider_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow provider exceptions to propagate to the caller."""
    download_mock = Mock(
        side_effect=ConnectionError("Provider connection failed")
    )

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    with pytest.raises(
        ConnectionError,
        match="Provider connection failed",
    ):
        download_market_data(
            tickers=["AAPL"],
            start_date="2024-01-01",
        )


def test_download_market_data_logs_download_start_and_completion(
    monkeypatch: pytest.MonkeyPatch,
    sample_market_data: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log the download start and successful completion."""
    download_mock = Mock(return_value=sample_market_data)

    monkeypatch.setattr(
        "quantfin.data.provider.yf.download",
        download_mock,
    )

    with caplog.at_level(
        logging.INFO,
        logger="quantfin.data.provider",
    ):
        download_market_data(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
        )

    messages = [record.getMessage() for record in caplog.records]

    assert "Downloading market data for 2 tickers." in messages
    assert any(
        "Download completed." in message
        and "Rows: 3" in message
        for message in messages
    )


def test_get_provider_metadata_returns_expected_fields() -> None:
    """Return the configured provider metadata."""
    metadata = get_provider_metadata()

    assert metadata == {
        "provider": "Yahoo Finance via yfinance",
        "library_version": yf.__version__,
        "data_interval": "1d",
        "default_adjustment": True,
        "timezone": "America/New_York",
    }


def test_get_provider_metadata_returns_dictionary() -> None:
    """Return metadata as a dictionary."""
    metadata = get_provider_metadata()

    assert isinstance(metadata, dict)


def test_get_provider_metadata_contains_installed_yfinance_version() -> None:
    """Report the installed yfinance version."""
    metadata = get_provider_metadata()

    assert metadata["library_version"] == yf.__version__
    assert isinstance(metadata["library_version"], str)
    assert metadata["library_version"]


def test_get_provider_metadata_reports_daily_interval() -> None:
    """Report the configured daily interval."""
    metadata = get_provider_metadata()

    assert metadata["data_interval"] == "1d"


def test_get_provider_metadata_reports_adjusted_prices() -> None:
    """Report that automatic price adjustment is enabled by default."""
    metadata = get_provider_metadata()

    assert metadata["default_adjustment"] is True


def test_get_provider_metadata_reports_new_york_timezone() -> None:
    """Report the configured market timezone."""
    metadata = get_provider_metadata()

    assert metadata["timezone"] == "America/New_York"
