
"""Unit tests for historical market data download utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

import quantfin.data.download as download_module
from quantfin.data.download import (
    download_group,
    load_asset_config,
    main,
    save_dataset,
)


@pytest.fixture
def sample_multiindex_data() -> pd.DataFrame:
    """Create representative multi-ticker market data."""
    index = pd.date_range("2024-01-02", periods=3, freq="B")

    columns = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Price"],
    )

    values = [
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
    ]

    return pd.DataFrame(values, index=index, columns=columns)


@pytest.fixture
def sample_singleindex_data() -> pd.DataFrame:
    """Create representative single-ticker market data."""
    index = pd.date_range("2024-01-02", periods=3, freq="B")

    return pd.DataFrame(
        {
            "SPY": [470.0, 472.0, 475.0],
            "VIX": [13.0, 14.0, 12.5],
        },
        index=index,
    )


def test_load_asset_config_reads_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the configured asset groups from a YAML file."""
    config_path = tmp_path / "assets.yaml"
    config_path.write_text(
        """
historical_data:
  start_date: "2015-01-01"
  end_date: null
assets:
  AAPL:
    name: Apple
  MSFT:
    name: Microsoft
benchmarks:
  - SPY
market_indicators:
  - ^VIX
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(download_module, "CONFIG_PATH", config_path)

    config = load_asset_config()

    assert config["historical_data"]["start_date"] == "2015-01-01"
    assert config["historical_data"]["end_date"] is None
    assert list(config["assets"]) == ["AAPL", "MSFT"]
    assert config["benchmarks"] == ["SPY"]
    assert config["market_indicators"] == ["^VIX"]


def test_load_asset_config_propagates_missing_file_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate an error when the configuration file is missing."""
    missing_path = tmp_path / "missing.yaml"

    monkeypatch.setattr(download_module, "CONFIG_PATH", missing_path)

    with pytest.raises(FileNotFoundError):
        load_asset_config()


def test_load_asset_config_propagates_invalid_yaml_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate YAML parsing errors."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "historical_data: [invalid",
        encoding="utf-8",
    )

    monkeypatch.setattr(download_module, "CONFIG_PATH", config_path)

    with pytest.raises(yaml.YAMLError):
        load_asset_config()


def test_save_dataset_creates_parquet_and_csv(
    tmp_path: Path,
    sample_multiindex_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save and verify a dataset in both supported formats."""
    raw_dir = tmp_path / "raw"

    monkeypatch.setattr(download_module, "RAW_DIR", raw_dir)

    result = save_dataset(sample_multiindex_data, "test_dataset")

    parquet_path = raw_dir / "test_dataset.parquet"
    csv_path = raw_dir / "test_dataset.csv"

    assert result == parquet_path
    assert parquet_path.exists()
    assert csv_path.exists()

    restored_parquet = pd.read_parquet(parquet_path)

    pd.testing.assert_frame_equal(
        restored_parquet,
        sample_multiindex_data,
        check_freq=False,
    )

    restored_csv = pd.read_csv(
        csv_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    pd.testing.assert_frame_equal(
        restored_csv,
        sample_multiindex_data,
        check_freq=False,
    )


def test_save_dataset_creates_missing_directory(
    tmp_path: Path,
    sample_singleindex_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create the raw-data directory when it does not exist."""
    raw_dir = tmp_path / "nested" / "raw"

    monkeypatch.setattr(download_module, "RAW_DIR", raw_dir)

    result = save_dataset(sample_singleindex_data, "single_index_dataset")

    assert raw_dir.is_dir()
    assert result.exists()
    assert (raw_dir / "single_index_dataset.csv").exists()


def test_save_dataset_logs_saved_paths(
    tmp_path: Path,
    sample_singleindex_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log both output paths after saving."""
    raw_dir = tmp_path / "raw"

    monkeypatch.setattr(download_module, "RAW_DIR", raw_dir)

    with caplog.at_level(
        logging.INFO,
        logger="quantfin.data.download",
    ):
        save_dataset(sample_singleindex_data, "logged_dataset")

    messages = [record.getMessage() for record in caplog.records]

    assert any("logged_dataset.parquet" in message for message in messages)
    assert any("logged_dataset.csv" in message for message in messages)


def test_download_group_downloads_and_saves_data(
    sample_multiindex_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download a group and save its raw data."""
    provider_mock = Mock(return_value=sample_multiindex_data)
    save_mock = Mock()

    monkeypatch.setattr(
        download_module,
        "download_market_data",
        provider_mock,
    )
    monkeypatch.setattr(download_module, "save_dataset", save_mock)

    result = download_group(
        tickers=["AAPL", "MSFT"],
        start_date="2015-01-01",
        end_date="2024-01-01",
        filename="equities",
    )

    pd.testing.assert_frame_equal(result, sample_multiindex_data)

    provider_mock.assert_called_once_with(
        tickers=["AAPL", "MSFT"],
        start_date="2015-01-01",
        end_date="2024-01-01",
        interval="1d",
        auto_adjust=True,
    )

    save_mock.assert_called_once_with(sample_multiindex_data, "equities")


def test_download_group_accepts_missing_end_date(
    sample_multiindex_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass a missing end date through to the provider."""
    provider_mock = Mock(return_value=sample_multiindex_data)
    save_mock = Mock()

    monkeypatch.setattr(
        download_module,
        "download_market_data",
        provider_mock,
    )
    monkeypatch.setattr(download_module, "save_dataset", save_mock)

    download_group(
        tickers=["AAPL"],
        start_date="2015-01-01",
        end_date=None,
        filename="equities",
    )

    assert provider_mock.call_args.kwargs["end_date"] is None


def test_download_group_rejects_empty_ticker_list() -> None:
    """Reject a group with no configured tickers."""
    with pytest.raises(
        ValueError,
        match="No tickers configured for equities",
    ):
        download_group(
            tickers=[],
            start_date="2015-01-01",
            end_date=None,
            filename="equities",
        )


def test_download_group_does_not_call_provider_for_empty_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid provider and save calls when the group is empty."""
    provider_mock = Mock()
    save_mock = Mock()

    monkeypatch.setattr(
        download_module,
        "download_market_data",
        provider_mock,
    )
    monkeypatch.setattr(download_module, "save_dataset", save_mock)

    with pytest.raises(ValueError):
        download_group(
            tickers=[],
            start_date="2015-01-01",
            end_date=None,
            filename="empty_group",
        )

    provider_mock.assert_not_called()
    save_mock.assert_not_called()


def test_download_group_propagates_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate download errors without attempting to save."""
    provider_mock = Mock(side_effect=RuntimeError("Download failed"))
    save_mock = Mock()

    monkeypatch.setattr(
        download_module,
        "download_market_data",
        provider_mock,
    )
    monkeypatch.setattr(download_module, "save_dataset", save_mock)

    with pytest.raises(RuntimeError, match="Download failed"):
        download_group(
            tickers=["AAPL"],
            start_date="2015-01-01",
            end_date=None,
            filename="equities",
        )

    save_mock.assert_not_called()


def test_download_group_propagates_save_error(
    sample_multiindex_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate errors raised while saving downloaded data."""
    monkeypatch.setattr(
        download_module,
        "download_market_data",
        Mock(return_value=sample_multiindex_data),
    )
    monkeypatch.setattr(
        download_module,
        "save_dataset",
        Mock(side_effect=OSError("Disk unavailable")),
    )

    with pytest.raises(OSError, match="Disk unavailable"):
        download_group(
            tickers=["AAPL"],
            start_date="2015-01-01",
            end_date=None,
            filename="equities",
        )


def test_main_downloads_all_configured_groups(
    sample_multiindex_data: pd.DataFrame,
    sample_singleindex_data: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download equities, benchmarks, and indicators."""
    config = {
        "historical_data": {
            "start_date": "2015-01-01",
            "end_date": "2026-11-03",
        },
        "assets": {
            "AAPL": {},
            "MSFT": {},
        },
        "benchmarks": ["SPY"],
        "market_indicators": ["^VIX"],
    }

    metadata_dir = tmp_path / "reference"

    monkeypatch.setattr(
        download_module,
        "load_asset_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(download_module, "METADATA_DIR", metadata_dir)

    group_mock = Mock(
        side_effect=[
            sample_multiindex_data,
            sample_singleindex_data,
            sample_singleindex_data,
        ]
    )

    monkeypatch.setattr(download_module, "download_group", group_mock)

    main()

    assert group_mock.call_count == 3

    assert group_mock.call_args_list[0].args == (
        ["AAPL", "MSFT"],
        "2015-01-01",
        "2026-11-03",
        "us_equities_raw",
    )
    assert group_mock.call_args_list[1].args == (
        ["SPY"],
        "2015-01-01",
        "2026-11-03",
        "us_benchmarks_raw",
    )
    assert group_mock.call_args_list[2].args == (
        ["^VIX"],
        "2015-01-01",
        "2026-11-03",
        "market_indicators_raw",
    )

    assert (metadata_dir / "download_metadata.json").exists()


def test_main_writes_correct_metadata(
    sample_multiindex_data: pd.DataFrame,
    sample_singleindex_data: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write group-level metadata for all downloaded datasets."""
    config = {
        "historical_data": {
            "start_date": "2015-01-01",
            "end_date": None,
        },
        "assets": {"AAPL": {}, "MSFT": {}},
        "benchmarks": ["SPY"],
        "market_indicators": ["^VIX"],
    }

    metadata_dir = tmp_path / "reference"

    monkeypatch.setattr(
        download_module,
        "load_asset_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(download_module, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(
        download_module,
        "download_group",
        Mock(
            side_effect=[
                sample_multiindex_data,
                sample_singleindex_data,
                sample_singleindex_data,
            ]
        ),
    )

    main()

    metadata_path = metadata_dir / "download_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["provider"] == "Yahoo Finance via yfinance"
    assert metadata["start_date_requested"] == "2015-01-01"
    assert metadata["end_date_requested"] is None

    timestamp = datetime.fromisoformat(metadata["downloaded_at_utc"])

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)

    assert metadata["groups"]["equities"] == {
        "tickers": ["AAPL", "MSFT"],
        "rows": 3,
        "columns": 10,
        "first_date": str(sample_multiindex_data.index.min()),
        "last_date": str(sample_multiindex_data.index.max()),
    }

    assert metadata["groups"]["benchmarks"] == {
        "tickers": ["SPY", "VIX"],
        "rows": 3,
        "columns": 2,
        "first_date": str(sample_singleindex_data.index.min()),
        "last_date": str(sample_singleindex_data.index.max()),
    }


def test_main_prints_download_summary(
    sample_multiindex_data: pd.DataFrame,
    sample_singleindex_data: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the market data download summary."""
    config = {
        "historical_data": {
            "start_date": "2015-01-01",
            "end_date": None,
        },
        "assets": {"AAPL": {}},
        "benchmarks": ["SPY"],
        "market_indicators": ["^VIX"],
    }

    monkeypatch.setattr(
        download_module,
        "load_asset_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        download_module,
        "METADATA_DIR",
        tmp_path / "reference",
    )
    monkeypatch.setattr(
        download_module,
        "download_group",
        Mock(
            side_effect=[
                sample_multiindex_data,
                sample_singleindex_data,
                sample_singleindex_data,
            ]
        ),
    )

    main()

    output = capsys.readouterr().out

    assert "MARKET DATA DOWNLOAD SUMMARY" in output
    assert "EQUITIES" in output
    assert "BENCHMARKS" in output
    assert "MARKET_INDICATORS" in output
    assert "Tickers:" in output
    assert "Rows:" in output
    assert "First date:" in output
    assert "Last date:" in output
    assert "Raw datasets saved successfully." in output


def test_main_supports_empty_optional_config_groups(
    sample_multiindex_data: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use empty lists when optional groups are absent."""
    config = {
        "historical_data": {
            "start_date": "2015-01-01",
        },
        "assets": {"AAPL": {}},
    }

    monkeypatch.setattr(
        download_module,
        "load_asset_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        download_module,
        "METADATA_DIR",
        tmp_path / "reference",
    )

    group_mock = Mock(
        side_effect=[
            sample_multiindex_data,
            ValueError("No tickers configured for us_benchmarks_raw"),
        ]
    )

    monkeypatch.setattr(download_module, "download_group", group_mock)

    with pytest.raises(
        ValueError,
        match="No tickers configured for us_benchmarks_raw",
    ):
        main()

    assert group_mock.call_count == 2
