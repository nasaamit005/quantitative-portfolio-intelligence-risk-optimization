
"""Unit tests for simple and logarithmic market returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfin.data import returns


def make_prices(
    *,
    tickers: tuple[str, ...] = ("AAPL",),
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a small positive closing-price dataset."""

    if dates is None:
        dates = pd.date_range("2025-01-02", periods=4, freq="B")

    values_by_ticker = {
        "AAPL": [100.0, 110.0, 99.0, 108.9],
        "MSFT": [200.0, 210.0, 220.5, 220.5],
        "NVDA": [50.0, 55.0, 60.5, 66.55],
    }

    data = {
        ticker: values_by_ticker.get(
            ticker,
            [100.0, 105.0, 110.0, 115.0],
        )
        for ticker in tickers
    }

    return pd.DataFrame(data, index=dates)


def make_clean_ohlcv_data() -> pd.DataFrame:
    """Build a cleaned multi-ticker OHLCV dataset."""

    dates = pd.date_range("2025-01-02", periods=3, freq="B")

    columns = pd.MultiIndex.from_product(
        [
            ["AAPL", "MSFT"],
            ["Open", "High", "Low", "Close", "Volume"],
        ],
        names=["Ticker", "Field"],
    )

    values = np.array(
        [
            [
                99.0, 105.0, 95.0, 100.0, 1000.0,
                199.0, 205.0, 195.0, 200.0, 2000.0,
            ],
            [
                109.0, 115.0, 105.0, 110.0, 1100.0,
                209.0, 215.0, 205.0, 210.0, 2100.0,
            ],
            [
                119.0, 125.0, 115.0, 120.0, 1200.0,
                219.0, 225.0, 215.0, 220.0, 2200.0,
            ],
        ],
        dtype=float,
    )

    return pd.DataFrame(
        values,
        index=dates,
        columns=columns,
    )


def test_load_clean_prices_loads_close_prices(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_data = make_clean_ohlcv_data()
    processed_dir = tmp_path

    parquet_path = processed_dir / "us_equities_clean.parquet"
    parquet_path.touch()

    monkeypatch.setattr(returns, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(
        returns.pd,
        "read_parquet",
        lambda path: raw_data.copy(),
    )

    prices = returns.load_clean_prices("us_equities")

    expected = raw_data.xs(
        "Close",
        axis=1,
        level=1,
    )

    pd.testing.assert_frame_equal(prices, expected)


def test_load_clean_prices_sorts_dates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_data = make_clean_ohlcv_data().iloc[[2, 0, 1]]

    parquet_path = tmp_path / "us_equities_clean.parquet"
    parquet_path.touch()

    monkeypatch.setattr(returns, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(
        returns.pd,
        "read_parquet",
        lambda path: raw_data.copy(),
    )

    prices = returns.load_clean_prices("us_equities")

    assert prices.index.is_monotonic_increasing
    assert prices.index[0] == pd.Timestamp("2025-01-02")
    assert prices.index[-1] == pd.Timestamp("2025-01-06")


def test_load_clean_prices_raises_when_file_is_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(returns, "PROCESSED_DIR", tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Clean dataset not found",
    ):
        returns.load_clean_prices("missing_dataset")


def test_load_clean_prices_rejects_non_multiindex_columns(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "us_equities_clean.parquet"
    parquet_path.touch()

    invalid_data = pd.DataFrame(
        {"AAPL": [100.0, 101.0]},
        index=pd.date_range("2025-01-02", periods=2),
    )

    monkeypatch.setattr(returns, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(
        returns.pd,
        "read_parquet",
        lambda path: invalid_data,
    )

    with pytest.raises(
        ValueError,
        match="Expected multi-level ticker and price columns",
    ):
        returns.load_clean_prices("us_equities")


def test_load_clean_prices_raises_when_close_field_is_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_path = tmp_path / "us_equities_clean.parquet"
    parquet_path.touch()

    columns = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Volume"]],
    )

    invalid_data = pd.DataFrame(
        [[99.0, 105.0, 95.0, 1000.0]],
        index=pd.DatetimeIndex(["2025-01-02"]),
        columns=columns,
    )

    monkeypatch.setattr(returns, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(
        returns.pd,
        "read_parquet",
        lambda path: invalid_data,
    )

    with pytest.raises(KeyError):
        returns.load_clean_prices("us_equities")


def test_calculate_returns_produces_correct_simple_returns() -> None:
    prices = make_prices()

    simple_returns, _ = returns.calculate_returns(prices)

    expected = pd.DataFrame(
        {
            "AAPL": [np.nan, 0.10, -0.10, 0.10],
        },
        index=prices.index,
    )

    pd.testing.assert_frame_equal(
        simple_returns,
        expected,
        check_freq=False,
    )


def test_calculate_returns_produces_correct_log_returns() -> None:
    prices = make_prices()

    _, log_returns = returns.calculate_returns(prices)

    expected_values = np.log(
        prices["AAPL"] / prices["AAPL"].shift(1)
    )

    expected = expected_values.to_frame()

    pd.testing.assert_frame_equal(
        log_returns,
        expected,
        check_freq=False,
    )


def test_calculate_returns_handles_multiple_assets() -> None:
    prices = make_prices(tickers=("AAPL", "MSFT", "NVDA"))

    simple_returns, log_returns = returns.calculate_returns(prices)

    assert list(simple_returns.columns) == ["AAPL", "MSFT", "NVDA"]
    assert list(log_returns.columns) == ["AAPL", "MSFT", "NVDA"]
    assert simple_returns.shape == prices.shape
    assert log_returns.shape == prices.shape

    assert simple_returns.iloc[0].isna().all()
    assert log_returns.iloc[0].isna().all()


def test_calculate_returns_sorts_dates_before_calculation() -> None:
    prices = make_prices()
    shuffled_prices = prices.iloc[[2, 0, 3, 1]]

    simple_returns, log_returns = returns.calculate_returns(
        shuffled_prices
    )

    assert simple_returns.index.is_monotonic_increasing
    assert log_returns.index.is_monotonic_increasing

    assert simple_returns.index.equals(prices.index)
    assert log_returns.index.equals(prices.index)


def test_calculate_returns_rejects_empty_prices() -> None:
    prices = make_prices().iloc[0:0]

    with pytest.raises(ValueError, match="Price dataset is empty"):
        returns.calculate_returns(prices)


def test_calculate_returns_rejects_duplicate_dates() -> None:
    prices = make_prices()

    prices.index = pd.DatetimeIndex(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-02",
            "2025-01-06",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Duplicate dates found in price index",
    ):
        returns.calculate_returns(prices)


@pytest.mark.parametrize(
    ("row", "invalid_price"),
    [
        (0, 0.0),
        (1, -1.0),
        (2, -100.0),
    ],
)
def test_calculate_returns_rejects_non_positive_prices(
    row: int,
    invalid_price: float,
) -> None:
    prices = make_prices()
    prices.iloc[row, 0] = invalid_price

    with pytest.raises(
        ValueError,
        match="Prices must be strictly positive",
    ):
        returns.calculate_returns(prices)


def test_calculate_returns_preserves_missing_prices_without_filling() -> None:
    prices = make_prices()

    prices.loc[prices.index[1], "AAPL"] = np.nan

    simple_returns, log_returns = returns.calculate_returns(prices)

    # The missing price must remain missing.
    assert np.isnan(prices.loc[prices.index[1], "AAPL"])

    # The return at the missing observation is unavailable.
    assert np.isnan(simple_returns.loc[prices.index[1], "AAPL"])
    assert np.isnan(log_returns.loc[prices.index[1], "AAPL"])

    # The next return is also unavailable because its previous
    # closing price is missing.
    assert np.isnan(simple_returns.loc[prices.index[2], "AAPL"])
    assert np.isnan(log_returns.loc[prices.index[2], "AAPL"])

    # Returns resume when two consecutive valid prices are available.
    assert np.isfinite(simple_returns.loc[prices.index[3], "AAPL"])
    assert np.isfinite(log_returns.loc[prices.index[3], "AAPL"])


def test_calculate_returns_preserves_first_observation_as_missing() -> None:
    prices = make_prices(tickers=("AAPL", "MSFT"))

    simple_returns, log_returns = returns.calculate_returns(prices)

    assert simple_returns.iloc[0].isna().all()
    assert log_returns.iloc[0].isna().all()

    assert simple_returns.iloc[1:].notna().all().all()
    assert log_returns.iloc[1:].notna().all().all()


def test_calculate_returns_does_not_modify_original_prices() -> None:
    prices = make_prices()
    original = prices.copy(deep=True)

    returns.calculate_returns(prices)

    pd.testing.assert_frame_equal(prices, original)


def test_calculate_returns_converts_numeric_prices_to_float() -> None:
    prices = pd.DataFrame(
        {"AAPL": [100, 110, 121]},
        index=pd.date_range("2025-01-02", periods=3),
        dtype=int,
    )

    simple_returns, log_returns = returns.calculate_returns(prices)

    assert all(
        pd.api.types.is_float_dtype(dtype)
        for dtype in simple_returns.dtypes
    )
    assert all(
        pd.api.types.is_float_dtype(dtype)
        for dtype in log_returns.dtypes
    )


def test_save_returns_writes_parquet_and_csv(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_prices()

    monkeypatch.setattr(returns, "PROCESSED_DIR", tmp_path)

    returns.save_returns(data, "test_simple_returns")

    parquet_path = tmp_path / "test_simple_returns.parquet"
    csv_path = tmp_path / "test_simple_returns.csv"

    assert parquet_path.is_file()
    assert csv_path.is_file()

    restored_parquet = pd.read_parquet(parquet_path)

    pd.testing.assert_frame_equal(
        restored_parquet,
        data,
        check_freq=False,
    )

    restored_csv = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=True,
    )

    restored_csv.index.name = data.index.name

    pd.testing.assert_frame_equal(
        restored_csv,
        data,
        check_freq=False,
    )


def test_save_returns_creates_output_directory(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "nested" / "processed"

    monkeypatch.setattr(returns, "PROCESSED_DIR", output_dir)

    returns.save_returns(
        make_prices(),
        "test_returns",
    )

    assert output_dir.is_dir()
    assert (output_dir / "test_returns.parquet").is_file()
    assert (output_dir / "test_returns.csv").is_file()


def test_main_generates_returns_for_both_datasets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prices = make_prices(tickers=("AAPL", "MSFT"))

    loaded_datasets: list[str] = []
    saved_outputs: dict[str, pd.DataFrame] = {}

    def fake_load_clean_prices(filename: str) -> pd.DataFrame:
        loaded_datasets.append(filename)
        return prices.copy()

    def fake_save_returns(
        data: pd.DataFrame,
        filename: str,
    ) -> None:
        saved_outputs[filename] = data.copy()

    monkeypatch.setattr(
        returns,
        "load_clean_prices",
        fake_load_clean_prices,
    )
    monkeypatch.setattr(
        returns,
        "save_returns",
        fake_save_returns,
    )

    returns.main()

    assert loaded_datasets == [
        "us_equities",
        "us_benchmarks",
    ]

    assert set(saved_outputs) == {
        "us_equities_simple_returns",
        "us_equities_log_returns",
        "us_benchmarks_simple_returns",
        "us_benchmarks_log_returns",
    }

    expected_simple, expected_log = returns.calculate_returns(prices)

    pd.testing.assert_frame_equal(
        saved_outputs["us_equities_simple_returns"],
        expected_simple,
    )
    pd.testing.assert_frame_equal(
        saved_outputs["us_equities_log_returns"],
        expected_log,
    )
    pd.testing.assert_frame_equal(
        saved_outputs["us_benchmarks_simple_returns"],
        expected_simple,
    )
    pd.testing.assert_frame_equal(
        saved_outputs["us_benchmarks_log_returns"],
        expected_log,
    )

    output = capsys.readouterr().out

    assert "US_EQUITIES" in output
    assert "US_BENCHMARKS" in output
    assert "Simple return shape:" in output
    assert "Log return shape:" in output
    assert "Missing simple returns:" in output
    assert "Missing log returns:" in output
