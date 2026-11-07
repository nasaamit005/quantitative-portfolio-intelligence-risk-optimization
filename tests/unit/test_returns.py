
"""Unit tests for simple and logarithmic return calculations."""

import numpy as np
import pandas as pd
import pytest

from quantfin.data.returns import calculate_returns


def test_calculate_simple_and_log_returns():
    """Verify simple and logarithmic returns against known values."""

    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 99.0]},
        index=pd.date_range("2025-01-01", periods=3),
    )

    simple, logarithmic = calculate_returns(prices)

    expected_simple = np.array([np.nan, 0.10, -0.10])
    expected_log = np.array(
        [np.nan, np.log(1.10), np.log(0.90)]
    )

    np.testing.assert_allclose(
        simple["AAPL"].to_numpy(),
        expected_simple,
        equal_nan=True,
    )

    np.testing.assert_allclose(
        logarithmic["AAPL"].to_numpy(),
        expected_log,
        equal_nan=True,
    )


def test_returns_are_calculated_in_chronological_order():
    """Verify that unsorted price dates are sorted before calculation."""

    dates = pd.to_datetime(
        ["2025-01-03", "2025-01-01", "2025-01-02"]
    )

    prices = pd.DataFrame(
        {"AAPL": [121.0, 100.0, 110.0]},
        index=dates,
    )

    simple, logarithmic = calculate_returns(prices)

    assert simple.index.is_monotonic_increasing
    assert logarithmic.index.is_monotonic_increasing

    assert np.isnan(simple.iloc[0, 0])
    assert simple.iloc[1, 0] == pytest.approx(0.10)
    assert simple.iloc[2, 0] == pytest.approx(0.10)


def test_missing_prices_are_not_forward_filled():
    """Verify missing prices do not create artificial returns."""

    prices = pd.DataFrame(
        {"AAPL": [100.0, np.nan, 110.0, 121.0]},
        index=pd.date_range("2025-01-01", periods=4),
    )

    simple, logarithmic = calculate_returns(prices)

    assert simple.iloc[0, 0] != simple.iloc[0, 0] or np.isnan(
        simple.iloc[0, 0]
    )
    assert np.isnan(simple.iloc[1, 0])
    assert np.isnan(simple.iloc[2, 0])
    assert simple.iloc[3, 0] == pytest.approx(0.10)

    assert np.isnan(logarithmic.iloc[1, 0])
    assert np.isnan(logarithmic.iloc[2, 0])
    assert logarithmic.iloc[3, 0] == pytest.approx(np.log(1.10))


def test_multiple_assets_are_calculated_independently():
    """Verify returns for multiple assets."""

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [200.0, 180.0, 198.0],
        },
        index=pd.date_range("2025-01-01", periods=3),
    )

    simple, logarithmic = calculate_returns(prices)

    assert simple.loc["2025-01-02", "AAPL"] == pytest.approx(0.10)
    assert simple.loc["2025-01-02", "MSFT"] == pytest.approx(-0.10)

    assert logarithmic.loc["2025-01-03", "AAPL"] == pytest.approx(
        np.log(1.10)
    )


def test_empty_price_dataframe_raises_error():
    """An empty price dataset must be rejected."""

    prices = pd.DataFrame()

    with pytest.raises(ValueError, match="Price dataset is empty"):
        calculate_returns(prices)


def test_duplicate_dates_raise_error():
    """Duplicate dates must be rejected."""

    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0]},
        index=pd.to_datetime(["2025-01-01", "2025-01-01"]),
    )

    with pytest.raises(ValueError, match="Duplicate dates"):
        calculate_returns(prices)


@pytest.mark.parametrize(
    "invalid_price",
    [0.0, -1.0, -100.0],
)
def test_non_positive_prices_raise_error(invalid_price):
    """Zero and negative prices must be rejected."""

    prices = pd.DataFrame(
        {"AAPL": [100.0, invalid_price]},
        index=pd.date_range("2025-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_returns(prices)


def test_first_return_observation_is_missing():
    """The first return must be NaN because no prior price exists."""

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 105.0],
            "MSFT": [200.0, 210.0],
        },
        index=pd.date_range("2025-01-01", periods=2),
    )

    simple, logarithmic = calculate_returns(prices)

    assert simple.iloc[0].isna().all()
    assert logarithmic.iloc[0].isna().all()


def test_simple_and_log_returns_satisfy_mathematical_identity():
    """Verify log return equals log(1 + simple return)."""

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 105.0, 110.0, 95.0, 120.0],
            "MSFT": [200.0, 190.0, 210.0, 215.0, 205.0],
        },
        index=pd.date_range("2025-01-01", periods=5),
    )

    simple, logarithmic = calculate_returns(prices)

    expected_log = np.log1p(simple.iloc[1:])

    np.testing.assert_allclose(
        logarithmic.iloc[1:].to_numpy(),
        expected_log.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
