
"""Unit tests for descriptive and annualized return statistics."""

import numpy as np
import pandas as pd
import pytest

from quantfin.analytics.statistics import (
    TRADING_DAYS,
    calculate_asset_statistics,
    calculate_return_quantiles,
    calculate_rolling_metrics,
)


def sample_returns() -> pd.Series:
    """Create deterministic returns with non-zero variance."""
    values = np.array(
        [0.01, -0.02, 0.015, 0.005, -0.01, 0.025, -0.005, 0.012]
    )
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=len(values)),
        name="AAPL",
    )


def test_asset_statistics_observation_count_and_dates():
    """Check observation count and analysis period."""
    result = calculate_asset_statistics(sample_returns())

    assert result["observations"] == 8
    assert result["start_date"] == "2025-01-01"
    assert result["end_date"] == "2025-01-08"


def test_annualized_arithmetic_return():
    """Annualized arithmetic return must equal daily mean times 252."""
    returns = sample_returns()
    result = calculate_asset_statistics(returns)

    expected = returns.mean() * TRADING_DAYS

    assert result["annualized_arithmetic_return"] == pytest.approx(
        expected
    )


def test_annualized_volatility():
    """Annualized volatility must use sample standard deviation."""
    returns = sample_returns()
    result = calculate_asset_statistics(returns)

    expected = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    assert result["annualized_volatility"] == pytest.approx(
        expected
    )


def test_cagr_uses_compounded_growth():
    """CAGR must be based on compounded returns."""
    returns = sample_returns()
    result = calculate_asset_statistics(returns)

    growth = np.prod(1.0 + returns.to_numpy())
    expected = growth ** (TRADING_DAYS / len(returns)) - 1.0

    assert result["annualized_cagr"] == pytest.approx(expected)


def test_positive_and_negative_day_statistics():
    """Check positive and negative daily return counts."""
    result = calculate_asset_statistics(sample_returns())

    assert result["positive_return_days"] == 5
    assert result["negative_return_days"] == 3
    assert result["positive_day_percentage"] == pytest.approx(62.5)


def test_distribution_statistics_are_finite():
    """Skewness, kurtosis and Jarque-Bera outputs should be finite."""
    result = calculate_asset_statistics(sample_returns())

    fields = [
        "skewness",
        "excess_kurtosis",
        "jarque_bera_statistic",
        "jarque_bera_p_value",
    ]

    for field in fields:
        assert np.isfinite(result[field])

    assert result["jarque_bera_statistic"] >= 0
    assert 0 <= result["jarque_bera_p_value"] <= 1


def test_insufficient_observations_raise_error():
    """At least two valid observations are required."""
    returns = pd.Series([0.01], name="AAPL")

    with pytest.raises(ValueError, match="insufficient observations"):
        calculate_asset_statistics(returns)


def test_non_finite_returns_raise_error():
    """Infinite returns must be rejected."""
    returns = pd.Series(
        [0.01, np.inf, -0.02],
        name="AAPL",
    )

    with pytest.raises(ValueError, match="infinite or invalid"):
        calculate_asset_statistics(returns)


def test_returns_at_or_below_negative_one_raise_error():
    """A return of -100% or lower must be rejected."""
    returns = pd.Series(
        [0.01, -1.0, 0.02],
        name="AAPL",
    )

    with pytest.raises(ValueError, match="at or below -100%"):
        calculate_asset_statistics(returns)


def test_quantile_labels_and_order():
    """Check quantile labels and monotonic ordering."""
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.05, 0.05, 101),
            "MSFT": np.linspace(-0.03, 0.07, 101),
        }
    )

    quantiles = calculate_return_quantiles(returns)

    expected_labels = [
        "Q01",
        "Q05",
        "Q10",
        "Q25",
        "Q50",
        "Q75",
        "Q90",
        "Q95",
        "Q99",
    ]

    assert list(quantiles.index) == expected_labels
    assert list(quantiles.columns) == ["AAPL", "MSFT"]
    assert (quantiles.diff().iloc[1:] >= 0).all().all()


def test_rolling_metrics_have_expected_names_and_shapes():
    """Check rolling metric output names and dimensions."""
    dates = pd.date_range("2020-01-01", periods=300)
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.02, 0.025, 300),
            "MSFT": np.linspace(0.015, -0.01, 300),
        },
        index=dates,
    )

    results = calculate_rolling_metrics(returns)

    assert set(results) == {
        "rolling_21d_volatility",
        "rolling_63d_volatility",
        "rolling_252d_return",
    }

    for metric in results.values():
        assert metric.shape == returns.shape
        assert metric.index.equals(returns.index)
        assert metric.columns.equals(returns.columns)


def test_rolling_metrics_respect_warmup_periods():
    """Check that rolling metrics do not populate incomplete windows."""
    dates = pd.date_range("2020-01-01", periods=300)
    returns = pd.DataFrame(
        {"AAPL": np.linspace(-0.01, 0.02, 300)},
        index=dates,
    )

    results = calculate_rolling_metrics(returns)

    assert results["rolling_21d_volatility"].iloc[:20].isna().all().all()
    assert results["rolling_63d_volatility"].iloc[:62].isna().all().all()
    assert results["rolling_252d_return"].iloc[:251].isna().all().all()

    assert results["rolling_21d_volatility"].iloc[20:].notna().all().all()
    assert results["rolling_63d_volatility"].iloc[62:].notna().all().all()
    assert results["rolling_252d_return"].iloc[251:].notna().all().all()
