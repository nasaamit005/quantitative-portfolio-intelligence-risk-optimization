
"""Unit tests for historical risk analytics."""

import numpy as np
import pandas as pd
import pytest

from quantfin.analytics.risk import (
    TRADING_DAYS,
    calculate_asset_risk,
    calculate_beta,
    calculate_downside_deviation,
    calculate_historical_cvar,
    calculate_historical_var,
    calculate_information_ratio,
    calculate_maximum_drawdown,
    calculate_tracking_error,
)


@pytest.fixture
def asset_returns():
    """Create deterministic daily asset returns."""
    return pd.Series(
        [0.02, -0.01, 0.015, -0.03, 0.01, -0.02, 0.025, -0.005],
        index=pd.date_range("2025-01-01", periods=8),
        name="AAPL",
    )


@pytest.fixture
def benchmark_returns():
    """Create deterministic daily benchmark returns."""
    return pd.Series(
        [0.01, -0.005, 0.01, -0.015, 0.005, -0.01, 0.015, 0.0],
        index=pd.date_range("2025-01-01", periods=8),
        name="SPY",
    )


def test_historical_var_95(asset_returns):
    """Verify historical 95% VaR is expressed as a positive loss."""
    result = calculate_historical_var(asset_returns, 0.95)

    expected = -asset_returns.quantile(0.05)

    assert result == pytest.approx(expected)


def test_historical_var_99(asset_returns):
    """Verify historical 99% VaR."""
    result = calculate_historical_var(asset_returns, 0.99)

    expected = -asset_returns.quantile(0.01)

    assert result == pytest.approx(expected)


def test_historical_cvar_95(asset_returns):
    """Verify CVaR is the positive mean loss in the 5% tail."""
    result = calculate_historical_cvar(asset_returns, 0.95)

    threshold = asset_returns.quantile(0.05)
    expected = -asset_returns[asset_returns <= threshold].mean()

    assert result == pytest.approx(expected)


def test_historical_cvar_is_at_least_var_for_loss_dominated_data():
    """CVaR should be at least VaR for this loss-dominated sample."""
    returns = pd.Series(
        [-0.20, -0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04]
    )

    var = calculate_historical_var(returns, 0.95)
    cvar = calculate_historical_cvar(returns, 0.95)

    assert cvar >= var


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_invalid_var_confidence_raises_error(confidence, asset_returns):
    """Invalid confidence levels must be rejected."""
    with pytest.raises(ValueError, match="Confidence"):
        calculate_historical_var(asset_returns, confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_invalid_cvar_confidence_raises_error(confidence, asset_returns):
    """Invalid CVaR confidence levels must be rejected."""
    with pytest.raises(ValueError, match="Confidence"):
        calculate_historical_cvar(asset_returns, confidence)


def test_maximum_drawdown_includes_initial_wealth():
    """Verify peak-to-trough drawdown includes initial portfolio wealth."""
    returns = pd.Series(
        [0.10, -0.20, 0.05],
        index=pd.date_range("2025-01-01", periods=3),
    )

    result = calculate_maximum_drawdown(returns)

    assert result == pytest.approx(-0.20)


def test_maximum_drawdown_for_monotonically_increasing_returns():
    """A monotonically increasing portfolio should have zero drawdown."""
    returns = pd.Series(
        [0.01, 0.02, 0.015, 0.01],
        index=pd.date_range("2025-01-01", periods=4),
    )

    result = calculate_maximum_drawdown(returns)

    assert result == pytest.approx(0.0)


def test_maximum_drawdown_empty_series_raises_error():
    """Empty return series must be rejected."""
    with pytest.raises(ValueError, match="empty returns"):
        calculate_maximum_drawdown(pd.Series(dtype=float))


def test_downside_deviation():
    """Verify annualized downside deviation mathematically."""
    returns = pd.Series([-0.02, 0.01, -0.04, 0.03])

    result = calculate_downside_deviation(returns)

    downside = np.minimum(returns.to_numpy(), 0.0)

    expected = np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)

    assert result == pytest.approx(expected)


def test_downside_deviation_with_nonzero_target():
    """Verify downside deviation with an annual target return."""
    returns = pd.Series([-0.01, 0.02, 0.005, -0.03])
    target = 0.05

    result = calculate_downside_deviation(returns, target)

    daily_target = target / TRADING_DAYS
    downside = np.minimum(returns.to_numpy() - daily_target, 0.0)

    expected = np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)

    assert result == pytest.approx(expected)


def test_beta_against_benchmark(asset_returns, benchmark_returns):
    """Verify beta equals covariance divided by benchmark variance."""
    result = calculate_beta(asset_returns, benchmark_returns)

    expected = (
        asset_returns.cov(benchmark_returns)
        / benchmark_returns.var(ddof=1)
    )

    assert result == pytest.approx(expected)


def test_beta_handles_zero_benchmark_variance(asset_returns):
    """Zero benchmark variance should return NaN."""
    benchmark = pd.Series(
        np.zeros(len(asset_returns)),
        index=asset_returns.index,
    )

    result = calculate_beta(asset_returns, benchmark)

    assert np.isnan(result)


def test_beta_handles_insufficient_aligned_observations():
    """Fewer than two aligned observations should return NaN."""
    asset = pd.Series(
        [0.01],
        index=pd.to_datetime(["2025-01-01"]),
    )
    benchmark = pd.Series(
        [0.02],
        index=pd.to_datetime(["2025-01-01"]),
    )

    assert np.isnan(calculate_beta(asset, benchmark))


def test_tracking_error(asset_returns, benchmark_returns):
    """Verify annualized tracking error."""
    result = calculate_tracking_error(asset_returns, benchmark_returns)

    active_returns = asset_returns - benchmark_returns
    expected = active_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    assert result == pytest.approx(expected)


def test_information_ratio(asset_returns, benchmark_returns):
    """Verify annualized information ratio."""
    result = calculate_information_ratio(
        asset_returns,
        benchmark_returns,
    )

    active_returns = asset_returns - benchmark_returns

    expected = (
        active_returns.mean() * TRADING_DAYS
        / (active_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    )

    assert result == pytest.approx(expected)


def test_information_ratio_handles_zero_tracking_error():
    """Identical asset and benchmark returns should produce NaN."""
    returns = pd.Series(
        [0.01, -0.02, 0.015, 0.005],
        index=pd.date_range("2025-01-01", periods=4),
    )

    result = calculate_information_ratio(returns, returns)

    assert np.isnan(result)


def test_asset_risk_returns_core_metrics(asset_returns):
    """Verify all core standalone risk metrics are returned."""
    result = calculate_asset_risk(asset_returns)

    expected_fields = {
        "observations",
        "annualized_volatility",
        "downside_deviation",
        "maximum_drawdown",
        "historical_var_95",
        "historical_cvar_95",
        "historical_var_99",
        "historical_cvar_99",
    }

    assert set(result) == expected_fields
    assert result["observations"] == len(asset_returns)
    assert result["annualized_volatility"] >= 0
    assert result["downside_deviation"] >= 0
    assert -1 <= result["maximum_drawdown"] <= 0


def test_asset_risk_includes_benchmark_relative_metrics(
    asset_returns,
    benchmark_returns,
):
    """Verify benchmark-relative measures are included when supplied."""
    result = calculate_asset_risk(
        asset_returns,
        benchmark_returns,
    )

    assert "beta_vs_spy" in result
    assert "tracking_error_vs_spy" in result
    assert "information_ratio_vs_spy" in result

    assert np.isfinite(result["beta_vs_spy"])
    assert np.isfinite(result["tracking_error_vs_spy"])
    assert np.isfinite(result["information_ratio_vs_spy"])


def test_asset_risk_rejects_insufficient_observations():
    """Asset risk requires at least two valid observations."""
    returns = pd.Series([0.01], name="AAPL")

    with pytest.raises(ValueError, match="insufficient observations"):
        calculate_asset_risk(returns)


def test_asset_risk_rejects_infinite_observations():
    """Infinite returns must be rejected."""
    returns = pd.Series([0.01, np.inf, -0.02], name="AAPL")

    with pytest.raises(ValueError, match="invalid return observations"):
        calculate_asset_risk(returns)


def test_asset_risk_rejects_returns_at_or_below_negative_one():
    """A return at or below -100% must be rejected."""
    returns = pd.Series([0.01, -1.0, 0.02], name="AAPL")

    with pytest.raises(ValueError, match="at or below -100%"):
        calculate_asset_risk(returns)

