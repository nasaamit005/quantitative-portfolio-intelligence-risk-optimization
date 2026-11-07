
"""Unit tests for the portfolio backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from quantfin.backtesting.engine import (
    BacktestConfig,
    _rebalance_dates,
    _validate_returns,
    _validate_weights,
    run_backtest,
)


@pytest.fixture
def sample_returns():
    """Small deterministic return dataset."""
    return pd.DataFrame(
        {
            "AAPL": [0.10, 0.00, -0.05, 0.02],
            "MSFT": [0.00, 0.10, 0.02, -0.01],
        },
        index=pd.bdate_range("2025-01-06", periods=4),
    )


@pytest.fixture
def sample_weights():
    """Equal-weight portfolio."""
    return pd.Series({"AAPL": 0.5, "MSFT": 0.5})


@pytest.fixture
def no_cost_config():
    """Backtest configuration without transaction costs."""
    return BacktestConfig(
        initial_capital=100_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        rebalance_frequency="M",
        charge_initial_transaction=False,
    )


# ---------------------------------------------------------------------
# BacktestConfig tests
# ---------------------------------------------------------------------


def test_default_backtest_config():
    config = BacktestConfig()

    assert config.initial_capital == 100_000.0
    assert config.transaction_cost_bps == 5.0
    assert config.slippage_bps == 2.0
    assert config.rebalance_frequency == "M"
    assert config.charge_initial_transaction is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_capital": 0},
        {"initial_capital": -100},
        {"initial_capital": np.nan},
        {"initial_capital": np.inf},
    ],
)
def test_config_rejects_invalid_initial_capital(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transaction_cost_bps": np.nan},
        {"transaction_cost_bps": np.inf},
        {"slippage_bps": np.nan},
        {"slippage_bps": np.inf},
    ],
)
def test_config_rejects_nonfinite_costs(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transaction_cost_bps": -1},
        {"slippage_bps": -1},
    ],
)
def test_config_rejects_negative_costs(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_config_rejects_combined_costs_at_10000_bps():
    with pytest.raises(ValueError):
        BacktestConfig(
            transaction_cost_bps=5000,
            slippage_bps=5000,
        )


@pytest.mark.parametrize("frequency", ["M", "Q", "Y"])
def test_config_accepts_supported_rebalance_frequencies(frequency):
    config = BacktestConfig(rebalance_frequency=frequency)

    assert config.rebalance_frequency == frequency


def test_config_rejects_unsupported_rebalance_frequency():
    with pytest.raises(ValueError):
        BacktestConfig(rebalance_frequency="W")


# ---------------------------------------------------------------------
# Return validation tests
# ---------------------------------------------------------------------


def test_validate_returns_accepts_valid_data(sample_returns):
    result = _validate_returns(sample_returns)

    pd.testing.assert_frame_equal(
        result,
        sample_returns.astype(float),
    )


def test_validate_returns_rejects_non_dataframe():
    with pytest.raises((TypeError, ValueError)):
        _validate_returns(np.array([[0.01, 0.02]]))


def test_validate_returns_rejects_empty_data():
    with pytest.raises(ValueError):
        _validate_returns(pd.DataFrame())


def test_validate_returns_rejects_no_asset_columns():
    data = pd.DataFrame(
        index=pd.date_range("2025-01-01", periods=3)
    )

    with pytest.raises(ValueError):
        _validate_returns(data)


def test_validate_returns_rejects_duplicate_columns():
    data = pd.DataFrame(
        [[0.01, 0.02], [0.02, 0.03]],
        columns=["AAPL", "AAPL"],
        index=pd.date_range("2025-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        _validate_returns(data)


def test_validate_returns_requires_datetime_index():
    data = pd.DataFrame(
        {"AAPL": [0.01, 0.02]},
        index=[1, 2],
    )

    with pytest.raises((TypeError, ValueError)):
        _validate_returns(data)


def test_validate_returns_rejects_duplicate_dates():
    dates = pd.to_datetime(
        ["2025-01-01", "2025-01-01"]
    )

    data = pd.DataFrame(
        {"AAPL": [0.01, 0.02]},
        index=dates,
    )

    with pytest.raises(ValueError):
        _validate_returns(data)


def test_validate_returns_rejects_unsorted_dates():
    dates = pd.to_datetime(
        ["2025-01-02", "2025-01-01"]
    )

    data = pd.DataFrame(
        {"AAPL": [0.01, 0.02]},
        index=dates,
    )

    with pytest.raises(ValueError):
        _validate_returns(data)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validate_returns_rejects_nonfinite_values(value):
    data = pd.DataFrame(
        {"AAPL": [0.01, value]},
        index=pd.date_range("2025-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        _validate_returns(data)


def test_validate_returns_rejects_returns_at_or_below_minus_one():
    data = pd.DataFrame(
        {"AAPL": [-1.0, 0.02]},
        index=pd.date_range("2025-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        _validate_returns(data)


# ---------------------------------------------------------------------
# Weight validation tests
# ---------------------------------------------------------------------


def test_validate_weights_aligns_to_asset_order():
    weights = pd.Series(
        {"MSFT": 0.5, "AAPL": 0.5}
    )

    result = _validate_weights(
        weights,
        ["AAPL", "MSFT"],
    )

    assert result.index.tolist() == ["AAPL", "MSFT"]
    assert result.tolist() == pytest.approx([0.5, 0.5])


def test_validate_weights_normalizes_small_tolerance_difference():
    weights = pd.Series(
        {"AAPL": 0.5000001, "MSFT": 0.5}
    )

    result = _validate_weights(
        weights,
        ["AAPL", "MSFT"],
    )

    assert result.sum() == pytest.approx(1.0)


def test_validate_weights_rejects_non_series():
    with pytest.raises((TypeError, ValueError)):
        _validate_weights(
            {"AAPL": 0.5, "MSFT": 0.5},
            ["AAPL", "MSFT"],
        )


def test_validate_weights_rejects_duplicate_labels():
    weights = pd.Series(
        [0.5, 0.5],
        index=["AAPL", "AAPL"],
    )

    with pytest.raises(ValueError):
        _validate_weights(
            weights,
            ["AAPL", "MSFT"],
        )


def test_validate_weights_rejects_missing_assets():
    weights = pd.Series({"AAPL": 1.0})

    with pytest.raises(ValueError):
        _validate_weights(
            weights,
            ["AAPL", "MSFT"],
        )


def test_validate_weights_rejects_nonfinite_values():
    weights = pd.Series(
        {"AAPL": np.nan, "MSFT": 1.0}
    )

    with pytest.raises(ValueError):
        _validate_weights(
            weights,
            ["AAPL", "MSFT"],
        )


def test_validate_weights_rejects_short_positions():
    weights = pd.Series(
        {"AAPL": 1.1, "MSFT": -0.1}
    )

    with pytest.raises(ValueError):
        _validate_weights(
            weights,
            ["AAPL", "MSFT"],
        )


def test_validate_weights_rejects_weights_not_summing_to_one():
    weights = pd.Series(
        {"AAPL": 0.3, "MSFT": 0.3}
    )

    with pytest.raises(ValueError):
        _validate_weights(
            weights,
            ["AAPL", "MSFT"],
        )


# ---------------------------------------------------------------------
# Rebalance date tests
# ---------------------------------------------------------------------


def test_rebalance_dates_excludes_final_dataset_date():
    dates = pd.to_datetime(
        [
            "2025-01-30",
            "2025-01-31",
            "2025-02-27",
            "2025-02-28",
        ]
    )

    result = _rebalance_dates(
        pd.DatetimeIndex(dates),
        "M",
    )

    assert result == {
        pd.Timestamp("2025-01-31"),
    }


def test_rebalance_dates_quarterly():
    dates = pd.to_datetime(
        [
            "2025-03-28",
            "2025-03-31",
            "2025-06-27",
            "2025-06-30",
            "2025-09-30",
            "2025-12-31",
        ]
    )

    result = _rebalance_dates(
        pd.DatetimeIndex(dates),
        "Q",
    )

    assert result == {
        pd.Timestamp("2025-03-31"),
        pd.Timestamp("2025-06-30"),
        pd.Timestamp("2025-09-30"),
    }


def test_rebalance_dates_annually():
    dates = pd.to_datetime(
        [
            "2024-12-31",
            "2025-06-30",
            "2025-12-31",
            "2026-06-30",
            "2026-12-31",
        ]
    )

    result = _rebalance_dates(
        pd.DatetimeIndex(dates),
        "Y",
    )

    assert result == {
        pd.Timestamp("2024-12-31"),
        pd.Timestamp("2025-12-31"),
    }


def test_rebalance_dates_returns_empty_for_single_period():
    dates = pd.to_datetime(
        [
            "2025-01-06",
            "2025-01-15",
            "2025-01-31",
        ]
    )

    result = _rebalance_dates(
        pd.DatetimeIndex(dates),
        "M",
    )

    assert result == set()


# ---------------------------------------------------------------------
# Backtest output tests
# ---------------------------------------------------------------------


def test_run_backtest_returns_expected_columns(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, _ = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    expected_columns = {
        "portfolio_value",
        "daily_return",
        "turnover",
        "trading_cost",
        "is_rebalance",
        "cumulative_return",
        "running_peak",
        "drawdown",
    }

    assert expected_columns.issubset(results.columns)


def test_run_backtest_preserves_input_dates(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, _ = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    expected_index = sample_returns.index.rename("date")

    pd.testing.assert_index_equal(
        results.index,
        expected_index,
        check_freq=False,
    )


def test_run_backtest_uses_initial_capital_without_costs(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, metrics = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    expected_value = (
        no_cost_config.initial_capital
        * (1 + results["daily_return"]).prod()
    )

    assert results["portfolio_value"].iloc[0] == pytest.approx(
        no_cost_config.initial_capital
        * (1 + results["daily_return"].iloc[0])
    )

    assert metrics["final_value"] == pytest.approx(expected_value)


def test_run_backtest_calculates_weighted_daily_return(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, _ = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    expected_returns = []
    current_weights = sample_weights.copy()

    for _, daily_returns in sample_returns.iterrows():
        portfolio_return = float(
            (current_weights * daily_returns).sum()
        )
        expected_returns.append(portfolio_return)

        updated_weights = (
            current_weights * (1.0 + daily_returns)
        )

        current_weights = (
            updated_weights / updated_weights.sum()
        )

    np.testing.assert_allclose(
        results["daily_return"].to_numpy(),
        np.asarray(expected_returns),
        rtol=1e-10,
        atol=1e-12,
    )


def test_run_backtest_applies_initial_transaction_cost(
    sample_returns,
    sample_weights,
):
    config = BacktestConfig(
        transaction_cost_bps=10,
        slippage_bps=0,
        charge_initial_transaction=True,
        rebalance_frequency="M",
    )

    results, metrics = run_backtest(
        sample_returns,
        sample_weights,
        config,
    )

    expected_cost = (
        config.initial_capital
        * 1.0
        * config.transaction_cost_bps
        / 10_000
    )

    assert results["trading_cost"].iloc[0] == pytest.approx(
        expected_cost
    )

    assert metrics["total_transaction_costs"] >= expected_cost


def test_run_backtest_can_disable_initial_transaction_cost(
    sample_returns,
    sample_weights,
):
    config = BacktestConfig(
        charge_initial_transaction=False,
    )

    results, metrics = run_backtest(
        sample_returns,
        sample_weights,
        config,
    )

    assert results["trading_cost"].iloc[0] == pytest.approx(0.0)
    assert metrics["total_transaction_costs"] == pytest.approx(0.0)


def test_run_backtest_allows_weights_to_drift_between_rebalances():
    dates = pd.bdate_range("2025-01-06", periods=3)

    returns = pd.DataFrame(
        {
            "AAPL": [0.10, 0.10, 0.00],
            "MSFT": [0.00, 0.00, 0.00],
        },
        index=dates,
    )

    weights = pd.Series(
        {"AAPL": 0.5, "MSFT": 0.5}
    )

    config = BacktestConfig(
        transaction_cost_bps=0,
        slippage_bps=0,
        charge_initial_transaction=False,
        rebalance_frequency="M",
    )

    results, _ = run_backtest(
        returns,
        weights,
        config,
    )

    assert results["daily_return"].iloc[0] == pytest.approx(0.05)
    assert results["daily_return"].iloc[1] > 0.05


def test_run_backtest_rebalances_at_month_end():
    dates = pd.to_datetime(
        [
            "2025-01-30",
            "2025-01-31",
            "2025-02-03",
            "2025-02-27",
            "2025-02-28",
            "2025-03-03",
        ]
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.01] * len(dates),
            "MSFT": [0.02] * len(dates),
        },
        index=dates,
    )

    weights = pd.Series(
        {"AAPL": 0.5, "MSFT": 0.5}
    )

    config = BacktestConfig(
        charge_initial_transaction=False,
        transaction_cost_bps=0,
        slippage_bps=0,
        rebalance_frequency="M",
    )

    results, _ = run_backtest(
        returns,
        weights,
        config,
    )

    rebalance_dates = results.index[
        results["is_rebalance"]
    ]

    assert pd.Timestamp("2025-01-31") in rebalance_dates
    assert pd.Timestamp("2025-02-28") in rebalance_dates


def test_run_backtest_does_not_rebalance_on_final_date(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, _ = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    assert not bool(results["is_rebalance"].iloc[-1])


def test_run_backtest_calculates_one_way_turnover_at_rebalance():
    dates = pd.to_datetime(
        [
            "2025-01-30",
            "2025-01-31",
            "2025-02-03",
            "2025-02-28",
        ]
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.0, 0.0, 0.0, 0.0],
            "MSFT": [0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )

    weights = pd.Series(
        {"AAPL": 0.5, "MSFT": 0.5}
    )

    config = BacktestConfig(
        charge_initial_transaction=False,
        transaction_cost_bps=0,
        slippage_bps=0,
        rebalance_frequency="M",
    )

    results, _ = run_backtest(
        returns,
        weights,
        config,
    )

    assert results.loc[
        pd.Timestamp("2025-01-31"),
        "turnover",
    ] == pytest.approx(0.0)


def test_run_backtest_charges_costs_on_rebalance_turnover():
    dates = pd.to_datetime(
        [
            "2025-01-30",
            "2025-01-31",
            "2025-02-03",
            "2025-02-28",
        ]
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.10, 0.00, 0.00, 0.00],
            "MSFT": [0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )

    weights = pd.Series(
        {"AAPL": 0.5, "MSFT": 0.5}
    )

    config = BacktestConfig(
        charge_initial_transaction=False,
        transaction_cost_bps=10,
        slippage_bps=0,
        rebalance_frequency="M",
    )

    results, _ = run_backtest(
        returns,
        weights,
        config,
    )

    rebalance_row = results.loc[
        pd.Timestamp("2025-01-31")
    ]

    pre_cost_value = (
        rebalance_row["portfolio_value"]
        + rebalance_row["trading_cost"]
    )

    expected_cost = (
        pre_cost_value
        * rebalance_row["turnover"]
        * config.transaction_cost_bps
        / 10_000
    )

    assert rebalance_row["trading_cost"] == pytest.approx(
        expected_cost,
        rel=1e-10,
        abs=1e-10,
    )


def test_run_backtest_calculates_cumulative_returns(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, _ = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    expected = (
        results["portfolio_value"]
        / no_cost_config.initial_capital
        - 1
    )

    np.testing.assert_allclose(
        results["cumulative_return"].to_numpy(),
        expected.to_numpy(),
    )


def test_run_backtest_includes_initial_capital_in_drawdown_peak(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    results, _ = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    assert results["running_peak"].iloc[0] >= (
        no_cost_config.initial_capital
    )


def test_run_backtest_suppresses_annualized_metrics_for_short_samples(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    _, metrics = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    assert np.isnan(metrics["annualized_return"])
    assert np.isnan(metrics["annualized_volatility"])


def test_run_backtest_reports_zero_sharpe_for_zero_volatility_as_nan():
    returns = pd.DataFrame(
        {
            "AAPL": [0.001] * 253,
            "MSFT": [0.001] * 253,
        },
        index=pd.bdate_range("2024-01-01", periods=253),
    )

    weights = pd.Series(
        {
            "AAPL": 0.5,
            "MSFT": 0.5,
        }
    )

    config = BacktestConfig(
        charge_initial_transaction=False,
    )

    _, metrics = run_backtest(
        returns,
        weights,
        config,
    )

    assert metrics["annualized_volatility"] == pytest.approx(0.0)
    assert np.isfinite(metrics["annualized_return"])
    assert np.isnan(metrics["sharpe_ratio_zero_rf"])


def test_run_backtest_calculates_annualized_metrics_for_sufficient_history():
    returns = pd.DataFrame(
        {
            "AAPL": np.linspace(-0.01, 0.01, 260),
            "MSFT": np.linspace(-0.005, 0.015, 260),
        },
        index=pd.bdate_range("2024-01-01", periods=260),
    )

    weights = pd.Series(
        {
            "AAPL": 0.5,
            "MSFT": 0.5,
        }
    )

    config = BacktestConfig(
        charge_initial_transaction=False,
    )

    _, metrics = run_backtest(
        returns,
        weights,
        config,
    )

    assert np.isfinite(metrics["annualized_return"])
    assert np.isfinite(metrics["annualized_volatility"])
    assert metrics["annualized_volatility"] > 0
    assert np.isfinite(metrics["sharpe_ratio_zero_rf"])


def test_run_backtest_reports_expected_metric_keys(
    sample_returns,
    sample_weights,
    no_cost_config,
):
    _, metrics = run_backtest(
        sample_returns,
        sample_weights,
        no_cost_config,
    )

    expected_keys = {
        "initial_capital",
        "final_value",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio_zero_rf",
        "max_drawdown",
        "total_transaction_costs",
        "total_turnover",
        "rebalance_count",
    }

    assert expected_keys.issubset(metrics.keys())


def test_run_backtest_rejects_invalid_returns(
    sample_weights,
    no_cost_config,
):
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, np.nan],
            "MSFT": [0.02, 0.03],
        },
        index=pd.date_range("2025-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        run_backtest(
            returns,
            sample_weights,
            no_cost_config,
        )


def test_run_backtest_rejects_invalid_target_weights(
    sample_returns,
    no_cost_config,
):
    weights = pd.Series(
        {"AAPL": 0.8, "MSFT": 0.8}
    )

    with pytest.raises(ValueError):
        run_backtest(
            sample_returns,
            weights,
            no_cost_config,
        )


def test_run_backtest_rejects_missing_target_asset(
    sample_returns,
    no_cost_config,
):
    weights = pd.Series(
        {"AAPL": 1.0}
    )

    with pytest.raises(ValueError):
        run_backtest(
            sample_returns,
            weights,
            no_cost_config,
        )


def test_run_backtest_uses_default_config_when_none_is_supplied(
    sample_returns,
    sample_weights,
):
    results, metrics = run_backtest(
        sample_returns,
        sample_weights,
        config=None,
    )

    assert len(results) == len(sample_returns)
    assert metrics["initial_capital"] == 100_000.0
