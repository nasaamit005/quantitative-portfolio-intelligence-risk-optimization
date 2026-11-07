
"""Unit tests for walk-forward portfolio backtesting."""

import numpy as np
import pandas as pd
import pytest

import quantfin.backtesting.walk_forward as walk_forward
from quantfin.backtesting.engine import BacktestConfig


def make_returns(
    observations: int = 8,
    asset_a_return: float = 0.01,
    asset_b_return: float = 0.02,
) -> pd.DataFrame:
    """Create deterministic daily returns for two assets."""
    dates = pd.bdate_range("2024-01-02", periods=observations)

    return pd.DataFrame(
        {
            "A": asset_a_return,
            "B": asset_b_return,
        },
        index=dates,
    )


def make_config(
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    charge_initial_transaction: bool = False,
    rebalance_frequency: str = "M",
) -> BacktestConfig:
    """Create a predictable backtesting configuration."""
    return BacktestConfig(
        initial_capital=1_000.0,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        rebalance_frequency=rebalance_frequency,
        charge_initial_transaction=charge_initial_transaction,
    )


def install_estimator(monkeypatch, estimator):
    """Replace strategy selection with a controlled estimator."""
    monkeypatch.setattr(
        walk_forward,
        "_get_strategy",
        lambda **kwargs: estimator,
    )


def test_walk_forward_returns_expected_columns(monkeypatch):
    """The result should contain portfolio and weight information."""
    returns = make_returns()

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    results, metrics = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    expected_columns = {
        "portfolio_value",
        "gross_daily_return",
        "net_daily_return",
        "turnover",
        "trading_cost",
        "is_rebalance",
        "weight_A",
        "weight_B",
        "cumulative_return",
        "running_peak",
        "drawdown",
    }

    assert expected_columns.issubset(results.columns)
    assert isinstance(metrics, dict)
    assert metrics["strategy"] == "equal_weight"


def test_walk_forward_starts_after_initial_lookback(monkeypatch):
    """The first out-of-sample observation follows the lookback window."""
    returns = make_returns(observations=8)
    observed_windows = []

    def estimator(history):
        observed_windows.append(history.index.copy())
        return pd.Series({"A": 0.5, "B": 0.5})

    install_estimator(monkeypatch, estimator)

    results, _ = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    assert results.index[0] == returns.index[3]
    assert observed_windows[0].equals(returns.index[:3])


def test_walk_forward_uses_only_prior_data_at_rebalance(monkeypatch):
    """The rebalance estimator must exclude the current date."""
    returns = make_returns(observations=8)
    rebalance_date = returns.index[4]
    observed_windows = []

    def estimator(history):
        observed_windows.append(history.index.copy())
        return pd.Series({"A": 0.5, "B": 0.5})

    install_estimator(monkeypatch, estimator)

    monkeypatch.setattr(
        walk_forward,
        "_rebalance_dates",
        lambda index, frequency: {rebalance_date},
    )

    walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    assert len(observed_windows) == 2
    assert observed_windows[0].equals(returns.index[:3])
    assert observed_windows[1].equals(returns.index[1:4])
    assert rebalance_date not in observed_windows[1]


def test_walk_forward_new_weights_apply_next_session(monkeypatch):
    """Closing rebalance weights must not earn the same day's return."""
    returns = make_returns(
        observations=7,
        asset_a_return=0.10,
        asset_b_return=0.00,
    )

    rebalance_date = returns.index[4]
    calls = 0

    def estimator(history):
        nonlocal calls
        calls += 1

        if calls == 1:
            return pd.Series({"A": 1.0, "B": 0.0})

        return pd.Series({"A": 0.0, "B": 1.0})

    install_estimator(monkeypatch, estimator)

    monkeypatch.setattr(
        walk_forward,
        "_rebalance_dates",
        lambda index, frequency: {rebalance_date},
    )

    results, _ = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    assert results.loc[rebalance_date, "gross_daily_return"] == pytest.approx(
        0.10
    )

    next_date = returns.index[5]

    assert results.loc[next_date, "gross_daily_return"] == pytest.approx(
        0.00
    )

    assert results.loc[rebalance_date, "weight_A"] == pytest.approx(0.0)
    assert results.loc[rebalance_date, "weight_B"] == pytest.approx(1.0)


def test_walk_forward_does_not_rebalance_final_date(monkeypatch):
    """The actual rebalance helper excludes the final dataset date."""
    returns = make_returns(observations=8)
    calls = 0

    def estimator(history):
        nonlocal calls
        calls += 1
        return pd.Series({"A": 0.5, "B": 0.5})

    install_estimator(monkeypatch, estimator)

    results, metrics = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    actual_rebalance_dates = walk_forward._rebalance_dates(
        returns.index,
        "M",
    )

    assert returns.index[-1] not in actual_rebalance_dates
    assert not results["is_rebalance"].iloc[-1]
    assert calls == 1
    assert metrics["rebalance_count"] == 0.0


def test_walk_forward_charges_initial_transaction_cost(monkeypatch):
    """Initial implementation costs are charged before daily returns."""
    returns = make_returns(observations=7)

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    config = make_config(
        transaction_cost_bps=10,
        slippage_bps=5,
        charge_initial_transaction=True,
    )

    results, metrics = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=config,
    )

    expected_cost = 1_000.0 * 1.0 * (15 / 10_000)
    expected_gross_return = 0.5 * 0.01 + 0.5 * 0.02
    expected_first_value = (
        1_000.0 - expected_cost
    ) * (1.0 + expected_gross_return)

    assert results["trading_cost"].iloc[0] == pytest.approx(
        expected_cost
    )
    assert results["portfolio_value"].iloc[0] == pytest.approx(
        expected_first_value
    )
    assert metrics["total_transaction_costs"] >= expected_cost


def test_walk_forward_does_not_charge_initial_cost_when_disabled(
    monkeypatch,
):
    """Initial implementation costs can be disabled."""
    returns = make_returns(observations=7)

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    results, _ = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(
            transaction_cost_bps=10,
            slippage_bps=5,
            charge_initial_transaction=False,
        ),
    )

    assert results["trading_cost"].iloc[0] == pytest.approx(0.0)
    assert results["portfolio_value"].iloc[0] > 1_000.0


def test_walk_forward_charges_rebalance_transaction_cost(monkeypatch):
    """Rebalance costs should be based on one-way portfolio turnover."""
    returns = make_returns(
        observations=7,
        asset_a_return=0.0,
        asset_b_return=0.0,
    )

    rebalance_date = returns.index[4]
    calls = 0

    def estimator(history):
        nonlocal calls
        calls += 1

        if calls == 1:
            return pd.Series({"A": 1.0, "B": 0.0})

        return pd.Series({"A": 0.0, "B": 1.0})

    install_estimator(monkeypatch, estimator)

    monkeypatch.setattr(
        walk_forward,
        "_rebalance_dates",
        lambda index, frequency: {rebalance_date},
    )

    config = make_config(
        transaction_cost_bps=10,
        slippage_bps=5,
    )

    results, _ = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=config,
    )

    row = results.loc[rebalance_date]

    expected_cost = 1_000.0 * (15 / 10_000)

    assert row["turnover"] == pytest.approx(1.0)
    assert row["trading_cost"] == pytest.approx(expected_cost)
    assert row["is_rebalance"]


def test_walk_forward_reports_net_returns_after_costs(monkeypatch):
    """Net returns should reflect transaction costs."""
    returns = make_returns(
        observations=7,
        asset_a_return=0.0,
        asset_b_return=0.0,
    )

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    config = make_config(
        transaction_cost_bps=10,
        slippage_bps=5,
        charge_initial_transaction=True,
    )

    results, _ = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=config,
    )

    assert results["net_daily_return"].iloc[0] < 0.0
    assert results["trading_cost"].iloc[0] > 0.0


def test_walk_forward_metrics_match_results(monkeypatch):
    """Summary metrics should reconcile with the result dataframe."""
    returns = make_returns(observations=270)

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    results, metrics = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    expected_total_return = (
        results["portfolio_value"].iloc[-1] / 1_000.0
    ) - 1.0

    assert metrics["test_observations"] == len(results)
    assert metrics["total_return"] == pytest.approx(expected_total_return)
    assert metrics["final_value"] == pytest.approx(
        results["portfolio_value"].iloc[-1]
    )
    assert metrics["total_transaction_costs"] == pytest.approx(
        results["trading_cost"].sum()
    )
    assert metrics["total_turnover"] == pytest.approx(
        results["turnover"].sum()
    )


def test_walk_forward_annualizes_only_with_252_observations(monkeypatch):
    """Short out-of-sample samples should not receive annualized metrics."""
    returns = make_returns(observations=20)

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    _, metrics = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    assert np.isnan(metrics["annualized_return"])
    assert np.isnan(metrics["annualized_volatility"])
    assert np.isnan(metrics["sharpe_ratio"])


def test_walk_forward_handles_zero_return_volatility(monkeypatch):
    """A zero-volatility return series should have an undefined Sharpe."""
    returns = make_returns(
        observations=270,
        asset_a_return=0.0,
        asset_b_return=0.0,
    )

    install_estimator(
        monkeypatch,
        lambda data: pd.Series({"A": 0.5, "B": 0.5}),
    )

    _, metrics = walk_forward.run_walk_forward_backtest(
        returns,
        lookback_days=3,
        config=make_config(),
    )

    assert metrics["annualized_volatility"] == pytest.approx(0.0)
    assert np.isnan(metrics["sharpe_ratio"])


@pytest.mark.parametrize("lookback_days", [0, 1, -1, 2.5])
def test_walk_forward_rejects_invalid_lookback(lookback_days):
    """Lookback must be an integer of at least two."""
    returns = make_returns()

    with pytest.raises(ValueError, match="lookback_days"):
        walk_forward.run_walk_forward_backtest(
            returns,
            lookback_days=lookback_days,
        )


def test_walk_forward_rejects_insufficient_observations():
    """The dataset must contain more rows than the lookback."""
    returns = make_returns(observations=3)

    with pytest.raises(ValueError, match="more observations"):
        walk_forward.run_walk_forward_backtest(
            returns,
            lookback_days=3,
        )


@pytest.mark.parametrize("risk_free_rate", [np.nan, np.inf, -np.inf])
def test_walk_forward_rejects_nonfinite_risk_free_rate(risk_free_rate):
    """Risk-free rate must be finite."""
    returns = make_returns()

    with pytest.raises(ValueError, match="risk_free_rate"):
        walk_forward.run_walk_forward_backtest(
            returns,
            lookback_days=3,
            risk_free_rate=risk_free_rate,
        )


@pytest.mark.parametrize("max_weight", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_walk_forward_rejects_invalid_max_weight(max_weight):
    """Maximum asset weight must be finite and in (0, 1]."""
    returns = make_returns()

    with pytest.raises(ValueError, match="max_weight"):
        walk_forward.run_walk_forward_backtest(
            returns,
            lookback_days=3,
            max_weight=max_weight,
        )


def test_walk_forward_rejects_unknown_strategy():
    """Unsupported strategy names should produce a clear error."""
    returns = make_returns()

    with pytest.raises(ValueError, match="Unknown strategy"):
        walk_forward.run_walk_forward_backtest(
            returns,
            strategy="unsupported_strategy",
            lookback_days=3,
        )


def test_get_strategy_rejects_unknown_strategy():
    """The strategy selector should list supported strategy names."""
    with pytest.raises(ValueError, match="Available strategies"):
        walk_forward._get_strategy(
            strategy="not_a_strategy",
            risk_free_rate=0.04,
            max_weight=0.50,
        )


@pytest.mark.parametrize(
    ("strategy", "expected_name"),
    [
        ("equal_weight", "equal_weight"),
        ("minimum_volatility", "minimum_volatility"),
        ("maximum_sharpe", "maximum_sharpe"),
        ("hrp", "hrp"),
        ("risk_parity", "risk_parity"),
    ],
)
def test_get_strategy_returns_callable(strategy, expected_name):
    """All supported strategy names should return callable estimators."""
    estimator = walk_forward._get_strategy(
        strategy=strategy,
        risk_free_rate=0.04,
        max_weight=0.50,
    )

    assert callable(estimator)
    assert expected_name in strategy


def test_get_strategy_passes_maximum_sharpe_parameters(monkeypatch):
    """Maximum-Sharpe strategy should receive the supplied parameters."""
    captured = {}

    def fake_maximum_sharpe(data, risk_free_rate, max_weight):
        captured["risk_free_rate"] = risk_free_rate
        captured["max_weight"] = max_weight
        return pd.Series({"A": 0.5, "B": 0.5})

    monkeypatch.setattr(
        walk_forward,
        "maximum_sharpe_strategy",
        fake_maximum_sharpe,
    )

    estimator = walk_forward._get_strategy(
        strategy="maximum_sharpe",
        risk_free_rate=0.03,
        max_weight=0.60,
    )

    result = estimator(make_returns())

    assert captured["risk_free_rate"] == 0.03
    assert captured["max_weight"] == 0.60
    assert result.sum() == pytest.approx(1.0)
