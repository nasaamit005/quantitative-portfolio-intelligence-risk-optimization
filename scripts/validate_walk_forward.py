
"""Controlled validation of the walk-forward backtesting engine."""

import numpy as np
import pandas as pd

from quantfin.backtesting import (
    BacktestConfig,
    run_walk_forward_backtest,
)


def check(condition: bool, description: str) -> None:
    """Print a validation result and stop if it fails."""
    if not condition:
        raise AssertionError(f"FAILED: {description}")

    print(f"PASS | {description}")


def make_test_returns() -> pd.DataFrame:
    """Create deterministic synthetic daily returns."""
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        start="2020-01-02",
        periods=320,
    )

    asset_names = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"]

    daily_returns = rng.normal(
        loc=0.0004,
        scale=0.01,
        size=(len(dates), len(asset_names)),
    )

    return pd.DataFrame(
        daily_returns,
        index=dates,
        columns=asset_names,
    )


def main() -> None:
    print("=" * 72)
    print("WALK-FORWARD BACKTESTING — CONTROLLED VALIDATION")
    print("=" * 72)

    returns = make_test_returns()
    lookback_days = 252

    config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=5,
        slippage_bps=2,
        rebalance_frequency="M",
        charge_initial_transaction=True,
    )

    # Test 1: Execute the walk-forward backtest.
    results, metrics = run_walk_forward_backtest(
        returns=returns,
        strategy="equal_weight",
        lookback_days=lookback_days,
        config=config,
    )

    expected_test_observations = len(returns) - lookback_days

    check(
        len(results) == expected_test_observations,
        "Backtest begins after the historical lookback window",
    )

    check(
        results.index[0] == returns.index[lookback_days],
        "First test date follows the estimation window",
    )

    check(
        results.index[-1] == returns.index[-1],
        "Backtest includes the final available observation",
    )

    # Test 2: Portfolio weight calculations.
    weight_columns = [
        column
        for column in results.columns
        if column.startswith("weight_")
    ]

    check(
        len(weight_columns) == returns.shape[1],
        "Portfolio weights are recorded for every asset",
    )

    weight_sums = results[weight_columns].sum(axis=1)

    check(
        np.allclose(weight_sums, 1.0, atol=1e-6),
        "Portfolio weights sum to one on every test date",
    )

    check(
        (results[weight_columns] >= -1e-10).all().all(),
        "Portfolio weights remain long-only",
    )

    # Test 3: Monthly rebalancing.
    check(
        results["is_rebalance"].any(),
        "At least one scheduled monthly rebalance occurs",
    )

    check(
        not bool(results["is_rebalance"].iloc[-1]),
        "Final dataset date is not unnecessarily rebalanced",
    )

    check(
        int(results["is_rebalance"].sum())
        == int(metrics["rebalance_count"]),
        "Rebalance count reconciles with the results",
    )

    # Test 4: Initial portfolio return.
    expected_first_gross_return = float(
        returns.iloc[lookback_days].mean()
    )

    check(
        np.isclose(
            results["gross_daily_return"].iloc[0],
            expected_first_gross_return,
        ),
        "Initial portfolio return uses equal-weight allocation",
    )

    # Test 5: Initial transaction costs.
    initial_cost_rate = (
        config.transaction_cost_bps + config.slippage_bps
    ) / 10_000.0

    expected_initial_cost = (
        config.initial_capital * initial_cost_rate
    )

    check(
        np.isclose(
            results["trading_cost"].iloc[0],
            expected_initial_cost,
        ),
        "Initial transaction costs are calculated correctly",
    )

    check(
        results["trading_cost"].sum() >= expected_initial_cost,
        "Total transaction costs include initial implementation",
    )

    # Test 6: Rebalancing turnover and trading costs.
    rebalance_rows = results.loc[results["is_rebalance"]]

    check(
        (rebalance_rows["turnover"] >= 0).all(),
        "Rebalancing turnover is non-negative",
    )

    check(
        (rebalance_rows["trading_cost"] >= 0).all(),
        "Rebalancing trading costs are non-negative",
    )

    # Test 7: Portfolio accounting.
    check(
        np.isfinite(results["portfolio_value"]).all(),
        "Portfolio values remain finite",
    )

    check(
        (results["portfolio_value"] > 0).all(),
        "Portfolio values remain positive",
    )

    check(
        np.isfinite(results["net_daily_return"]).all(),
        "Net daily returns remain finite",
    )

    check(
        np.allclose(
            results["cumulative_return"],
            results["portfolio_value"] / config.initial_capital - 1.0,
        ),
        "Cumulative returns reconcile with portfolio values",
    )

    check(
        (results["drawdown"] <= 1e-10).all(),
        "Drawdowns never exceed zero",
    )

    # Test 8: Short test periods must not produce misleading annualization.
    check(
        np.isnan(metrics["annualized_return"]),
        "Annualized return suppressed for short test periods",
    )

    check(
        np.isnan(metrics["annualized_volatility"]),
        "Annualized volatility suppressed for short test periods",
    )

    check(
        np.isnan(metrics["sharpe_ratio"]),
        "Sharpe ratio suppressed for short test periods",
    )

    # Test 9: Reject invalid strategy names.
    rejected = False

    try:
        run_walk_forward_backtest(
            returns=returns,
            strategy="unknown_strategy",
            lookback_days=lookback_days,
            config=config,
        )
    except ValueError:
        rejected = True

    check(
        rejected,
        "Unknown strategy names are rejected",
    )

    # Test 10: Reject an insufficient historical window.
    rejected = False

    try:
        run_walk_forward_backtest(
            returns=returns.iloc[:200],
            strategy="equal_weight",
            lookback_days=252,
            config=config,
        )
    except ValueError:
        rejected = True

    check(
        rejected,
        "Insufficient historical observations are rejected",
    )

    print("\nWALK-FORWARD PERFORMANCE SUMMARY")
    print("-" * 72)

    for name, value in metrics.items():
        print(f"{name:<32} {value}")

    print("\nALL CONTROLLED WALK-FORWARD TESTS PASSED.")


if __name__ == "__main__":
    main()
