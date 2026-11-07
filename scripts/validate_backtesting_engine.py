
"""Controlled validation tests for the portfolio backtesting engine."""

import numpy as np
import pandas as pd

from quantfin.backtesting import BacktestConfig, run_backtest


def make_test_returns() -> pd.DataFrame:
    """Create deterministic returns spanning two calendar months."""
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-02-01",
        ]
    )

    return pd.DataFrame(
        {
            "ASSET_A": [0.10, 0.10, 0.00, 0.00],
            "ASSET_B": [0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )


def check(condition: bool, description: str) -> None:
    """Print a validation result and stop if it fails."""
    if not condition:
        raise AssertionError(f"FAILED: {description}")

    print(f"PASS | {description}")


def main() -> None:
    print("=" * 72)
    print("BACKTESTING ENGINE — CONTROLLED VALIDATION")
    print("=" * 72)

    returns = make_test_returns()

    weights = pd.Series(
        {"ASSET_A": 0.50, "ASSET_B": 0.50}
    )

    config = BacktestConfig(
        initial_capital=10_000,
        transaction_cost_bps=0,
        slippage_bps=0,
        rebalance_frequency="M",
        charge_initial_transaction=False,
    )

    results, metrics = run_backtest(
        returns=returns,
        target_weights=weights,
        config=config,
    )

    # Test 1: Basic execution.
    check(
        len(results) == 4,
        "Four daily observations processed",
    )

    # Day 1: 50% * 10% + 50% * 0% = 5%.
    check(
        np.isclose(results["daily_return"].iloc[0], 0.05),
        "First-day portfolio return equals 5%",
    )

    # After day 1, asset A weight drifts to:
    # 0.50 * 1.10 / (0.50 * 1.10 + 0.50)
    drifted_a_weight = 0.55 / 1.05

    expected_second_return = drifted_a_weight * 0.10

    check(
        np.isclose(
            results["daily_return"].iloc[1],
            expected_second_return,
        ),
        "Second-day return correctly uses drifted weights",
    )

    # Test 2: January month-end rebalance.
    # After two 10% returns in asset A, its weight is:
    final_drifted_a_weight = (
        0.50 * (1.10**2)
    ) / (
        0.50 * (1.10**2) + 0.50
    )

    expected_turnover = abs(final_drifted_a_weight - 0.50)

    check(
        np.isclose(
            results["turnover"].iloc[2],
            expected_turnover,
        ),
        "January month-end turnover is calculated correctly",
    )

    check(
        bool(results["is_rebalance"].iloc[2]),
        "January month-end rebalance is recorded",
    )

    check(
        not bool(results["is_rebalance"].iloc[3]),
        "Final dataset date is not unnecessarily rebalanced",
    )

    check(
        int(metrics["rebalance_count"]) == 1,
        "Exactly one monthly rebalance is recorded",
    )

    # Test 3: Initial transaction costs and rebalancing costs.
    cost_config = BacktestConfig(
        initial_capital=10_000,
        transaction_cost_bps=10,
        slippage_bps=0,
        rebalance_frequency="M",
        charge_initial_transaction=True,
    )

    cost_results, cost_metrics = run_backtest(
        returns=returns,
        target_weights=weights,
        config=cost_config,
    )

    # Initial one-way turnover = 100%.
    # 10 bps = 0.001, initial cost = $10.
    check(
        np.isclose(
            cost_results["trading_cost"].iloc[0],
            10.0,
        ),
        "Initial transaction cost equals $10",
    )

    expected_first_value = (10_000 - 10) * 1.05

    check(
        np.isclose(
            cost_results["portfolio_value"].iloc[0],
            expected_first_value,
        ),
        "Initial trading cost is deducted before investment return",
    )

    check(
        cost_metrics["total_transaction_costs"] > 10.0,
        "Subsequent rebalancing costs are included",
    )

    check(
        cost_results["trading_cost"].iloc[2] > 0,
        "Transaction cost is charged at January month-end",
    )

    check(
        np.isclose(cost_results["trading_cost"].iloc[3], 0.0),
        "No transaction cost is charged on the final date",
    )

    # Test 4: Portfolio accounting and drawdown.
    check(
        np.isfinite(results["portfolio_value"]).all(),
        "Portfolio values remain finite",
    )

    check(
        (results["portfolio_value"] > 0).all(),
        "Portfolio values remain positive",
    )

    check(
        (results["drawdown"] <= 1e-10).all(),
        "Drawdowns never exceed zero",
    )

    check(
        np.isclose(
            results["cumulative_return"].iloc[-1],
            results["portfolio_value"].iloc[-1] / 10_000 - 1,
        ),
        "Cumulative return reconciles with portfolio value",
    )

    # Test 5: Initial capital is included in the drawdown peak.
    loss_returns = pd.DataFrame(
        {"ASSET_A": [-0.10], "ASSET_B": [-0.10]},
        index=pd.to_datetime(["2024-01-02"]),
    )

    loss_results, _ = run_backtest(
        returns=loss_returns,
        target_weights=weights,
        config=config,
    )

    check(
        np.isclose(loss_results["drawdown"].iloc[0], -0.10),
        "First-day loss is correctly included in maximum drawdown",
    )

    # Test 6: Short samples must not produce misleading annualized metrics.
    check(
        np.isnan(metrics["annualized_return"]),
        "Annualized return suppressed for short samples",
    )

    check(
        np.isnan(metrics["annualized_volatility"]),
        "Annualized volatility suppressed for short samples",
    )

    check(
        np.isnan(metrics["sharpe_ratio_zero_rf"]),
        "Sharpe ratio suppressed for short samples",
    )

    # Test 7: Invalid weights must be rejected.
    invalid_weights = pd.Series(
        {"ASSET_A": 0.60, "ASSET_B": 0.30}
    )

    rejected = False

    try:
        run_backtest(
            returns=returns,
            target_weights=invalid_weights,
            config=config,
        )
    except ValueError:
        rejected = True

    check(
        rejected,
        "Invalid portfolio weights are rejected",
    )

    print("\nFINAL PORTFOLIO METRICS")
    print("-" * 72)

    for name, value in metrics.items():
        print(f"{name:<32} {value:,.6f}")

    print("\nALL CONTROLLED BACKTESTING TESTS PASSED.")


if __name__ == "__main__":
    main()
