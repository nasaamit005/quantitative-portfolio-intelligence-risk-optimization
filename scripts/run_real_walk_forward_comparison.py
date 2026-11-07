
"""Run a real-data walk-forward comparison of portfolio strategies."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.backtesting import (
    BacktestConfig,
    run_walk_forward_backtest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "us_equities_simple_returns.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "walk_forward_comparison"

LOOKBACK_DAYS = 252
RISK_FREE_RATE = 0.04
MAX_WEIGHT = 0.50

STRATEGIES = [
    "equal_weight",
    "minimum_volatility",
    "maximum_sharpe",
    "hrp",
    "risk_parity",
]


def load_returns() -> pd.DataFrame:
    """Load and validate the processed equity returns dataset."""

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns dataset not found: {RETURNS_FILE}"
        )

    returns = pd.read_csv(
        RETURNS_FILE,
        index_col=0,
        parse_dates=True,
    )

    if returns.empty:
        raise ValueError("The returns dataset is empty.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("The returns index must contain valid dates.")

    if returns.index.isna().any():
        raise ValueError("The returns dataset contains invalid dates.")

    returns = returns.sort_index()

    if returns.index.has_duplicates:
        raise ValueError("The returns dataset contains duplicate dates.")

    if returns.columns.has_duplicates:
        raise ValueError("The returns dataset contains duplicate assets.")

    # Convert all asset columns to numeric values.
    returns = returns.apply(pd.to_numeric, errors="coerce")

    # Remove only the first row when every asset return is missing.
    # This is expected when no previous price exists to calculate returns.
    if returns.iloc[0].isna().all():
        print(
            "Removing initial return-calculation row: "
            f"{returns.index[0].date()}"
        )
        returns = returns.iloc[1:].copy()

    if returns.empty:
        raise ValueError(
            "No observations remain after initial-row processing."
        )

    # Reject all remaining missing or non-numeric observations.
    if returns.isna().any().any():
        missing_by_asset = returns.isna().sum()
        missing_by_asset = missing_by_asset[missing_by_asset > 0]

        raise ValueError(
            "Missing or non-numeric observations remain:\n"
            f"{missing_by_asset.to_string()}"
        )

    # Reject infinite values.
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("The returns dataset contains infinite values.")

    # Simple returns cannot be less than or equal to -100%.
    if (returns <= -1.0).any().any():
        raise ValueError("Returns must be greater than -100%.")

    if len(returns) <= LOOKBACK_DAYS:
        raise ValueError(
            f"At least {LOOKBACK_DAYS + 1} observations are required."
        )

    if len(returns.columns) < 2:
        raise ValueError("At least two assets are required.")

    print("\nDATA QUALITY VALIDATION")
    print("-" * 80)
    print("Missing values:       0")
    print("Infinite values:      0")
    print("Duplicate dates:      0")
    print("Duplicate assets:     0")
    print("Return validation:    PASSED")

    return returns


def save_results(
    summary: pd.DataFrame,
    daily_results: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
) -> None:
    """Save comparison results in CSV and Parquet formats."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUTPUT_DIR / "strategy_comparison_summary.csv",
        index=False,
    )
    summary.to_parquet(
        OUTPUT_DIR / "strategy_comparison_summary.parquet",
        index=False,
    )

    daily_results.to_csv(
        OUTPUT_DIR / "strategy_daily_results.csv",
    )
    daily_results.to_parquet(
        OUTPUT_DIR / "strategy_daily_results.parquet",
    )

    portfolio_weights.to_csv(
        OUTPUT_DIR / "strategy_daily_weights.csv",
    )
    portfolio_weights.to_parquet(
        OUTPUT_DIR / "strategy_daily_weights.parquet",
    )


def main() -> None:
    """Execute and save the real-market strategy comparison."""

    print("=" * 80)
    print("REAL-MARKET WALK-FORWARD STRATEGY COMPARISON")
    print("=" * 80)

    returns = load_returns()

    print("\nDATASET INFORMATION")
    print("-" * 80)
    print(f"Observations:       {len(returns):,}")
    print(f"Assets:             {len(returns.columns)}")
    print(f"First date:         {returns.index.min().date()}")
    print(f"Last date:          {returns.index.max().date()}")
    print(f"Estimation window:  {LOOKBACK_DAYS} trading observations")
    print(f"Test observations:  {len(returns) - LOOKBACK_DAYS:,}")

    config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=5,
        slippage_bps=2,
        rebalance_frequency="M",
        charge_initial_transaction=True,
    )

    print("\nBACKTEST CONFIGURATION")
    print("-" * 80)
    print(f"Initial capital:    ${config.initial_capital:,.2f}")
    print(f"Transaction costs:  {config.transaction_cost_bps} bps")
    print(f"Slippage:           {config.slippage_bps} bps")
    print(f"Rebalancing:        {config.rebalance_frequency}")
    print(f"Risk-free rate:     {RISK_FREE_RATE:.2%}")
    print(f"Maximum weight:     {MAX_WEIGHT:.0%}")

    all_metrics: list[dict[str, object]] = []
    all_daily_results: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []

    common_test_index: pd.Index | None = None

    for strategy in STRATEGIES:
        print(f"\nRunning strategy: {strategy}")
        print("-" * 80)

        results, metrics = run_walk_forward_backtest(
            returns=returns,
            strategy=strategy,
            lookback_days=LOOKBACK_DAYS,
            config=config,
            risk_free_rate=RISK_FREE_RATE,
            max_weight=MAX_WEIGHT,
        )

        if results.empty:
            raise ValueError(
                f"Strategy '{strategy}' produced no test observations."
            )

        # Ensure every strategy uses identical out-of-sample dates.
        if common_test_index is None:
            common_test_index = results.index
        elif not results.index.equals(common_test_index):
            raise AssertionError(
                "Strategies do not share the same out-of-sample dates."
            )

        metrics["first_test_date"] = results.index[0].date().isoformat()
        metrics["last_test_date"] = results.index[-1].date().isoformat()

        all_metrics.append(metrics)

        # Store daily portfolio performance.
        daily = results.copy()
        daily.insert(0, "strategy", strategy)
        all_daily_results.append(daily)

        # Store daily portfolio weights.
        weight_columns = [
            column
            for column in results.columns
            if column.startswith("weight_")
        ]

        weights = results[weight_columns].copy()
        weights.insert(0, "strategy", strategy)
        all_weights.append(weights)

        print(f"Final value:          ${metrics['final_value']:,.2f}")
        print(f"Total return:          {metrics['total_return']:.4%}")
        print(
            f"Annualized return:     "
            f"{metrics['annualized_return']:.4%}"
        )
        print(
            f"Annualized volatility: "
            f"{metrics['annualized_volatility']:.4%}"
        )
        print(f"Sharpe ratio:          {metrics['sharpe_ratio']:.4f}")
        print(f"Maximum drawdown:      {metrics['max_drawdown']:.4%}")
        print(
            "Transaction costs:    "
            f"${metrics['total_transaction_costs']:,.2f}"
        )
        print(f"Total turnover:        {metrics['total_turnover']:.4f}")
        print(f"Rebalances:            {metrics['rebalance_count']:.0f}")

    # Build the strategy comparison summary.
    summary = pd.DataFrame(all_metrics)

    # Rank strategies by out-of-sample Sharpe ratio.
    summary = summary.sort_values(
        "sharpe_ratio",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    summary.insert(
        0,
        "sharpe_rank",
        range(1, len(summary) + 1),
    )

    daily_results = pd.concat(
        all_daily_results,
        axis=0,
    )

    portfolio_weights = pd.concat(
        all_weights,
        axis=0,
    )

    # Save all reproducible outputs.
    save_results(
        summary=summary,
        daily_results=daily_results,
        portfolio_weights=portfolio_weights,
    )

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE STRATEGY COMPARISON")
    print("=" * 80)

    display_columns = [
        "sharpe_rank",
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "total_transaction_costs",
        "rebalance_count",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "total_return": "{:.2%}".format,
                "annualized_return": "{:.2%}".format,
                "annualized_volatility": "{:.2%}".format,
                "sharpe_ratio": "{:.4f}".format,
                "max_drawdown": "{:.2%}".format,
                "total_transaction_costs": "${:,.2f}".format,
                "rebalance_count": "{:.0f}".format,
            },
        )
    )

    print("\nOUTPUT FILES")
    print("-" * 80)

    for file_path in sorted(OUTPUT_DIR.iterdir()):
        if file_path.is_file():
            print(file_path)

    print("\nREAL-MARKET WALK-FORWARD COMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
