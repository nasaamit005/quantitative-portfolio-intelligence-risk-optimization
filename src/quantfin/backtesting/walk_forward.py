
"""Walk-forward portfolio backtesting with historical weight estimation."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantfin.backtesting.engine import (
    BacktestConfig,
    _rebalance_dates,
    _validate_returns,
    _validate_weights,
)
from quantfin.strategies import (
    equal_weight_weights,
    hrp_strategy,
    maximum_sharpe_strategy,
    minimum_volatility_strategy,
    risk_parity_strategy,
)

StrategyFunction = Callable[[pd.DataFrame], pd.Series]


def _get_strategy(
    strategy: str,
    risk_free_rate: float,
    max_weight: float,
) -> StrategyFunction:
    """Return the selected portfolio-weight estimation function."""

    strategies: dict[str, StrategyFunction] = {
        "equal_weight": lambda data: equal_weight_weights(data),
        "minimum_volatility": lambda data: minimum_volatility_strategy(
            data,
            max_weight=max_weight,
        ),
        "maximum_sharpe": lambda data: maximum_sharpe_strategy(
            data,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        ),
        "hrp": lambda data: hrp_strategy(data),
        "risk_parity": lambda data: risk_parity_strategy(
            data,
            max_weight=max_weight,
        ),
    }

    if strategy not in strategies:
        available = ", ".join(strategies)
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available strategies: {available}"
        )

    return strategies[strategy]


def run_walk_forward_backtest(
    returns: pd.DataFrame,
    strategy: str = "equal_weight",
    lookback_days: int = 252,
    config: BacktestConfig | None = None,
    risk_free_rate: float = 0.04,
    max_weight: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a walk-forward backtest using historical information.

    The initial portfolio is estimated using observations immediately
    preceding the out-of-sample test period.

    At each scheduled rebalance:
    - The current day's return is earned using existing portfolio weights.
    - Portfolio weights drift naturally with asset returns.
    - New weights are estimated using observations strictly before
      the current rebalance date.
    - Trading costs are charged at the current date's close.
    - New weights apply from the next trading session.

    This uses a conservative information-timing assumption. It does not
    model intraday execution, bid-ask spreads, or market impact separately.

    The final dataset date is not rebalanced because there is no subsequent
    holding period.
    """
    config = config or BacktestConfig()

    clean_returns = _validate_returns(returns)

    if not isinstance(lookback_days, int) or lookback_days < 2:
        raise ValueError("lookback_days must be an integer of at least 2.")

    if len(clean_returns) <= lookback_days:
        raise ValueError(
            "Returns must contain more observations than lookback_days."
        )

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in the interval (0, 1].")

    estimate_weights = _get_strategy(
        strategy=strategy,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )

    assets = clean_returns.columns

    # The first test observation follows the initial estimation window.
    first_test_position = lookback_days
    test_returns = clean_returns.iloc[first_test_position:]

    # Estimate initial weights using historical observations only.
    initial_history = clean_returns.iloc[
        first_test_position - lookback_days:first_test_position
    ]

    current_weights = _validate_weights(
        estimate_weights(initial_history),
        assets,
    )

    # Rebalance at the final available trading date of each period,
    # excluding the final dataset date.
    rebalance_dates = _rebalance_dates(
        clean_returns.index,
        config.rebalance_frequency,
    )

    total_cost_rate = (
        config.transaction_cost_bps + config.slippage_bps
    ) / 10_000.0

    portfolio_value = float(config.initial_capital)

    records: list[dict[str, object]] = []

    for date, asset_returns in test_returns.iterrows():
        previous_value = portfolio_value

        # Charge initial portfolio implementation costs.
        if date == test_returns.index[0] and config.charge_initial_transaction:
            turnover = float(current_weights.abs().sum())
            trading_cost = portfolio_value * turnover * total_cost_rate
            portfolio_value -= trading_cost
        else:
            turnover = 0.0
            trading_cost = 0.0

        # Apply existing weights to the current day's asset returns.
        gross_daily_return = float(current_weights @ asset_returns)
        portfolio_value *= 1.0 + gross_daily_return

        # Calculate naturally drifted portfolio weights.
        asset_growth = 1.0 + asset_returns
        drifted_values = current_weights * asset_growth
        growth_total = float(drifted_values.sum())

        if growth_total <= 0 or not np.isfinite(growth_total):
            raise ValueError(f"Invalid portfolio value on {date}.")

        drifted_weights = drifted_values / growth_total
        current_weights = drifted_weights

        is_rebalance = date in rebalance_dates

        if is_rebalance:
            # Exclude the current date's return from estimation.
            # Only information available before this date is used.
            history_end_position = clean_returns.index.get_loc(date)
            history_start_position = history_end_position - lookback_days

            if history_start_position < 0:
                raise ValueError(
                    f"Insufficient historical data before rebalance date {date}."
                )

            estimation_returns = clean_returns.iloc[
                history_start_position:history_end_position
            ]

            new_weights = _validate_weights(
                estimate_weights(estimation_returns),
                assets,
            )

            # One-way turnover for a fully invested long-only portfolio.
            turnover = float(
                0.5 * (new_weights - current_weights).abs().sum()
            )

            trading_cost = portfolio_value * turnover * total_cost_rate
            portfolio_value -= trading_cost

            current_weights = new_weights

        net_daily_return = portfolio_value / previous_value - 1.0

        record: dict[str, object] = {
            "date": date,
            "portfolio_value": portfolio_value,
            "gross_daily_return": gross_daily_return,
            "net_daily_return": net_daily_return,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "is_rebalance": is_rebalance,
        }

        # These are the weights held after any closing rebalance.
        # They apply to the next trading session.
        for asset in assets:
            record[f"weight_{asset}"] = float(current_weights[asset])

        records.append(record)

    results = pd.DataFrame(records).set_index("date")

    results["cumulative_return"] = (
        results["portfolio_value"] / config.initial_capital
    ) - 1.0

    results["running_peak"] = results["portfolio_value"].cummax().clip(
        lower=config.initial_capital
    )

    results["drawdown"] = (
        results["portfolio_value"] / results["running_peak"]
    ) - 1.0

    net_daily_returns = results["net_daily_return"]

    total_return = (
        results["portfolio_value"].iloc[-1] / config.initial_capital
    ) - 1.0

    # Annualize only when at least 252 out-of-sample observations exist.
    if len(net_daily_returns) >= 252 and total_return > -1.0:
        elapsed_years = len(net_daily_returns) / 252.0

        annualized_return = (
            (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
        )

        annualized_volatility = float(
            net_daily_returns.std(ddof=1) * np.sqrt(252.0)
        )

        daily_return_std = float(net_daily_returns.std(ddof=1))

        # Standard Sharpe ratio based on daily excess returns.
        if daily_return_std > 0:
            daily_risk_free_rate = risk_free_rate / 252.0

            sharpe_ratio = (
                (float(net_daily_returns.mean()) - daily_risk_free_rate)
                * 252.0
                / (daily_return_std * np.sqrt(252.0))
            )
        else:
            sharpe_ratio = np.nan

    else:
        annualized_return = np.nan
        annualized_volatility = np.nan
        sharpe_ratio = np.nan

    metrics = {
        "strategy": strategy,
        "lookback_days": float(lookback_days),
        "test_observations": float(len(results)),
        "initial_capital": float(config.initial_capital),
        "final_value": float(results["portfolio_value"].iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(results["drawdown"].min()),
        "total_transaction_costs": float(results["trading_cost"].sum()),
        "total_turnover": float(results["turnover"].sum()),
        "rebalance_count": float(results["is_rebalance"].sum()),
    }

    return results, metrics
