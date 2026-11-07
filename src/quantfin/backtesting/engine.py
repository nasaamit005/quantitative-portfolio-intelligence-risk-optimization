
"""Daily portfolio backtesting with rebalancing and trading costs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for portfolio backtesting."""

    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    rebalance_frequency: str = "M"
    charge_initial_transaction: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive and finite.")

        if not np.isfinite(self.transaction_cost_bps):
            raise ValueError("transaction_cost_bps must be finite.")

        if not np.isfinite(self.slippage_bps):
            raise ValueError("slippage_bps must be finite.")

        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Trading costs cannot be negative.")

        if self.transaction_cost_bps + self.slippage_bps >= 10_000:
            raise ValueError("Combined trading costs must be below 10,000 bps.")

        if self.rebalance_frequency not in {"M", "Q", "Y"}:
            raise ValueError(
                "Frequency must be M (monthly), Q (quarterly), "
                "or Y (annual)."
            )


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate daily simple asset returns."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must contain observations and assets.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset columns must be unique.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex.")

    if returns.index.has_duplicates:
        raise ValueError("Return dates must be unique.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must be chronologically ordered.")

    clean = returns.astype(float)

    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Returns contain missing or non-finite values.")

    if (clean <= -1.0).any().any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    return clean


def _validate_weights(
    weights: pd.Series,
    assets: pd.Index,
) -> pd.Series:
    """Validate long-only fully invested target weights."""
    if not isinstance(weights, pd.Series):
        raise TypeError("Target weights must be a pandas Series.")

    if weights.index.has_duplicates:
        raise ValueError("Weight asset labels must be unique.")

    weights = weights.reindex(assets)

    if weights.isna().any():
        raise ValueError("Target weights must cover every asset.")

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Weights must be finite.")

    if (values < -1e-8).any():
        raise ValueError("Short positions are not supported.")

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Target weights must sum to one.")

    weights = weights.clip(lower=0.0)
    return weights / weights.sum()


def _rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str,
) -> set[pd.Timestamp]:
    """Return period-end trading dates, excluding the final dataset date."""
    periods = index.to_period(frequency)
    dates = pd.Series(index, index=index)

    period_end_dates = dates.groupby(periods).last().tolist()

    # Do not trade on the final date: there is no subsequent holding period.
    if period_end_dates:
        period_end_dates = period_end_dates[:-1]

    return set(period_end_dates)


def run_backtest(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a daily backtest with periodic rebalancing.

    Timing convention:
    - Target weights are applied at the beginning of a return period.
    - Daily asset returns are then earned.
    - At a scheduled rebalance date, the portfolio is rebalanced at that
      date's close, after that day's return has been earned.
    - The resulting target allocation applies to the next trading session.

    This convention avoids using the current day's return to choose that
    day's opening portfolio allocation.

    This function uses fixed target weights. Walk-forward weight estimation
    is a separate layer and must supply weights known before each period.
    """
    config = config or BacktestConfig()
    clean_returns = _validate_returns(returns)
    assets = clean_returns.columns

    weights = _validate_weights(target_weights, assets)
    rebalance_dates = _rebalance_dates(
        clean_returns.index,
        config.rebalance_frequency,
    )

    total_cost_rate = (
        config.transaction_cost_bps + config.slippage_bps
    ) / 10_000.0

    portfolio_value = float(config.initial_capital)
    current_weights = weights.copy()

    records: list[dict[str, float | bool]] = []

    for position, (date, asset_returns) in enumerate(
        clean_returns.iterrows()
    ):
        # Apply initial trading costs before the first holding-period return.
        initial_rebalance = (
            position == 0 and config.charge_initial_transaction
        )

        if initial_rebalance:
            turnover = float(current_weights.abs().sum())
            trading_cost = portfolio_value * turnover * total_cost_rate
            portfolio_value -= trading_cost
        else:
            turnover = 0.0
            trading_cost = 0.0

        # Earn the day's return using the weights held at the start of day.
        daily_return = float(current_weights @ asset_returns)
        portfolio_value *= 1.0 + daily_return

        # Drift weights after the daily asset returns.
        asset_growth = 1.0 + asset_returns
        drifted_values = current_weights * asset_growth
        growth_total = float(drifted_values.sum())

        if growth_total <= 0 or not np.isfinite(growth_total):
            raise ValueError(f"Invalid portfolio value on {date}.")

        drifted_weights = drifted_values / growth_total

        is_rebalance = date in rebalance_dates

        if is_rebalance:
            # One-way turnover: half the sum of absolute weight changes.
            turnover = float(
                0.5 * (weights - drifted_weights).abs().sum()
            )

            # Trading cost is charged on traded notional.
            trading_cost = portfolio_value * turnover * total_cost_rate
            portfolio_value -= trading_cost

            current_weights = weights.copy()
        else:
            current_weights = drifted_weights

        records.append(
            {
                "date": date,
                "portfolio_value": portfolio_value,
                "daily_return": daily_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "is_rebalance": is_rebalance,
            }
        )

    results = pd.DataFrame(records).set_index("date")

    results["cumulative_return"] = (
        results["portfolio_value"] / config.initial_capital
    ) - 1.0

    # Include initial capital in the running peak so that losses on the
    # first trading day are reflected in drawdown.
    results["running_peak"] = results["portfolio_value"].cummax().clip(
        lower=config.initial_capital
    )

    results["drawdown"] = (
        results["portfolio_value"] / results["running_peak"]
    ) - 1.0

    # Portfolio-value returns include the effect of trading costs.
    net_daily_returns = results["portfolio_value"].pct_change().dropna()

    total_return = (
        results["portfolio_value"].iloc[-1] / config.initial_capital
    ) - 1.0

    # Suppress annualized metrics for samples shorter than one trading year.
    # This avoids misleading annualized figures from very short datasets.
    if len(net_daily_returns) >= 252 and total_return > -1.0:
        elapsed_years = len(net_daily_returns) / 252.0

        annualized_return = (
            (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
        )

        annualized_volatility = float(
            net_daily_returns.std(ddof=1) * np.sqrt(252)
        )
    else:
        annualized_return = np.nan
        annualized_volatility = np.nan

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if (
            np.isfinite(annualized_return)
            and np.isfinite(annualized_volatility)
            and annualized_volatility > 0
        )
        else np.nan
    )

    metrics = {
        "initial_capital": float(config.initial_capital),
        "final_value": float(results["portfolio_value"].iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio_zero_rf": float(sharpe_ratio),
        "max_drawdown": float(results["drawdown"].min()),
        "total_transaction_costs": float(results["trading_cost"].sum()),
        "total_turnover": float(results["turnover"].sum()),
        "rebalance_count": float(results["is_rebalance"].sum()),
    }

    return results, metrics
