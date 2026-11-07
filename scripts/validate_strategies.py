
"""Validate portfolio investment strategies using processed US equity returns."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.strategies import (
    black_litterman_strategy,
    buy_and_hold_weights,
    equal_weight_weights,
    hrp_strategy,
    market_cap_weighted_weights,
    maximum_sharpe_strategy,
    minimum_volatility_strategy,
    risk_parity_strategy,
)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
MAX_WEIGHT = 0.50

ROOT = Path(__file__).resolve().parents[1]
RETURNS_FILE = ROOT / "data" / "processed" / "us_equities_simple_returns.csv"
OUTPUT_DIR = ROOT / "reports" / "investment_strategies"


def load_returns() -> pd.DataFrame:
    """Load and validate processed simple returns."""
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(f"Returns file not found: {RETURNS_FILE}")

    returns = pd.read_csv(
        RETURNS_FILE,
        index_col=0,
        parse_dates=True,
    )

    returns = returns.sort_index()
    returns = returns.dropna(how="all")
    returns = returns.dropna(axis=0, how="any")
    returns = returns.astype(float)

    if returns.empty:
        raise ValueError("No complete return observations were loaded.")

    if returns.columns.has_duplicates:
        raise ValueError("Duplicate asset columns detected.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    if (returns <= -1.0).any().any():
        raise ValueError("Simple returns contain values at or below -100%.")

    return returns


def validate_weights(
    name: str,
    weights: pd.Series,
    assets: pd.Index,
    max_weight: float | None = None,
) -> None:
    """Check portfolio weights and raise an error if validation fails."""
    if not isinstance(weights, pd.Series):
        raise TypeError(f"{name}: weights must be a pandas Series.")

    if not weights.index.equals(assets):
        raise ValueError(f"{name}: asset labels or ordering do not match.")

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite weights detected.")

    if (values < -1e-8).any():
        raise ValueError(f"{name}: negative weights detected.")

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"{name}: weights do not sum to one.")

    if max_weight is not None and (weights > max_weight + 1e-6).any():
        raise ValueError(f"{name}: maximum weight constraint violated.")

    print(f"PASS | {name:<25} | Sum: {weights.sum():.8f}")


def calculate_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> dict[str, float]:
    """Calculate annualized in-sample portfolio statistics."""
    expected_returns = returns.mean() * TRADING_DAYS
    covariance = returns.cov() * TRADING_DAYS

    annual_return = float(weights @ expected_returns)
    annual_variance = float(weights @ covariance @ weights)
    annual_volatility = float(np.sqrt(max(annual_variance, 0.0)))

    sharpe = (
        (annual_return - RISK_FREE_RATE) / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    return {
        "Expected Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
    }


def main() -> None:
    print("=" * 72)
    print("INVESTMENT STRATEGY VALIDATION")
    print("=" * 72)

    returns = load_returns()
    assets = returns.columns

    print(f"\nObservations: {len(returns):,}")
    print(f"Assets:       {len(assets)}")
    print(f"First date:   {returns.index.min().date()}")
    print(f"Last date:    {returns.index.max().date()}")

    # Common equal-weight starting allocation.
    equal_weights = equal_weight_weights(returns)

    # Market-cap function test using illustrative values only.
    # These are NOT actual historical market capitalizations.
    illustrative_caps = pd.Series(
        np.arange(1, len(assets) + 1, dtype=float) * 1_000_000,
        index=assets,
    )
    cap_test_weights = market_cap_weighted_weights(
        illustrative_caps,
        assets=assets,
    )
    validate_weights(
        "Market Cap Function Test",
        cap_test_weights,
        assets,
    )

    # Black-Litterman investor view:
    # NVDA expected excess return minus KO expected excess return = 4%.
    if "NVDA" not in assets or "KO" not in assets:
        raise ValueError("NVDA and KO are required for the configured BL view.")

    view_matrix = pd.DataFrame(
        0.0,
        index=["NVDA_minus_KO"],
        columns=assets,
    )
    view_matrix.loc["NVDA_minus_KO", "NVDA"] = 1.0
    view_matrix.loc["NVDA_minus_KO", "KO"] = -1.0

    view_returns = pd.Series(
        [0.04],
        index=view_matrix.index,
    )

    strategies = {
        "Equal Weight": equal_weights,
        "Minimum Volatility": minimum_volatility_strategy(
            returns,
            max_weight=MAX_WEIGHT,
        ),
        "Maximum Sharpe": maximum_sharpe_strategy(
            returns,
            risk_free_rate=RISK_FREE_RATE,
            max_weight=MAX_WEIGHT,
        ),
        "HRP": hrp_strategy(returns),
        "Risk Parity": risk_parity_strategy(
            returns,
            max_weight=MAX_WEIGHT,
        ),
        "Black-Litterman": black_litterman_strategy(
            returns=returns,
            market_weights=equal_weights,
            view_matrix=view_matrix,
            view_returns=view_returns,
            risk_free_rate=RISK_FREE_RATE,
            confidence=0.60,
            max_weight=MAX_WEIGHT,
        ),
        "Buy and Hold": buy_and_hold_weights(equal_weights),
    }

    print("\nWEIGHT VALIDATION")
    print("-" * 72)

    for name, weights in strategies.items():
        constraint = (
            MAX_WEIGHT
            if name in {
                "Minimum Volatility",
                "Maximum Sharpe",
                "Risk Parity",
                "Black-Litterman",
            }
            else None
        )

        validate_weights(name, weights, assets, constraint)

    weights_table = pd.DataFrame(strategies)
    metrics_table = pd.DataFrame(
        {
            name: calculate_metrics(returns, weights)
            for name, weights in strategies.items()
        }
    ).T

    # Save outputs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weights_table.to_csv(OUTPUT_DIR / "strategy_weights.csv")
    weights_table.to_parquet(OUTPUT_DIR / "strategy_weights.parquet")

    metrics_table.to_csv(OUTPUT_DIR / "strategy_performance.csv")
    metrics_table.to_parquet(OUTPUT_DIR / "strategy_performance.parquet")

    # Save the illustrative market-cap function test separately.
    cap_test_weights.rename("Illustrative Weight").to_csv(
        OUTPUT_DIR / "market_cap_function_test_illustrative.csv"
    )

    print("\nSTRATEGY PERFORMANCE — IN-SAMPLE ESTIMATES")
    print("-" * 72)
    print(metrics_table.round(4).to_string())

    print("\nOUTPUT FILES")
    print("-" * 72)
    print(f"Folder: {OUTPUT_DIR}")
    print(" - strategy_weights.csv / .parquet")
    print(" - strategy_performance.csv / .parquet")
    print(" - market_cap_function_test_illustrative.csv")

    print("\nVALIDATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
