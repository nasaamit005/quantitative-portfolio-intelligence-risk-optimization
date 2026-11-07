"""Validate the Hierarchical Risk Parity optimizer."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.optimization import (
    hierarchical_risk_parity_weights,
)

ROOT = Path(__file__).resolve().parents[1]

RETURNS_FILE = (
    ROOT / "data" / "processed"
    / "us_equities_simple_returns.parquet"
)

REPORT_DIR = ROOT / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
MAX_WEIGHT = 0.50
TOLERANCE = 1e-8

checks = []


def record_check(name: str, passed: bool, details: str) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )
    print(
        f"{name}: {'PASS' if passed else 'FAIL'}"
        f" | {details}"
    )


def main() -> None:
    returns = pd.read_parquet(RETURNS_FILE)

    weights = hierarchical_risk_parity_weights(returns)

    # Align data and portfolio weights.
    aligned_returns = returns[weights.index].dropna()

    expected_returns = aligned_returns.mean() * TRADING_DAYS
    covariance = aligned_returns.cov() * TRADING_DAYS

    portfolio_return = float(weights @ expected_returns)

    portfolio_variance = float(
        weights.to_numpy()
        @ covariance.to_numpy()
        @ weights.to_numpy()
    )

    portfolio_volatility = float(np.sqrt(portfolio_variance))

    sharpe_ratio = (
        portfolio_return - RISK_FREE_RATE
    ) / portfolio_volatility

    # Validation checks.
    record_check(
        "Asset count",
        len(weights) == returns.shape[1],
        f"{len(weights)} assets",
    )

    record_check(
        "Asset labels",
        set(weights.index) == set(returns.columns),
        "HRP weights match the equity universe",
    )

    record_check(
        "Finite weights",
        np.isfinite(weights.to_numpy()).all(),
        "All weights are finite",
    )

    record_check(
        "Fully invested",
        abs(weights.sum() - 1.0) <= TOLERANCE,
        f"Weight sum = {weights.sum():.10f}",
    )

    record_check(
        "No short positions",
        (weights >= -TOLERANCE).all(),
        f"Minimum weight = {weights.min():.6f}",
    )

    record_check(
        "Maximum allocation",
        (weights <= MAX_WEIGHT + TOLERANCE).all(),
        f"Maximum weight = {weights.max():.6f}",
    )

    record_check(
        "Positive portfolio volatility",
        np.isfinite(portfolio_volatility)
        and portfolio_volatility > 0,
        f"Annualized volatility = {portfolio_volatility:.4%}",
    )

    record_check(
        "Finite expected return",
        np.isfinite(portfolio_return),
        f"Annualized expected return = {portfolio_return:.4%}",
    )

    record_check(
        "Finite Sharpe ratio",
        np.isfinite(sharpe_ratio),
        f"Sharpe ratio = {sharpe_ratio:.4f}",
    )

    # Save portfolio weights.
    weights_output = (
        weights.rename("weight")
        .rename_axis("ticker")
        .reset_index()
        .sort_values("weight", ascending=False)
    )

    weights_output.to_csv(
        REPORT_DIR / "hrp_portfolio_weights.csv",
        index=False,
    )

    weights_output.to_parquet(
        REPORT_DIR / "hrp_portfolio_weights.parquet",
        index=False,
    )

    # Save portfolio performance.
    performance = pd.DataFrame(
        [
            {
                "strategy": "Hierarchical Risk Parity",
                "expected_annual_return": portfolio_return,
                "annualized_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "risk_free_rate": RISK_FREE_RATE,
                "asset_count": len(weights),
                "observations": len(aligned_returns),
            }
        ]
    )

    performance.to_csv(
        REPORT_DIR / "hrp_portfolio_performance.csv",
        index=False,
    )

    performance.to_parquet(
        REPORT_DIR / "hrp_portfolio_performance.parquet",
        index=False,
    )

    # Save validation summary.
    validation = pd.DataFrame(checks)

    validation.to_csv(
        REPORT_DIR / "hrp_validation_summary.csv",
        index=False,
    )

    validation.to_parquet(
        REPORT_DIR / "hrp_validation_summary.parquet",
        index=False,
    )

    failed = validation["status"].ne("PASS").sum()

    print("\nHRP PORTFOLIO PERFORMANCE")
    print(performance.to_string(index=False))

    print("\nVALIDATION SUMMARY")
    print(validation.to_string(index=False))

    print(
        f"\nTOTAL CHECKS: {len(validation)}"
        f"\nPASSED: {(validation['status'] == 'PASS').sum()}"
        f"\nFAILED: {failed}"
    )

    if failed:
        raise SystemExit("HRP VALIDATION FAILED.")

    print("\nALL HRP VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()