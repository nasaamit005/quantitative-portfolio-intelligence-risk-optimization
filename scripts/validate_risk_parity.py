
"""Validate Equal Risk Contribution portfolio optimization."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.optimization import (
    calculate_risk_contributions,
    equal_risk_contribution_weights,
)

ROOT = Path(__file__).resolve().parents[1]

RETURNS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "us_equities_simple_returns.parquet"
)

REPORT_DIR = ROOT / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
MAX_WEIGHT = 0.50
TOLERANCE = 1e-8
TARGET_RISK_SHARE = 1.0 / 12

checks = []


def record_check(
    name: str,
    passed: bool,
    details: str,
) -> None:
    """Record and print a validation result."""

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
    """Run Risk Parity validation and save reports."""

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns dataset not found: {RETURNS_FILE}"
        )

    returns = pd.read_parquet(RETURNS_FILE)

    weights = equal_risk_contribution_weights(
        returns,
        max_weight=MAX_WEIGHT,
    )

    aligned_returns = returns[weights.index].dropna()

    annualized_covariance = (
        aligned_returns.cov() * TRADING_DAYS
    )

    annualized_expected_returns = (
        aligned_returns.mean() * TRADING_DAYS
    )

    contributions = calculate_risk_contributions(
        weights,
        annualized_covariance,
    )

    portfolio_return = float(
        weights @ annualized_expected_returns
    )

    portfolio_volatility = float(
        np.sqrt(
            weights.to_numpy()
            @ annualized_covariance.to_numpy()
            @ weights.to_numpy()
        )
    )

    sharpe_ratio = (
        portfolio_return - RISK_FREE_RATE
    ) / portfolio_volatility

    risk_shares = contributions["risk_contribution_pct"]

    maximum_risk_deviation = float(
        (risk_shares - TARGET_RISK_SHARE).abs().max()
    )

    # 1. Asset universe.
    record_check(
        "Asset count",
        len(weights) == returns.shape[1] == 12,
        f"Expected 12 assets; found {len(weights)}",
    )

    record_check(
        "Asset labels",
        set(weights.index) == set(returns.columns),
        "All equity labels match",
    )

    # 2. Weight integrity.
    record_check(
        "Finite weights",
        np.isfinite(weights.to_numpy()).all(),
        "All portfolio weights are finite",
    )

    record_check(
        "Fully invested",
        abs(weights.sum() - 1.0) <= TOLERANCE,
        f"Total weight = {weights.sum():.10f}",
    )

    record_check(
        "Long-only constraint",
        (weights >= -TOLERANCE).all(),
        f"Minimum weight = {weights.min():.8f}",
    )

    record_check(
        "Maximum allocation",
        (weights <= MAX_WEIGHT + TOLERANCE).all(),
        f"Maximum weight = {weights.max():.8f}",
    )

    # 3. Risk contribution integrity.
    record_check(
        "Finite risk contributions",
        np.isfinite(risk_shares.to_numpy()).all(),
        "All risk contribution percentages are finite",
    )

    record_check(
        "Risk contributions sum to 100%",
        abs(risk_shares.sum() - 1.0) <= TOLERANCE,
        f"Total contribution = {risk_shares.sum():.10f}",
    )

    record_check(
        "Equal risk contribution",
        maximum_risk_deviation <= 1e-5,
        (
            f"Maximum deviation from equal risk budget = "
            f"{maximum_risk_deviation:.10f}"
        ),
    )

    record_check(
        "Positive risk contributions",
        (risk_shares > 0).all(),
        "Every asset contributes positive portfolio risk",
    )

    # 4. Portfolio performance.
    record_check(
        "Positive portfolio volatility",
        np.isfinite(portfolio_volatility)
        and portfolio_volatility > 0,
        f"Annualized volatility = {portfolio_volatility:.6%}",
    )

    record_check(
        "Finite expected return",
        np.isfinite(portfolio_return),
        f"Annualized expected return = {portfolio_return:.6%}",
    )

    record_check(
        "Finite Sharpe ratio",
        np.isfinite(sharpe_ratio),
        f"Sharpe ratio = {sharpe_ratio:.6f}",
    )

    # 5. Save weights and risk contribution details.
    weights_output = (
        contributions.reset_index()
        .rename(columns={"index": "ticker"})
        .sort_values(
            "risk_contribution_pct",
            ascending=False,
        )
    )

    weights_output.to_csv(
        REPORT_DIR / "risk_parity_weights.csv",
        index=False,
    )

    weights_output.to_parquet(
        REPORT_DIR / "risk_parity_weights.parquet",
        index=False,
    )

    # 6. Save performance summary.
    performance = pd.DataFrame(
        [
            {
                "strategy": "Equal Risk Contribution",
                "expected_annual_return": portfolio_return,
                "annualized_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "risk_free_rate": RISK_FREE_RATE,
                "asset_count": len(weights),
                "observations": len(aligned_returns),
                "risk_contribution_sum": risk_shares.sum(),
                "maximum_risk_contribution_deviation": (
                    maximum_risk_deviation
                ),
            }
        ]
    )

    performance.to_csv(
        REPORT_DIR / "risk_parity_performance.csv",
        index=False,
    )

    performance.to_parquet(
        REPORT_DIR / "risk_parity_performance.parquet",
        index=False,
    )

    # 7. Save validation summary.
    validation = pd.DataFrame(checks)

    validation.to_csv(
        REPORT_DIR / "risk_parity_validation_summary.csv",
        index=False,
    )

    validation.to_parquet(
        REPORT_DIR / "risk_parity_validation_summary.parquet",
        index=False,
    )

    failed = int(
        validation["status"].ne("PASS").sum()
    )

    print("\nRISK PARITY PERFORMANCE")
    print(performance.to_string(index=False))

    print("\nRISK CONTRIBUTION DETAILS")
    print(weights_output.round(6).to_string(index=False))

    print("\nVALIDATION SUMMARY")
    print(validation.to_string(index=False))

    print(
        f"\nTOTAL CHECKS: {len(validation)}"
        f"\nPASSED: {(validation['status'] == 'PASS').sum()}"
        f"\nFAILED: {failed}"
    )

    if failed:
        raise SystemExit(
            "RISK PARITY VALIDATION FAILED."
        )

    print("\nALL RISK PARITY VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
