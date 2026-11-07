"""Validate and report mean-variance portfolio optimization results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantfin.estimators import estimate_market_inputs
from quantfin.optimization import (
    PortfolioConstraints,
    efficient_frontier,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    portfolio_performance,
)

ROOT = pd.Path if False else None  # No filesystem dependency required.
TOLERANCE = 1e-5


def main() -> None:
    """Run optimization algorithms, validate results and save reports."""

    returns = pd.read_parquet(
        "data/processed/us_equities_simple_returns.parquet"
    )

    expected_returns, covariance = estimate_market_inputs(returns)

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.50,
        allow_short_selling=False,
        fully_invested=True,
        weight_sum=1.0,
    )

    risk_free_rate = 0.04

    minimum_weights = minimum_volatility_weights(
        expected_returns, covariance, constraints
    )

    maximum_weights = maximum_sharpe_weights(
        expected_returns,
        covariance,
        risk_free_rate,
        constraints,
    )

    frontier = efficient_frontier(
        expected_returns,
        covariance,
        constraints,
        points=30,
    )

    assets = expected_returns.index.tolist()
    weight_columns = [f"weight_{asset}" for asset in assets]

    checks: list[dict[str, object]] = []

    def record(
        name: str,
        passed: bool,
        detail: str,
    ) -> None:
        checks.append(
            {
                "validation": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    # 1. Frontier dimensions.
    record(
        "Frontier portfolio count",
        len(frontier) == 30,
        f"Found {len(frontier)} portfolios; expected 30.",
    )

    # 2. Asset weight columns.
    record(
        "Asset weight columns",
        all(column in frontier.columns for column in weight_columns),
        f"Found {len(weight_columns)} expected asset weight columns.",
    )

    weights = frontier[weight_columns].to_numpy(dtype=float)

    # 3. Finite values.
    numeric_values = frontier.select_dtypes(include="number").to_numpy()

    record(
        "Finite frontier values",
        bool(np.isfinite(numeric_values).all()),
        "All frontier numeric values must be finite.",
    )

    # 4. Fully invested portfolios.
    weight_sums = weights.sum(axis=1)

    record(
        "Portfolio weights sum to one",
        bool(np.allclose(weight_sums, 1.0, atol=TOLERANCE)),
        f"Weight-sum range: {weight_sums.min():.8f} to "
        f"{weight_sums.max():.8f}.",
    )

    # 5. Long-only constraint.
    record(
        "No short positions",
        bool((weights >= -TOLERANCE).all()),
        f"Minimum allocation: {weights.min():.8f}.",
    )

    # 6. Maximum asset allocation.
    record(
        "Maximum allocation constraint",
        bool((weights <= 0.50 + TOLERANCE).all()),
        f"Maximum allocation: {weights.max():.8f}.",
    )

    # 7. Target and achieved expected returns.
    return_difference = np.abs(
        frontier["target_return"].to_numpy()
        - frontier["expected_annual_return"].to_numpy()
    )

    record(
        "Target return accuracy",
        bool((return_difference <= TOLERANCE).all()),
        f"Maximum difference: {return_difference.max():.8e}.",
    )

    # 8. Positive volatility.
    record(
        "Positive portfolio volatility",
        bool((frontier["annual_volatility"] > 0).all()),
        f"Minimum volatility: "
        f"{frontier['annual_volatility'].min():.8f}.",
    )

    # 9. Expected return ordering.
    record(
        "Increasing target returns",
        bool(
            np.all(
                np.diff(frontier["target_return"].to_numpy()) > 0
            )
        ),
        "Target returns must increase across frontier portfolios.",
    )

    # 10. Standalone minimum-volatility comparison.
    minimum_performance = portfolio_performance(
        minimum_weights,
        expected_returns,
        covariance,
        risk_free_rate,
    )

    frontier_minimum_volatility = float(
        frontier["annual_volatility"].min()
    )

    minimum_volatility_difference = abs(
        minimum_performance["annual_volatility"]
        - frontier_minimum_volatility
    )

    record(
        "Minimum-volatility consistency",
        minimum_volatility_difference <= 0.002,
        f"Standalone: {minimum_performance['annual_volatility']:.6f}; "
        f"frontier: {frontier_minimum_volatility:.6f}.",
    )

    # 11. Standalone maximum-Sharpe comparison.
    maximum_performance = portfolio_performance(
        maximum_weights,
        expected_returns,
        covariance,
        risk_free_rate,
    )

    frontier_maximum_sharpe = float(
        frontier["sharpe_ratio"].max()
    )

    sharpe_difference = abs(
        maximum_performance["sharpe_ratio"]
        - frontier_maximum_sharpe
    )

    record(
        "Maximum-Sharpe consistency",
        sharpe_difference <= 0.01,
        f"Standalone: {maximum_performance['sharpe_ratio']:.6f}; "
        f"frontier: {frontier_maximum_sharpe:.6f}.",
    )

    # Save frontier results.
    frontier.to_csv(
        "reports/tables/efficient_frontier.csv",
        index=False,
    )

    frontier.to_parquet(
        "reports/tables/efficient_frontier.parquet",
        index=False,
    )

    # Save individual optimized portfolio weights.
    portfolio_weights = pd.DataFrame(
        {
            "minimum_volatility": minimum_weights,
            "maximum_sharpe": maximum_weights,
        }
    )

    portfolio_weights.to_csv(
        "reports/tables/optimized_portfolio_weights.csv"
    )

    portfolio_weights.to_parquet(
        "reports/tables/optimized_portfolio_weights.parquet"
    )

    # Save standalone performance comparison.
    performance_summary = pd.DataFrame(
        [
            {
                "portfolio": "Minimum Volatility",
                **minimum_performance,
            },
            {
                "portfolio": "Maximum Sharpe",
                **maximum_performance,
            },
        ]
    )

    performance_summary.to_csv(
        "reports/tables/optimized_portfolio_performance.csv",
        index=False,
    )

    performance_summary.to_parquet(
        "reports/tables/optimized_portfolio_performance.parquet",
        index=False,
    )

    # Save validation report.
    validation_report = pd.DataFrame(checks)

    validation_report.to_csv(
        "reports/tables/optimization_validation_summary.csv",
        index=False,
    )

    validation_report.to_parquet(
        "reports/tables/optimization_validation_summary.parquet",
        index=False,
    )

    print("\nOPTIMIZATION VALIDATION RESULTS")
    print("=" * 75)
    print(validation_report.to_string(index=False))

    passed = int((validation_report["status"] == "PASS").sum())
    failed = int((validation_report["status"] == "FAIL").sum())

    print("\nTOTAL CHECKS :", len(checks))
    print("PASSED       :", passed)
    print("FAILED       :", failed)

    print("\nGENERATED REPORTS")
    print("- Efficient frontier CSV and Parquet")
    print("- Optimized portfolio weights CSV and Parquet")
    print("- Optimized portfolio performance CSV and Parquet")
    print("- Optimization validation summary CSV and Parquet")

    if failed:
        raise SystemExit(
            "OPTIMIZATION VALIDATION FAILED. Review the report."
        )

    print("\nALL OPTIMIZATION VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()