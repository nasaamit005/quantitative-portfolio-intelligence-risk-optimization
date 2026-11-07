
"""Validate and compare Black-Litterman portfolio optimization."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.optimization import (
    PortfolioConstraints,
    black_litterman_posterior,
    calculate_risk_contributions,
    equal_risk_contribution_weights,
    hierarchical_risk_parity_weights,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    portfolio_performance,
)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
RISK_AVERSION = 2.5
TAU = 0.05
VIEW_CONFIDENCE = 0.60
MAX_WEIGHT = 0.50

DATA_PATH = Path("data/processed/us_equities_simple_returns.parquet")
REPORT_DIR = Path("reports/black_litterman_optimization")


def check(condition: bool, name: str, details: str) -> dict:
    """Record a validation result."""
    status = "PASS" if condition else "FAIL"
    print(f"{name}: {status} | {details}")
    return {
        "check": name,
        "status": status,
        "details": details,
    }


def validate_weights(
    weights: pd.Series,
    assets: pd.Index,
    method: str,
) -> list[dict]:
    """Validate portfolio weights."""
    results = []

    aligned = weights.reindex(assets)

    results.append(
        check(
            aligned.notna().all(),
            f"{method}: complete asset weights",
            "All assets have portfolio weights",
        )
    )

    results.append(
        check(
            np.isfinite(aligned.to_numpy()).all(),
            f"{method}: finite weights",
            "All portfolio weights are finite",
        )
    )

    results.append(
        check(
            abs(float(aligned.sum()) - 1.0) < 1e-6,
            f"{method}: fully invested",
            f"Weight sum = {aligned.sum():.10f}",
        )
    )

    results.append(
        check(
            float(aligned.min()) >= -1e-7,
            f"{method}: long-only",
            f"Minimum weight = {aligned.min():.8f}",
        )
    )

    results.append(
        check(
            float(aligned.max()) <= MAX_WEIGHT + 1e-6,
            f"{method}: maximum weight",
            f"Maximum weight = {aligned.max():.8f}",
        )
    )

    return results


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Returns data not found: {DATA_PATH}")

    # ---------------------------------------------------------
    # 1. Load historical returns
    # ---------------------------------------------------------
    returns = pd.read_parquet(DATA_PATH).dropna().astype(float)

    if returns.empty:
        raise ValueError("Historical returns dataset is empty.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Historical returns contain non-finite values.")

    assets = returns.columns

    print("\nHISTORICAL DATA")
    print(f"Observations: {len(returns)}")
    print(f"Assets: {len(assets)}")
    print(f"Period: {returns.index.min()} to {returns.index.max()}")

    # Common historical estimates for portfolio comparison.
    historical_expected_returns = returns.mean() * TRADING_DAYS
    historical_covariance = returns.cov() * TRADING_DAYS

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=MAX_WEIGHT,
        allow_short_selling=False,
        fully_invested=True,
    )

    # ---------------------------------------------------------
    # 2. Construct illustrative Black-Litterman investor view
    # ---------------------------------------------------------
    market_weights = pd.Series(
        1.0 / len(assets),
        index=assets,
        dtype=float,
    )

    view_matrix = pd.DataFrame(
        0.0,
        index=["NVDA_vs_KO"],
        columns=assets,
    )

    view_matrix.loc["NVDA_vs_KO", "NVDA"] = 1.0
    view_matrix.loc["NVDA_vs_KO", "KO"] = -1.0

    view_returns = pd.Series(
        [0.04],
        index=view_matrix.index,
        dtype=float,
    )

    (
        posterior_excess_returns,
        posterior_covariance,
        equilibrium_excess_returns,
        omega,
    ) = black_litterman_posterior(
        covariance=historical_covariance,
        market_weights=market_weights,
        view_matrix=view_matrix,
        view_returns=view_returns,
        risk_aversion=RISK_AVERSION,
        tau=TAU,
        confidence=VIEW_CONFIDENCE,
    )

    posterior_total_returns = (
        posterior_excess_returns + RISK_FREE_RATE
    )

    # ---------------------------------------------------------
    # 3. Optimize all five portfolio strategies
    # ---------------------------------------------------------
    portfolios = {}

    portfolios["Minimum Volatility"] = minimum_volatility_weights(
        expected_returns=historical_expected_returns,
        covariance=historical_covariance,
        constraints=constraints,
    )

    portfolios["Maximum Sharpe"] = maximum_sharpe_weights(
        expected_returns=historical_expected_returns,
        covariance=historical_covariance,
        risk_free_rate=RISK_FREE_RATE,
        constraints=constraints,
    )

    portfolios["HRP"] = hierarchical_risk_parity_weights(
        returns=returns,
    )

    portfolios["Risk Parity"] = equal_risk_contribution_weights(
        returns=returns,
        max_weight=MAX_WEIGHT,
    )

    portfolios["Black-Litterman"] = maximum_sharpe_weights(
        expected_returns=posterior_total_returns,
        covariance=posterior_covariance,
        risk_free_rate=RISK_FREE_RATE,
        constraints=constraints,
    )

    # ---------------------------------------------------------
    # 4. Validate weights
    # ---------------------------------------------------------
    validation_results = []

    for method, weights in portfolios.items():
        validation_results.extend(
            validate_weights(weights, assets, method)
        )

    # ---------------------------------------------------------
    # 5. Portfolio performance comparison
    # ---------------------------------------------------------
    performance_rows = []

    for method, weights in portfolios.items():
        # Common historical evaluation for fair comparison.
        historical_metrics = portfolio_performance(
            weights=weights,
            expected_returns=historical_expected_returns,
            covariance=historical_covariance,
            risk_free_rate=RISK_FREE_RATE,
        )

        row = {
            "method": method,
            "evaluation_basis": "Historical common estimates",
            **historical_metrics,
        }
        performance_rows.append(row)

    # Additional Black-Litterman posterior evaluation.
    bl_posterior_metrics = portfolio_performance(
        weights=portfolios["Black-Litterman"],
        expected_returns=posterior_total_returns,
        covariance=posterior_covariance,
        risk_free_rate=RISK_FREE_RATE,
    )

    performance_rows.append(
        {
            "method": "Black-Litterman",
            "evaluation_basis": "Black-Litterman posterior estimates",
            **bl_posterior_metrics,
        }
    )

    performance = pd.DataFrame(performance_rows)

    # ---------------------------------------------------------
    # 6. Portfolio weights report
    # ---------------------------------------------------------
    weights_table = pd.DataFrame(portfolios).reindex(assets)
    weights_table.index.name = "ticker"

    # ---------------------------------------------------------
    # 7. Asset-level risk contributions
    # ---------------------------------------------------------
    risk_contribution_frames = []

    for method, weights in portfolios.items():
        contributions = calculate_risk_contributions(
            weights=weights,
            covariance=historical_covariance,
        ).copy()

        contributions["method"] = method
        contributions["ticker"] = contributions.index
        risk_contribution_frames.append(contributions.reset_index(drop=True))

    risk_contributions = pd.concat(
        risk_contribution_frames,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 8. Save Black-Litterman model inputs and outputs
    # ---------------------------------------------------------
    posterior_returns_report = pd.DataFrame(
        {
            "equilibrium_excess_return": equilibrium_excess_returns,
            "posterior_excess_return": posterior_excess_returns,
            "posterior_total_return": posterior_total_returns,
        }
    )
    posterior_returns_report.index.name = "ticker"

    # ---------------------------------------------------------
    # 9. Automated validation
    # ---------------------------------------------------------
    validation_results.append(
        check(
            len(portfolios) == 5,
            "Five optimization methods",
            f"Found {len(portfolios)} methods",
        )
    )

    validation_results.append(
        check(
            posterior_covariance.shape == (len(assets), len(assets)),
            "Posterior covariance dimensions",
            f"Shape = {posterior_covariance.shape}",
        )
    )

    covariance_eigenvalues = np.linalg.eigvalsh(
        posterior_covariance.to_numpy()
    )

    validation_results.append(
        check(
            covariance_eigenvalues.min() > 0,
            "Posterior covariance positive definite",
            f"Minimum eigenvalue = {covariance_eigenvalues.min():.10f}",
        )
    )

    validation_results.append(
        check(
            performance.notna().all().all(),
            "Performance metrics complete",
            "All reported performance metrics are populated",
        )
    )

    validation_results.append(
        check(
            len(risk_contributions) > 0,
            "Risk contributions generated",
            f"Rows generated = {len(risk_contributions)}",
        )
    )

    # ---------------------------------------------------------
    # 10. Export CSV and Parquet reports
    # ---------------------------------------------------------
    reports = {
        "portfolio_weights": weights_table,
        "portfolio_performance": performance,
        "risk_contributions": risk_contributions,
        "black_litterman_returns": posterior_returns_report,
        "black_litterman_posterior_covariance": posterior_covariance,
        "black_litterman_view_uncertainty": omega,
        "black_litterman_investor_views": view_matrix,
        "validation_summary": pd.DataFrame(validation_results),
    }

    for name, dataframe in reports.items():
        dataframe.to_csv(REPORT_DIR / f"{name}.csv")
        dataframe.to_parquet(REPORT_DIR / f"{name}.parquet")

    # ---------------------------------------------------------
    # 11. Display results
    # ---------------------------------------------------------
    print("\nPORTFOLIO WEIGHTS")
    print(weights_table.round(4).to_string())

    print("\nPORTFOLIO PERFORMANCE")
    print(performance.to_string(index=False))

    print("\nRISK CONTRIBUTIONS")
    print(risk_contributions.to_string(index=False))

    validation_summary = pd.DataFrame(validation_results)

    print("\nVALIDATION SUMMARY")
    print(validation_summary.to_string(index=False))

    passed = int((validation_summary["status"] == "PASS").sum())
    failed = int((validation_summary["status"] == "FAIL").sum())

    print(f"\nTOTAL CHECKS: {len(validation_summary)}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")

    print(f"\nREPORT DIRECTORY: {REPORT_DIR.resolve()}")

    if failed:
        raise AssertionError(
            f"Black-Litterman optimization validation failed: {failed} checks"
        )

    print("\nALL BLACK-LITTERMAN OPTIMIZATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
