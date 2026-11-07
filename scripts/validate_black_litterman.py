
"""Validate Black-Litterman portfolio return estimation."""

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.optimization import (
    black_litterman_posterior,
    market_implied_equilibrium_returns,
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

RISK_AVERSION = 2.5
RISK_FREE_RATE = 0.04
TAU = 0.05
CONFIDENCE = 0.60
VIEW_RETURN = 0.04
TOLERANCE = 1e-8

checks = []


def record_check(
    name: str,
    passed: bool,
    details: str,
) -> None:
    """Record and print validation results."""

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
    """Run Black-Litterman validation and save outputs."""

    if not RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Returns dataset not found: {RETURNS_FILE}"
        )

    returns = pd.read_parquet(RETURNS_FILE).dropna()

    asset_count = returns.shape[1]

    covariance = returns.cov() * 252

    # Equal weights are used as a transparent market-weight proxy.
    # These are not actual market capitalization weights.
    market_weights = pd.Series(
        1.0 / asset_count,
        index=returns.columns,
        name="market_weight_proxy",
    )

    # Illustrative relative view:
    # NVDA excess return - KO excess return = 4%.
    view_matrix = pd.DataFrame(
        0.0,
        index=["NVDA_vs_KO"],
        columns=returns.columns,
    )

    view_matrix.loc["NVDA_vs_KO", "NVDA"] = 1.0
    view_matrix.loc["NVDA_vs_KO", "KO"] = -1.0

    view_returns = pd.Series(
        [VIEW_RETURN],
        index=view_matrix.index,
        name="view_excess_return",
    )

    equilibrium = market_implied_equilibrium_returns(
        covariance,
        market_weights,
        risk_aversion=RISK_AVERSION,
        risk_free_rate=RISK_FREE_RATE,
    )

    (
        posterior,
        posterior_covariance,
        returned_equilibrium,
        omega,
    ) = black_litterman_posterior(
        covariance=covariance,
        market_weights=market_weights,
        view_matrix=view_matrix,
        view_returns=view_returns,
        risk_aversion=RISK_AVERSION,
        tau=TAU,
        confidence=CONFIDENCE,
    )

    # Numerical diagnostics.
    posterior_values = posterior.to_numpy()
    covariance_values = posterior_covariance.to_numpy()
    omega_values = omega.to_numpy()

    posterior_eigenvalues = np.linalg.eigvalsh(
        covariance_values
    )

    omega_eigenvalues = np.linalg.eigvalsh(
        omega_values
    )

    view_prior = float(
    (
        view_matrix.to_numpy()
        @ equilibrium.to_numpy()
    ).item()
)

    view_posterior = float(
    (
        view_matrix.to_numpy()
        @ posterior_values
    ).item()
)

    # Validation checks.
    record_check(
        "Asset count",
        asset_count == 12,
        f"Expected 12 assets; found {asset_count}",
    )

    record_check(
        "Equilibrium return labels",
        equilibrium.index.equals(returns.columns),
        "Equilibrium returns match asset universe",
    )

    record_check(
        "Posterior return labels",
        posterior.index.equals(returns.columns),
        "Posterior returns match asset universe",
    )

    record_check(
        "Finite equilibrium returns",
        np.isfinite(equilibrium.to_numpy()).all(),
        "All equilibrium returns are finite",
    )

    record_check(
        "Finite posterior returns",
        np.isfinite(posterior_values).all(),
        "All posterior returns are finite",
    )

    record_check(
        "Posterior covariance dimensions",
        posterior_covariance.shape == (
            asset_count,
            asset_count,
        ),
        f"Shape = {posterior_covariance.shape}",
    )

    record_check(
        "Posterior covariance symmetry",
        np.allclose(
            covariance_values,
            covariance_values.T,
            atol=TOLERANCE,
        ),
        "Posterior covariance is symmetric",
    )

    record_check(
        "Posterior covariance positive definite",
        posterior_eigenvalues.min() > 0,
        (
            "Minimum eigenvalue = "
            f"{posterior_eigenvalues.min():.10f}"
        ),
    )

    record_check(
        "View uncertainty positive definite",
        np.isfinite(omega_values).all()
        and omega_eigenvalues.min() > 0,
        (
            "Minimum view uncertainty eigenvalue = "
            f"{omega_eigenvalues.min():.10f}"
        ),
    )

    record_check(
        "Investor view labels",
        omega.index.equals(view_matrix.index)
        and omega.columns.equals(view_matrix.index),
        "View uncertainty labels match investor views",
    )

    record_check(
        "Equilibrium output consistency",
        np.allclose(
            equilibrium.to_numpy(),
            returned_equilibrium.to_numpy(),
            atol=TOLERANCE,
        ),
        "Both equilibrium calculations agree",
    )

    record_check(
        "Investor view influences posterior",
        abs(view_posterior - view_prior) > TOLERANCE,
        (
            f"Prior spread = {view_prior:.6f}; "
            f"posterior spread = {view_posterior:.6f}"
        ),
    )

    # Save equilibrium and posterior return estimates.
    return_output = pd.DataFrame(
        {
            "ticker": returns.columns,
            "equilibrium_excess_return": (
                equilibrium.reindex(returns.columns).to_numpy()
            ),
            "posterior_excess_return": (
                posterior.reindex(returns.columns).to_numpy()
            ),
        }
    )

    return_output["equilibrium_total_return"] = (
        return_output["equilibrium_excess_return"]
        + RISK_FREE_RATE
    )

    return_output["posterior_total_return"] = (
        return_output["posterior_excess_return"]
        + RISK_FREE_RATE
    )

    return_output.to_csv(
        REPORT_DIR / "black_litterman_returns.csv",
        index=False,
    )

    return_output.to_parquet(
        REPORT_DIR / "black_litterman_returns.parquet",
        index=False,
    )

    # Save posterior covariance.
    posterior_covariance.to_csv(
        REPORT_DIR / "black_litterman_posterior_covariance.csv"
    )

    posterior_covariance.to_parquet(
        REPORT_DIR / "black_litterman_posterior_covariance.parquet"
    )

    # Save view uncertainty.
    omega.to_csv(
        REPORT_DIR / "black_litterman_view_uncertainty.csv"
    )

    omega.to_parquet(
        REPORT_DIR / "black_litterman_view_uncertainty.parquet"
    )

    # Save investor view definition.
    view_output = view_matrix.copy()
    view_output["view_excess_return"] = view_returns

    view_output.to_csv(
        REPORT_DIR / "black_litterman_investor_views.csv"
    )

    view_output.to_parquet(
        REPORT_DIR / "black_litterman_investor_views.parquet"
    )

    # Save validation summary.
    validation = pd.DataFrame(checks)

    validation.to_csv(
        REPORT_DIR / "black_litterman_validation_summary.csv",
        index=False,
    )

    validation.to_parquet(
        REPORT_DIR / "black_litterman_validation_summary.parquet",
        index=False,
    )

    failed = int(
        validation["status"].ne("PASS").sum()
    )

    print("\nBLACK-LITTERMAN RETURN ESTIMATES")
    print(return_output.round(6).to_string(index=False))

    print("\nINVESTOR VIEW ANALYSIS")
    print(
        f"Prior NVDA-KO spread: {view_prior:.6%}"
        f"\nPosterior NVDA-KO spread: {view_posterior:.6%}"
        f"\nSpecified view spread: {VIEW_RETURN:.6%}"
    )

    print("\nVALIDATION SUMMARY")
    print(validation.to_string(index=False))

    print(
        f"\nTOTAL CHECKS: {len(validation)}"
        f"\nPASSED: {(validation['status'] == 'PASS').sum()}"
        f"\nFAILED: {failed}"
    )

    if failed:
        raise SystemExit(
            "BLACK-LITTERMAN VALIDATION FAILED."
        )

    print(
        "\nALL BLACK-LITTERMAN VALIDATION CHECKS PASSED."
    )


if __name__ == "__main__":
    main()
