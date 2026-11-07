
"""Validate expected-return and covariance estimators on real market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantfin.estimators import estimate_market_inputs

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "us_equities_simple_returns.parquet"
REPORT_DIR = ROOT / "reports" / "tables"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input data not found: {INPUT_FILE}")

    returns = pd.read_parquet(INPUT_FILE)

    expected_returns, covariance = estimate_market_inputs(
        returns,
        return_method="historical_mean",
        covariance_method="ledoit_wolf",
        periods_per_year=252,
    )

    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "Expected 12 equity assets",
        len(expected_returns) == 12,
        f"Found {len(expected_returns)} assets",
    ))
    checks.append((
        "Expected returns are finite",
        bool(np.isfinite(expected_returns.to_numpy()).all()),
        "No NaN or infinite expected returns",
    ))
    checks.append((
        "Covariance dimensions",
        covariance.shape == (12, 12),
        f"Found {covariance.shape}",
    ))
    checks.append((
        "Covariance values are finite",
        bool(np.isfinite(covariance.to_numpy()).all()),
        "No NaN or infinite covariance values",
    ))
    checks.append((
        "Covariance symmetry",
        bool(np.allclose(covariance, covariance.T, atol=1e-10)),
        "Symmetry checked",
    ))

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
    minimum_eigenvalue = float(eigenvalues.min())

    checks.append((
        "Covariance positive definite",
        minimum_eigenvalue > 0,
        f"Minimum eigenvalue: {minimum_eigenvalue:.10e}",
    ))
    checks.append((
        "Asset labels aligned",
        expected_returns.index.equals(covariance.index)
        and covariance.index.equals(covariance.columns),
        "Expected returns and covariance labels match",
    ))

    print("\nESTIMATOR VALIDATION")
    print("=" * 78)

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<35} {status:<8} {detail}")

    print("\nEXPECTED ANNUAL RETURNS")
    print(expected_returns.sort_values(ascending=False).to_string())

    print("\nANNUALIZED COVARIANCE MATRIX")
    print(covariance.round(6).to_string())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    expected_returns.to_csv(
        REPORT_DIR / "estimated_annual_returns.csv",
        header=True,
    )
    covariance.to_csv(
        REPORT_DIR / "estimated_ledoit_wolf_covariance.csv",
    )

    pd.DataFrame({
        "asset": expected_returns.index,
        "expected_annual_return": expected_returns.to_numpy(),
        "annualized_volatility": np.sqrt(np.diag(covariance.to_numpy())),
    }).to_csv(
        REPORT_DIR / "estimated_market_inputs_summary.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "details": detail,
            }
            for name, passed, detail in checks
        ]
    )
    summary.to_csv(
        REPORT_DIR / "estimator_validation_summary.csv",
        index=False,
    )
    summary.to_parquet(
        REPORT_DIR / "estimator_validation_summary.parquet",
        index=False,
    )

    passed_count = sum(passed for _, passed, _ in checks)
    failed_count = len(checks) - passed_count

    print("\n" + "=" * 78)
    print(f"Total checks : {len(checks)}")
    print(f"Passed       : {passed_count}")
    print(f"Failed       : {failed_count}")
    print(f"Minimum covariance eigenvalue: {minimum_eigenvalue:.10e}")

    if failed_count:
        raise SystemExit("ESTIMATOR VALIDATION FAILED.")

    print("\nALL ESTIMATOR VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
