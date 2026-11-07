"""Validate correlation, covariance, and diversification outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables"
PROCESSED_DIR = ROOT / "data" / "processed"

DATASETS = {
    "us_equities": 12,
    "us_benchmarks": 3,
}

TRADING_DAYS = 252
TOLERANCE = 1e-8
CONDITION_WARNING_THRESHOLD = 1000.0

CORRELATION_FILES = (
    "pearson_correlation_matrix",
    "spearman_correlation_matrix",
)

COVARIANCE_FILES = (
    "sample_covariance_matrix",
    "annualized_covariance_matrix",
    "ledoit_wolf_covariance_matrix",
)


# ---------------------------------------------------------------------
# VALIDATION RESULT HANDLER
# ---------------------------------------------------------------------

def record(
    results: list[dict[str, str]],
    dataset: str,
    check: str,
    passed: bool,
    details: str,
) -> None:
    """Store an individual validation result."""

    results.append(
        {
            "dataset": dataset,
            "validation": check,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


# ---------------------------------------------------------------------
# MATRIX LOADING
# ---------------------------------------------------------------------

def load_matrix(
    dataset: str,
    matrix_name: str,
) -> pd.DataFrame:
    """Load a matrix from the reports directory."""

    path = TABLE_DIR / f"{dataset}_{matrix_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing matrix file: {path}"
        )

    matrix = pd.read_csv(path, index_col=0)

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    return matrix


# ---------------------------------------------------------------------
# MATRIX VALIDATION
# ---------------------------------------------------------------------

def validate_matrix(
    results: list[dict[str, str]],
    dataset: str,
    matrix_name: str,
    matrix: pd.DataFrame,
    expected_dimension: int,
    is_correlation: bool,
) -> None:
    """Validate matrix dimensions, numerical properties and labels."""

    values = matrix.to_numpy(dtype=float)

    # Dimension check
    dimension_ok = (
        values.shape
        == (expected_dimension, expected_dimension)
    )

    record(
        results,
        dataset,
        f"{matrix_name}: dimension",
        dimension_ok,
        (
            f"Expected {expected_dimension}x{expected_dimension}; "
            f"found {values.shape}."
        ),
    )

    # Instrument labels
    labels_ok = (
        matrix.index.equals(matrix.columns)
        and matrix.index.is_unique
        and matrix.columns.is_unique
    )

    record(
        results,
        dataset,
        f"{matrix_name}: matching labels",
        labels_ok,
        (
            "Row and column labels match and are unique."
            if labels_ok
            else "Labels do not match or contain duplicates."
        ),
    )

    # Finite values
    finite = bool(np.isfinite(values).all())

    record(
        results,
        dataset,
        f"{matrix_name}: finite values",
        finite,
        (
            "All matrix entries are finite."
            if finite
            else "NaN or infinite values detected."
        ),
    )

    # Stop numerical checks if the matrix cannot be evaluated safely.
    if (
        not finite
        or values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] == 0
    ):
        return

    # Symmetry
    symmetric = bool(
        np.allclose(
            values,
            values.T,
            atol=TOLERANCE,
            rtol=TOLERANCE,
        )
    )

    record(
        results,
        dataset,
        f"{matrix_name}: symmetry",
        symmetric,
        (
            "Matrix is symmetric."
            if symmetric
            else "Matrix is not symmetric."
        ),
    )

    # Eigenvalue diagnostics
    symmetric_values = (values + values.T) / 2

    eigenvalues = np.linalg.eigvalsh(symmetric_values)

    minimum_eigenvalue = float(eigenvalues.min())

    positive_semidefinite = (
        minimum_eigenvalue >= -TOLERANCE
    )

    record(
        results,
        dataset,
        f"{matrix_name}: positive semidefinite",
        positive_semidefinite,
        (
            f"Minimum eigenvalue: "
            f"{minimum_eigenvalue:.10g}."
        ),
    )

    positive_definite = (
        minimum_eigenvalue > TOLERANCE
    )

    record(
        results,
        dataset,
        f"{matrix_name}: positive definite",
        positive_definite,
        (
            f"Minimum eigenvalue: "
            f"{minimum_eigenvalue:.10g}."
        ),
    )

    # Condition number: high values are warnings, not automatic failures.
    condition_number = float(
        np.linalg.cond(values)
    )

    condition_finite = bool(
        np.isfinite(condition_number)
    )

    if condition_finite:
        condition_details = (
            f"Condition number: {condition_number:.4f}. "
            + (
                "Numerically well-conditioned under the "
                "configured warning threshold."
                if condition_number < CONDITION_WARNING_THRESHOLD
                else (
                    "WARNING: high numerical sensitivity; "
                    "inspect before portfolio optimization."
                )
            )
        )
    else:
        condition_details = (
            "Condition number is not finite; "
            "matrix may be singular."
        )

    record(
        results,
        dataset,
        f"{matrix_name}: condition number",
        condition_finite,
        condition_details,
    )

    # Correlation-specific checks
    if is_correlation:

        diagonal = np.diag(values)

        diagonal_ok = bool(
            np.allclose(
                diagonal,
                1.0,
                atol=TOLERANCE,
                rtol=0.0,
            )
        )

        record(
            results,
            dataset,
            f"{matrix_name}: unit diagonal",
            diagonal_ok,
            (
                "All diagonal values equal 1."
                if diagonal_ok
                else (
                    "Correlation diagonal contains "
                    "values other than 1."
                )
            ),
        )

        bounds_ok = bool(
            (values >= -1.0 - TOLERANCE).all()
            and (values <= 1.0 + TOLERANCE).all()
        )

        record(
            results,
            dataset,
            f"{matrix_name}: correlation bounds",
            bounds_ok,
            (
                "All correlations lie between -1 and +1."
                if bounds_ok
                else (
                    "Correlation values outside "
                    "[-1, 1] detected."
                )
            ),
        )


# ---------------------------------------------------------------------
# DATASET VALIDATION
# ---------------------------------------------------------------------

def validate_dataset(
    dataset: str,
    expected_dimension: int,
) -> list[dict[str, str]]:

    results: list[dict[str, str]] = []

    matrices: dict[str, pd.DataFrame] = {}

    # Validate correlation and covariance matrices
    for matrix_name in (
        CORRELATION_FILES + COVARIANCE_FILES
    ):

        matrix = load_matrix(
            dataset,
            matrix_name,
        )

        matrices[matrix_name] = matrix

        validate_matrix(
            results=results,
            dataset=dataset,
            matrix_name=matrix_name,
            matrix=matrix,
            expected_dimension=expected_dimension,
            is_correlation=(
                matrix_name in CORRELATION_FILES
            ),
        )

    # -----------------------------------------------------------------
    # Annualized covariance validation
    # -----------------------------------------------------------------

    sample = matrices["sample_covariance_matrix"]

    annualized = matrices["annualized_covariance_matrix"]

    annualization_ok = bool(
        sample.index.equals(annualized.index)
        and sample.columns.equals(annualized.columns)
        and np.allclose(
            annualized.to_numpy(dtype=float),
            sample.to_numpy(dtype=float) * TRADING_DAYS,
            rtol=1e-8,
            atol=1e-12,
        )
    )

    record(
        results,
        dataset,
        "Annualized covariance scaling",
        annualization_ok,
        (
            "Annualized covariance equals daily sample "
            "covariance multiplied by 252."
            if annualization_ok
            else (
                "Annualization formula or matrix labels "
                "do not match."
            )
        ),
    )

    # -----------------------------------------------------------------
    # Diversification metrics validation
    # -----------------------------------------------------------------

    diversification_path = (
        TABLE_DIR
        / f"{dataset}_diversification_metrics.csv"
    )

    if not diversification_path.exists():
        raise FileNotFoundError(
            f"Missing diversification file: "
            f"{diversification_path}"
        )

    diversification = pd.read_csv(
        diversification_path,
    )

    required_columns = [
        "instrument_count",
        "observations",
        "equal_weight_portfolio_volatility",
        "weighted_average_asset_volatility",
        "diversification_ratio",
        "average_pairwise_correlation",
    ]

    columns_ok = all(
        column in diversification.columns
        for column in required_columns
    )

    record(
        results,
        dataset,
        "Diversification metric columns",
        columns_ok,
        (
            "All required diversification fields are present."
            if columns_ok
            else "One or more required fields are missing."
        ),
    )

    if columns_ok:

        if diversification.empty:
            raise ValueError(
                f"{dataset}: diversification report is empty."
            )

        row = diversification.iloc[0]

        # Instrument count
        count_value = float(row["instrument_count"])

        count_ok = (
            np.isfinite(count_value)
            and count_value == expected_dimension
        )

        record(
            results,
            dataset,
            "Diversification instrument count",
            bool(count_ok),
            (
                f"Expected {expected_dimension}; "
                f"found {count_value}."
            ),
        )

        # Metric finiteness
        metric_values = (
            row[required_columns]
            .to_numpy(dtype=float)
        )

        metrics_finite = bool(
            np.isfinite(metric_values).all()
        )

        record(
            results,
            dataset,
            "Diversification metrics finite",
            metrics_finite,
            (
                "All diversification metrics are finite."
                if metrics_finite
                else "Non-finite diversification metric detected."
            ),
        )

        if metrics_finite:

            ratio = float(
                row["diversification_ratio"]
            )

            pairwise = float(
                row["average_pairwise_correlation"]
            )

            portfolio_vol = float(
                row["equal_weight_portfolio_volatility"]
            )

            weighted_vol = float(
                row["weighted_average_asset_volatility"]
            )

            observations = float(
                row["observations"]
            )

            # Diversification ratio
            ratio_ok = (
                ratio >= 1.0 - TOLERANCE
            )

            record(
                results,
                dataset,
                "Diversification ratio",
                ratio_ok,
                f"Diversification ratio: {ratio:.6f}.",
            )

            # Average pairwise correlation
            pairwise_ok = (
                -1.0 - TOLERANCE
                <= pairwise
                <= 1.0 + TOLERANCE
            )

            record(
                results,
                dataset,
                "Average pairwise correlation bounds",
                pairwise_ok,
                (
                    f"Average pairwise correlation: "
                    f"{pairwise:.6f}."
                ),
            )

            # Portfolio volatility consistency
            volatility_ok = (
                portfolio_vol >= 0.0
                and weighted_vol >= 0.0
                and portfolio_vol <= weighted_vol + TOLERANCE
            )

            record(
                results,
                dataset,
                "Portfolio volatility consistency",
                volatility_ok,
                (
                    f"Equal-weight portfolio volatility: "
                    f"{portfolio_vol:.6f}; "
                    f"weighted average asset volatility: "
                    f"{weighted_vol:.6f}."
                ),
            )

            # Observation count
            observations_ok = (
                np.isfinite(observations)
                and observations > 0
            )

            record(
                results,
                dataset,
                "Diversification observation count",
                bool(observations_ok),
                f"Observations: {observations}.",
            )

    # -----------------------------------------------------------------
    # Rolling correlation validation
    # -----------------------------------------------------------------

    rolling_path = (
        PROCESSED_DIR
        / f"{dataset}_rolling_63d_correlations.parquet"
    )

    rolling_exists = rolling_path.exists()

    record(
        results,
        dataset,
        "Rolling correlation report",
        rolling_exists,
        (
            "63-day rolling correlation Parquet file exists."
            if rolling_exists
            else "Rolling correlation report is missing."
        ),
    )

    if rolling_exists:

        rolling = pd.read_parquet(
            rolling_path
        )

        rolling_values = rolling.to_numpy(
            dtype=float
        )

        # NaN values are allowed during the initial rolling window.
        # Infinite values are not allowed.
        non_missing = rolling_values[
            ~np.isnan(rolling_values)
        ]

        rolling_finite = bool(
            np.isfinite(non_missing).all()
        )

        record(
            results,
            dataset,
            "Rolling correlation finite values",
            rolling_finite,
            (
                "All non-missing rolling values are finite."
                if rolling_finite
                else "Infinite rolling correlation values detected."
            ),
        )

        rolling_bounds = bool(
            (
                (non_missing >= -1.0 - TOLERANCE)
                & (non_missing <= 1.0 + TOLERANCE)
            ).all()
        )

        record(
            results,
            dataset,
            "Rolling correlation bounds",
            rolling_bounds,
            (
                "All rolling correlations lie within [-1, 1]."
                if rolling_bounds
                else (
                    "Rolling correlations outside "
                    "[-1, 1] detected."
                )
            ),
        )

    return results


# ---------------------------------------------------------------------
# MAIN VALIDATION PIPELINE
# ---------------------------------------------------------------------

def main() -> None:

    all_results: list[dict[str, str]] = []

    for dataset, expected_dimension in DATASETS.items():

        print("\n" + "=" * 78)
        print(f"VALIDATING: {dataset.upper()}")
        print("=" * 78)

        results = validate_dataset(
            dataset,
            expected_dimension,
        )

        all_results.extend(results)

        for result in results:

            print(
                f"{result['validation']}: "
                f"{result['status']} | "
                f"{result['details']}"
            )

        failures = sum(
            result["status"] == "FAIL"
            for result in results
        )

        print(
            "\nDataset status:",
            (
                "PASS"
                if failures == 0
                else f"FAIL ({failures} failures)"
            ),
        )

    # -----------------------------------------------------------------
    # Consolidated validation report
    # -----------------------------------------------------------------

    summary = pd.DataFrame(
        all_results
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        TABLE_DIR
        / "correlation_validation_summary.csv"
    )

    parquet_path = (
        TABLE_DIR
        / "correlation_validation_summary.parquet"
    )

    summary.to_csv(
        csv_path,
        index=False,
    )

    summary.to_parquet(
        parquet_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 78)
    print("CORRELATION VALIDATION SUMMARY")
    print("=" * 78)

    pivot = summary.pivot(
        index="validation",
        columns="dataset",
        values="status",
    )

    print(
        pivot.fillna("N/A").to_string()
    )

    failures = summary[
        summary["status"] == "FAIL"
    ]

    passed_count = int(
        (summary["status"] == "PASS").sum()
    )

    failed_count = len(failures)

    total_checks = len(summary)

    print("\n" + "-" * 78)

    print(f"Total checks : {total_checks}")
    print(f"Passed       : {passed_count}")
    print(f"Failed       : {failed_count}")

    print("\nValidation reports saved:")

    print(f"CSV     : {csv_path}")
    print(f"Parquet : {parquet_path}")

    if failed_count > 0:

        print("\nFAILED VALIDATION CHECKS:")

        print(
            failures[
                ["dataset", "validation", "details"]
            ].to_string(index=False)
        )

        raise SystemExit(
            "\nCORRELATION VALIDATION FAILED."
        )

    print(
        "\nALL CORRELATION VALIDATION CHECKS PASSED."
    )


if __name__ == "__main__":
    main()