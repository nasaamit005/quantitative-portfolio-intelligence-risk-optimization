
"""Validate statistical analysis outputs."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "reports" / "tables"

DATASETS = {
    "us_equities": 12,
    "us_benchmarks": 3,
}

ROLLING_CHECKS = {
    "rolling_21d_volatility": 20,
    "rolling_63d_volatility": 62,
    "rolling_252d_return": 251,
}

SUMMARY_COLUMNS = [
    "observations",
    "annualized_arithmetic_return",
    "annualized_cagr",
    "annualized_volatility",
    "skewness",
    "excess_kurtosis",
    "jarque_bera_statistic",
    "jarque_bera_p_value",
]


def validate_dataset(
    dataset: str,
    expected_instruments: int,
) -> None:
    """Validate summary, quantiles and rolling metric outputs."""

    summary_path = (
        TABLES_DIR / f"{dataset}_statistical_summary.parquet"
    )

    quantile_path = (
        TABLES_DIR / f"{dataset}_return_quantiles.parquet"
    )

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    if not quantile_path.exists():
        raise FileNotFoundError(quantile_path)

    summary = pd.read_parquet(summary_path)
    quantiles = pd.read_parquet(quantile_path)

    returns = pd.read_parquet(
        PROCESSED_DIR / f"{dataset}_simple_returns.parquet"
    ).dropna(how="all")

    # Check instrument counts.
    if len(summary) != expected_instruments:
        raise ValueError(
            f"{dataset}: unexpected number of summary instruments."
        )

    if len(quantiles.columns) != expected_instruments:
        raise ValueError(
            f"{dataset}: unexpected number of quantile instruments."
        )

    if set(summary.index) != set(returns.columns):
        raise ValueError(
            f"{dataset}: summary tickers do not match return tickers."
        )

    if set(quantiles.columns) != set(returns.columns):
        raise ValueError(
            f"{dataset}: quantile tickers do not match return tickers."
        )

    # Check required summary metrics.
    missing_columns = (
        set(SUMMARY_COLUMNS) - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset}: missing summary columns: {missing_columns}"
        )

    if not (summary["observations"] == len(returns)).all():
        raise ValueError(
            f"{dataset}: observation counts are incorrect."
        )

    numeric_summary = summary.select_dtypes(include=[np.number])

    if not np.isfinite(numeric_summary.to_numpy()).all():
        raise ValueError(
            f"{dataset}: non-finite summary values detected."
        )

    if (summary["annualized_volatility"] < 0).any():
        raise ValueError(
            f"{dataset}: negative annualized volatility detected."
        )

    # Check quantile ordering.
    expected_quantiles = [
        "Q01", "Q05", "Q10", "Q25", "Q50",
        "Q75", "Q90", "Q95", "Q99",
    ]

    if list(quantiles.index) != expected_quantiles:
        raise ValueError(
            f"{dataset}: unexpected quantile labels or order."
        )

    if not np.isfinite(quantiles.to_numpy()).all():
        raise ValueError(
            f"{dataset}: non-finite quantile values detected."
        )

    if (quantiles.diff().iloc[1:] < -1e-12).any().any():
        raise ValueError(
            f"{dataset}: return quantiles are not monotonic."
        )

    # Check rolling metrics and expected warm-up periods.
    for metric, initial_nan_count in ROLLING_CHECKS.items():

        path = PROCESSED_DIR / f"{dataset}_{metric}.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        rolling = pd.read_parquet(path)

        if not rolling.index.equals(returns.index):
            raise ValueError(
                f"{dataset}: {metric} date index mismatch."
            )

        if not rolling.columns.equals(returns.columns):
            raise ValueError(
                f"{dataset}: {metric} instrument mismatch."
            )

        if not rolling.iloc[:initial_nan_count].isna().all().all():
            raise ValueError(
                f"{dataset}: incorrect initial rolling warm-up period "
                f"for {metric}."
            )

        if rolling.iloc[initial_nan_count:].isna().any().any():
            raise ValueError(
                f"{dataset}: unexpected missing rolling values "
                f"for {metric}."
            )

        if not np.isfinite(
            rolling.iloc[initial_nan_count:].to_numpy()
        ).all():
            raise ValueError(
                f"{dataset}: non-finite rolling values in {metric}."
            )

    print(f"\n{dataset.upper()}")
    print("-" * 60)
    print("Instruments:", len(summary))
    print("Valid observations per instrument:", len(returns))
    print("Statistical summary: PASS")
    print("Return quantiles: PASS")
    print("Rolling metrics: PASS")
    print("Status: PASS")


def main() -> None:
    """Validate all statistical datasets."""

    for dataset, expected_count in DATASETS.items():
        validate_dataset(dataset, expected_count)

    print("\nALL STATISTICAL VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
