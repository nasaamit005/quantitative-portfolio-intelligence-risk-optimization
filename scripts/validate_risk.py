
"""Validate historical risk analytics and generate validation summaries."""

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

CORE_METRICS = [
    "annualized_volatility",
    "downside_deviation",
    "maximum_drawdown",
    "historical_var_95",
    "historical_cvar_95",
    "historical_var_99",
    "historical_cvar_99",
]


def validate_dataset(
    dataset: str,
    expected_instruments: int,
) -> list[dict[str, str | int]]:
    """Validate risk outputs and return validation records."""

    path = TABLES_DIR / f"{dataset}_risk_summary.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    risk = pd.read_parquet(path)

    returns_path = (
        PROCESSED_DIR / f"{dataset}_simple_returns.parquet"
    )

    returns = pd.read_parquet(returns_path).dropna(how="all")

    records = []

    def record(
        check: str,
        passed: bool,
        detail: str,
    ) -> None:
        records.append({
            "dataset": dataset,
            "validation": check,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

        if not passed:
            raise ValueError(
                f"{dataset}: validation failed: {check}. {detail}"
            )

    record(
        "Instrument count",
        len(risk) == expected_instruments,
        f"Expected {expected_instruments}; found {len(risk)}.",
    )

    record(
        "Ticker consistency",
        set(risk.index) == set(returns.columns),
        "Risk report tickers match return dataset tickers.",
    )

    record(
        "Observation counts",
        (risk["observations"] == len(returns)).all(),
        f"Expected {len(returns)} observations per instrument.",
    )

    record(
        "Required risk metrics",
        set(CORE_METRICS).issubset(risk.columns),
        "All required risk metric columns are present.",
    )

    numeric = risk[CORE_METRICS].to_numpy()

    record(
        "Finite risk metrics",
        np.isfinite(numeric).all(),
        "No NaN or infinite values in core risk metrics.",
    )

    record(
        "Non-negative volatility",
        (risk["annualized_volatility"] >= 0).all(),
        "Annualized volatility is non-negative.",
    )

    record(
        "Non-negative downside deviation",
        (risk["downside_deviation"] >= 0).all(),
        "Downside deviation is non-negative.",
    )

    record(
        "Drawdown bounds",
        (
            (risk["maximum_drawdown"] <= 0).all()
            and (risk["maximum_drawdown"] >= -1).all()
        ),
        "Maximum drawdown lies between -100% and 0%.",
    )

    record(
        "VaR and CVaR ordering",
        (
            (risk["historical_cvar_95"] >= risk["historical_var_95"]).all()
            and (risk["historical_cvar_99"] >= risk["historical_var_99"]).all()
        ),
        "CVaR is at least as large as VaR at each confidence level.",
    )

    record(
        "Confidence-level ordering",
        (
            (risk["historical_var_99"] >= risk["historical_var_95"]).all()
            and (risk["historical_cvar_99"] >= risk["historical_cvar_95"]).all()
        ),
        "99% VaR and CVaR are at least as large as their 95% equivalents.",
    )

    if dataset == "us_equities":
        relative_metrics = [
            "beta_vs_spy",
            "tracking_error_vs_spy",
            "information_ratio_vs_spy",
        ]

        record(
            "Benchmark-relative metrics",
            set(relative_metrics).issubset(risk.columns)
            and np.isfinite(
                risk[relative_metrics].to_numpy()
            ).all(),
            "Beta, tracking error and information ratio are present.",
        )

    print(f"\n{dataset.upper()}")
    print("-" * 60)

    for item in records:
        print(
            f"{item['validation']}: {item['status']} "
            f"| {item['detail']}"
        )

    print("Status: PASS")

    return records


def main() -> None:
    """Validate both risk datasets and save a consolidated report."""

    all_records = []

    for dataset, expected_count in DATASETS.items():
        all_records.extend(
            validate_dataset(dataset, expected_count)
        )

    validation_table = pd.DataFrame(all_records)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    validation_table.to_csv(
        TABLES_DIR / "risk_validation_summary.csv",
        index=False,
    )

    validation_table.to_parquet(
        TABLES_DIR / "risk_validation_summary.parquet",
        index=False,
    )

    print("\n" + "=" * 65)
    print("RISK VALIDATION SUMMARY")
    print("=" * 65)

    print(
        validation_table.pivot(
            index="validation",
            columns="dataset",
            values="status",
        ).to_string()
    )

    print("\nValidation report saved:")
    print(TABLES_DIR / "risk_validation_summary.csv")

    print("\nALL RISK VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
