"""Generate 63-day rolling correlations of US equities against SPY."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"

WINDOW = 63

EQUITY_FILE = (
    PROCESSED_DIR / "us_equities_simple_returns.parquet"
)

BENCHMARK_FILE = (
    PROCESSED_DIR / "us_benchmarks_simple_returns.parquet"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "us_equities_rolling_63d_correlations.parquet"
)

OUTPUT_CSV = (
    PROCESSED_DIR / "us_equities_rolling_63d_correlations.csv"
)


def main() -> None:
    """Calculate and save rolling equity correlations against SPY."""

    if not EQUITY_FILE.exists():
        raise FileNotFoundError(
            f"Equity returns file not found: {EQUITY_FILE}"
        )

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark returns file not found: {BENCHMARK_FILE}"
        )

    equities = pd.read_parquet(EQUITY_FILE)
    benchmarks = pd.read_parquet(BENCHMARK_FILE)

    # Ensure datetime indexes.
    equities.index = pd.to_datetime(equities.index)
    benchmarks.index = pd.to_datetime(benchmarks.index)

    # Remove duplicate dates and sort chronologically.
    equities = equities.loc[
        ~equities.index.duplicated(keep="last")
    ].sort_index()

    benchmarks = benchmarks.loc[
        ~benchmarks.index.duplicated(keep="last")
    ].sort_index()

    if "SPY" not in benchmarks.columns:
        raise KeyError(
            "SPY was not found in benchmark return columns. "
            f"Available columns: {benchmarks.columns.tolist()}"
        )

    # Convert return columns to numeric.
    equities = equities.apply(
        pd.to_numeric,
        errors="coerce",
    )

    spy = pd.to_numeric(
        benchmarks["SPY"],
        errors="coerce",
    ).rename("SPY")

    # Align equity and benchmark dates.
    aligned = equities.join(
        spy,
        how="inner",
    )

    equity_columns = [
        column
        for column in equities.columns
        if column in aligned.columns
    ]

    if not equity_columns:
        raise ValueError(
            "No equity return columns available for correlation."
        )

    # Calculate rolling 63-trading-day correlations.
    rolling_correlations = (
        aligned[equity_columns]
        .rolling(
            window=WINDOW,
            min_periods=WINDOW,
        )
        .corr(aligned["SPY"])
    )

    rolling_correlations.index.name = "Date"

    # Save outputs.
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rolling_correlations.to_parquet(
        OUTPUT_FILE
    )

    rolling_correlations.to_csv(
        OUTPUT_CSV
    )

    # Summary.
    valid_counts = rolling_correlations.count()

    print("\n63-DAY EQUITY ROLLING CORRELATIONS")
    print("-" * 65)

    print(f"Equities processed : {len(equity_columns)}")
    print(f"Aligned dates      : {len(aligned)}")
    print(f"Rolling window     : {WINDOW} trading days")
    print(
        f"First valid date   : "
        f"{rolling_correlations.dropna(how='all').index.min()}"
    )
    print(
        f"Last valid date    : "
        f"{rolling_correlations.dropna(how='all').index.max()}"
    )

    print("\nValid observations per equity:")
    print(valid_counts.to_string())

    print("\nOutput files:")
    print(OUTPUT_FILE)
    print(OUTPUT_CSV)

    print("\nEQUITY ROLLING CORRELATIONS GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()