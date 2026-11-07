
"""Validate calculated simple and logarithmic returns."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"

DATASETS = ["us_equities", "us_benchmarks"]

TOLERANCE = 1e-10


def validate_returns(name: str) -> None:
    """Check return files, missing observations and mathematical identity."""

    simple_path = DATA_DIR / f"{name}_simple_returns.parquet"
    log_path = DATA_DIR / f"{name}_log_returns.parquet"

    if not simple_path.exists() or not log_path.exists():
        raise FileNotFoundError(f"Return files missing for {name}.")

    simple = pd.read_parquet(simple_path)
    logarithmic = pd.read_parquet(log_path)

    if not simple.index.equals(logarithmic.index):
        raise ValueError(f"{name}: return dates do not match.")

    if not simple.columns.equals(logarithmic.columns):
        raise ValueError(f"{name}: instrument columns do not match.")

    if simple.index.has_duplicates:
        raise ValueError(f"{name}: duplicate dates detected.")

    if not simple.iloc[0].isna().all():
        raise ValueError(f"{name}: first simple return row should be NaN.")

    if not logarithmic.iloc[0].isna().all():
        raise ValueError(f"{name}: first log return row should be NaN.")

    later_simple = simple.iloc[1:]
    later_log = logarithmic.iloc[1:]

    if later_simple.isna().any().any():
        raise ValueError(f"{name}: unexpected missing simple returns.")

    if later_log.isna().any().any():
        raise ValueError(f"{name}: unexpected missing log returns.")

    if not np.isfinite(later_simple.to_numpy()).all():
        raise ValueError(f"{name}: non-finite simple returns detected.")

    if not np.isfinite(later_log.to_numpy()).all():
        raise ValueError(f"{name}: non-finite log returns detected.")

    if (later_simple <= -1).any().any():
        raise ValueError(f"{name}: simple return at or below -100%.")

    expected_log = np.log1p(later_simple)

    max_error = np.abs(
        expected_log.to_numpy() - later_log.to_numpy()
    ).max()

    if max_error > TOLERANCE:
        raise ValueError(
            f"{name}: simple/log return identity failed. "
            f"Maximum error: {max_error}"
        )

    print(f"\n{name.upper()}")
    print("-" * 55)
    print("Instruments:", len(simple.columns))
    print("Observations:", len(simple))
    print("Valid return observations:", len(simple) - 1)
    print("Initial missing values:", int(simple.iloc[0].isna().sum()))
    print("Maximum mathematical identity error:", f"{max_error:.3e}")
    print("Status: PASS")


def main() -> None:
    for dataset in DATASETS:
        validate_returns(dataset)

    print("\nALL RETURN VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
