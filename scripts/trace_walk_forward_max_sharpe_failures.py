
"""Trace maximum-Sharpe failures across actual walk-forward estimation windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantfin.backtesting.engine import (
    _rebalance_dates,
    _validate_returns,
)
from quantfin.strategies import maximum_sharpe_strategy

ROOT = Path(__file__).resolve().parents[1]

RETURNS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "us_equities_simple_returns.csv"
)

OUTPUT_DIR = ROOT / "reports" / "robustness_analysis"
OUTPUT_PATH = OUTPUT_DIR / "max_sharpe_failure_trace.csv"

ASSETS = [
    "AAPL", "MSFT", "NVDA", "GOOGL",
    "AMZN", "META", "JPM", "XOM",
    "JNJ", "COST", "CAT", "KO",
]

RISK_FREE_RATE = 0.04
REBALANCE_FREQUENCY = "M"

EXPERIMENTS = [
    {
        "name": "126d_max_weight_050",
        "lookback_days": 126,
        "max_weight": 0.50,
    },
    {
        "name": "252d_max_weight_030",
        "lookback_days": 252,
        "max_weight": 0.30,
    },
]


def load_returns() -> pd.DataFrame:
    """Load the same processed return dataset as the robustness analysis."""

    if not RETURNS_PATH.exists():
        raise FileNotFoundError(RETURNS_PATH)

    frame = pd.read_csv(RETURNS_PATH)

    date_columns = [
        column
        for column in frame.columns
        if str(column).strip().lower() == "date"
    ]

    if not date_columns:
        raise ValueError("Dataset must contain a Date column.")

    date_column = date_columns[0]

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    if frame[date_column].isna().any():
        raise ValueError("Invalid dates found in the dataset.")

    frame = (
        frame.rename(columns={date_column: "date"})
        .set_index("date")
        .sort_index()
    )

    missing = sorted(set(ASSETS).difference(frame.columns))

    if missing:
        raise ValueError(f"Missing assets: {missing}")

    frame = frame[ASSETS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    frame = frame.dropna(how="all").dropna(subset=ASSETS)

    return _validate_returns(frame)


def inspect_window(
    history: pd.DataFrame,
) -> dict[str, float]:
    """Calculate diagnostics for the exact optimizer input."""

    expected_returns = history.mean() * 252.0
    covariance = history.cov() * 252.0

    eigenvalues = np.linalg.eigvalsh(
        covariance.to_numpy(dtype=float)
    )

    minimum_eigenvalue = float(eigenvalues.min())
    maximum_eigenvalue = float(eigenvalues.max())

    condition_number = (
        maximum_eigenvalue / minimum_eigenvalue
        if minimum_eigenvalue > 0
        else np.inf
    )

    return {
        "expected_return_min": float(expected_returns.min()),
        "expected_return_max": float(expected_returns.max()),
        "covariance_min_eigenvalue": minimum_eigenvalue,
        "covariance_max_eigenvalue": maximum_eigenvalue,
        "covariance_condition_number": float(condition_number),
    }


def trace_experiment(
    returns: pd.DataFrame,
    experiment: dict,
) -> tuple[list[dict], int, int]:
    """Run the production strategy on every actual estimation window."""

    name = experiment["name"]
    lookback = experiment["lookback_days"]
    max_weight = experiment["max_weight"]

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)

    first_test_position = lookback

    if len(returns) <= first_test_position:
        raise ValueError(
            f"Insufficient data for {name}."
        )

    test_returns = returns.iloc[first_test_position:]

    rebalance_dates = _rebalance_dates(
        returns.index,
        REBALANCE_FREQUENCY,
    )

    attempts: list[dict] = []
    success_count = 0
    failure_count = 0

    # Match the production backtest's initial estimation window.
    initial_history = returns.iloc[
        first_test_position - lookback:first_test_position
    ]

    windows = [
        (
            "initial",
            test_returns.index[0],
            initial_history,
        )
    ]

    # Match the production backtest's monthly rebalance windows.
    for date in test_returns.index:
        if date not in rebalance_dates:
            continue

        history_end_position = returns.index.get_loc(date)
        history_start_position = history_end_position - lookback

        if history_start_position < 0:
            raise ValueError(
                f"Insufficient history before {date}."
            )

        estimation_returns = returns.iloc[
            history_start_position:history_end_position
        ]

        windows.append(
            (
                "monthly_rebalance",
                date,
                estimation_returns,
            )
        )

    for attempt_number, (window_type, as_of_date, history) in enumerate(
        windows,
        start=1,
    ):
        diagnostics = inspect_window(history)

        record = {
            "experiment": name,
            "attempt_number": attempt_number,
            "window_type": window_type,
            "as_of_date": pd.Timestamp(as_of_date),
            "history_start": history.index.min(),
            "history_end": history.index.max(),
            "observations": len(history),
            "max_weight": max_weight,
            "risk_free_rate": RISK_FREE_RATE,
            **diagnostics,
        }

        try:
            weights = maximum_sharpe_strategy(
                returns=history,
                risk_free_rate=RISK_FREE_RATE,
                max_weight=max_weight,
            )

            record.update({
                "status": "success",
                "error": None,
                "weight_sum": float(weights.sum()),
                "minimum_weight": float(weights.min()),
                "maximum_weight": float(weights.max()),
            })

            success_count += 1

        except Exception as exc:
            record.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "weight_sum": np.nan,
                "minimum_weight": np.nan,
                "maximum_weight": np.nan,
            })

            failure_count += 1

            print("\nOPTIMIZER FAILURE")
            print(f"  Attempt:       {attempt_number}")
            print(f"  Window type:   {window_type}")
            print(f"  As-of date:    {pd.Timestamp(as_of_date).date()}")
            print(f"  History start: {history.index.min().date()}")
            print(f"  History end:   {history.index.max().date()}")
            print(f"  Observations:  {len(history)}")
            print(f"  Maximum cap:   {max_weight:.2f}")
            print(f"  Error:         {record['error']}")

        attempts.append(record)

    print(f"\nTotal optimizer calls: {len(attempts)}")
    print(f"Successful calls:     {success_count}")
    print(f"Failed calls:         {failure_count}")

    return attempts, success_count, failure_count


def main() -> None:
    print("=" * 80)
    print("WALK-FORWARD MAXIMUM-SHARPE FAILURE TRACE")
    print("=" * 80)

    returns = load_returns()

    print(f"Assets: {len(returns.columns)}")
    print(f"Observations: {len(returns):,}")
    print(
        f"Dataset period: {returns.index.min().date()} "
        f"to {returns.index.max().date()}"
    )

    all_records = []

    for experiment in EXPERIMENTS:
        records, _, _ = trace_experiment(
            returns=returns,
            experiment=experiment,
        )
        all_records.extend(records)

    results = pd.DataFrame(all_records)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    failures = results[
        results["status"] == "failed"
    ]

    print("\n" + "=" * 80)
    print("TRACE SUMMARY")
    print("=" * 80)

    print(f"Total estimation windows: {len(results)}")
    print(f"Successful windows: {(results['status'] == 'success').sum()}")
    print(f"Failed windows: {len(failures)}")

    if not failures.empty:
        print("\nFailed estimation windows:")
        print(
            failures[
                [
                    "experiment",
                    "window_type",
                    "as_of_date",
                    "history_start",
                    "history_end",
                    "observations",
                    "max_weight",
                    "covariance_min_eigenvalue",
                    "covariance_condition_number",
                    "error",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nNo optimizer failures reproduced.")

    print(f"\nSaved trace: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
