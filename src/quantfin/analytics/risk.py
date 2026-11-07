
"""Historical risk analytics for equities and benchmarks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "reports" / "tables"

TRADING_DAYS = 252
CONFIDENCE_LEVELS = [0.95, 0.99]
BENCHMARK_TICKER = "SPY"

DATASETS = ["us_equities", "us_benchmarks"]

logger = logging.getLogger(__name__)


def load_returns(dataset: str) -> pd.DataFrame:
    """Load validated simple daily returns."""

    path = PROCESSED_DIR / f"{dataset}_simple_returns.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    returns = pd.read_parquet(path)

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate return dates detected.")

    return returns.sort_index()


def calculate_historical_var(
    returns: pd.Series,
    confidence: float,
) -> float:
    """Calculate historical one-day VaR as a positive loss measure."""

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    return float(-returns.quantile(1.0 - confidence))


def calculate_historical_cvar(
    returns: pd.Series,
    confidence: float,
) -> float:
    """Calculate historical one-day CVaR as a positive loss measure."""

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    threshold = returns.quantile(1.0 - confidence)
    tail_returns = returns[returns <= threshold]

    if tail_returns.empty:
        return float("nan")

    return float(-tail_returns.mean())



def calculate_maximum_drawdown(
    returns: pd.Series,
) -> float:
    """Calculate maximum peak-to-trough drawdown, including initial wealth."""

    clean = returns.dropna().astype(float)

    if clean.empty:
        raise ValueError("Cannot calculate drawdown from empty returns.")

    wealth = (1.0 + clean).cumprod()

    # Include initial portfolio wealth before the first return.
    initial_date = clean.index[0] - pd.Timedelta(days=1)

    initial_wealth = pd.Series(
        [1.0],
        index=[initial_date],
    )

    wealth = pd.concat([initial_wealth, wealth])

    running_peak = wealth.cummax()
    drawdowns = wealth / running_peak - 1.0

    return float(drawdowns.min())


def calculate_downside_deviation(
    returns: pd.Series,
    target_return: float = 0.0,
) -> float:
    """Calculate annualized downside deviation."""

    daily_target = target_return / TRADING_DAYS

    downside_returns = np.minimum(
        returns.to_numpy() - daily_target,
        0.0,
    )

    return float(
        np.sqrt(np.mean(downside_returns ** 2))
        * np.sqrt(TRADING_DAYS)
    )


def calculate_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate beta relative to benchmark returns."""

    aligned = pd.concat(
        [asset_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        return float("nan")

    asset = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]

    variance = benchmark.var(ddof=1)

    if variance <= 0:
        return float("nan")

    return float(asset.cov(benchmark) / variance)


def calculate_tracking_error(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate annualized tracking error."""

    aligned = pd.concat(
        [asset_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        return float("nan")

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    return float(
        active_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )


def calculate_information_ratio(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate annualized information ratio."""

    aligned = pd.concat(
        [asset_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2:
        return float("nan")

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    tracking_error = (
        active_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    if tracking_error <= 0:
        return float("nan")

    annualized_active_return = (
        active_returns.mean() * TRADING_DAYS
    )

    return float(annualized_active_return / tracking_error)


def calculate_asset_risk(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float | int]:
    """Calculate historical risk statistics for one instrument."""

    clean = asset_returns.dropna().astype(float)

    if len(clean) < 2:
        raise ValueError(
            f"{asset_returns.name}: insufficient observations."
        )

    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError(
            f"{asset_returns.name}: invalid return observations."
        )

    if (clean <= -1).any():
        raise ValueError(
            f"{asset_returns.name}: return at or below -100%."
        )

    result: dict[str, float | int] = {
        "observations": len(clean),
        "annualized_volatility": float(
            clean.std(ddof=1) * np.sqrt(TRADING_DAYS)
        ),
        "downside_deviation": calculate_downside_deviation(clean),
        "maximum_drawdown": calculate_maximum_drawdown(clean),
    }

    for confidence in CONFIDENCE_LEVELS:
        label = int(confidence * 100)

        result[f"historical_var_{label}"] = (
            calculate_historical_var(clean, confidence)
        )

        result[f"historical_cvar_{label}"] = (
            calculate_historical_cvar(clean, confidence)
        )

    if benchmark_returns is not None:
        result["beta_vs_spy"] = calculate_beta(
            clean, benchmark_returns
        )

        result["tracking_error_vs_spy"] = calculate_tracking_error(
            clean, benchmark_returns
        )

        result["information_ratio_vs_spy"] = (
            calculate_information_ratio(clean, benchmark_returns)
        )

    return result


def save_risk_results(
    results: pd.DataFrame,
    dataset: str,
) -> None:
    """Save risk results in CSV and Parquet formats."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        TABLES_DIR / f"{dataset}_risk_summary.csv"
    )

    results.to_parquet(
        TABLES_DIR / f"{dataset}_risk_summary.parquet"
    )


def analyze_dataset(dataset: str) -> None:
    """Run risk analysis for one dataset."""

    logger.info("Starting risk analysis: %s", dataset)

    returns = load_returns(dataset)

    benchmark_returns = None

    if dataset == "us_equities":
        benchmark_data = load_returns("us_benchmarks")

        if BENCHMARK_TICKER not in benchmark_data.columns:
            raise ValueError(
                f"Benchmark {BENCHMARK_TICKER} not found."
            )

        benchmark_returns = benchmark_data[BENCHMARK_TICKER]

    results = {}

    for ticker in returns.columns:
        benchmark = None

        if dataset == "us_equities" and ticker != BENCHMARK_TICKER:
            benchmark = benchmark_returns

        results[ticker] = calculate_asset_risk(
            returns[ticker],
            benchmark,
        )

    risk_table = pd.DataFrame.from_dict(
        results,
        orient="index",
    )

    risk_table.index.name = "ticker"

    save_risk_results(risk_table, dataset)

    print("\n" + "=" * 75)
    print(f"RISK ANALYSIS: {dataset.upper()}")
    print("=" * 75)

    display_columns = [
        "annualized_volatility",
        "downside_deviation",
        "maximum_drawdown",
        "historical_var_95",
        "historical_cvar_95",
        "historical_var_99",
        "historical_cvar_99",
    ]

    print(risk_table[display_columns].round(4).to_string())

    if dataset == "us_equities":
        print("\nBenchmark-relative risk measures:")
        print(
            risk_table[
                [
                    "beta_vs_spy",
                    "tracking_error_vs_spy",
                    "information_ratio_vs_spy",
                ]
            ].round(4).to_string()
        )

    print("\nSaved risk report:")
    print(TABLES_DIR / f"{dataset}_risk_summary.csv")

    logger.info("Risk analysis completed: %s", dataset)


def main() -> None:
    """Run risk analytics for equities and benchmarks."""

    for dataset in DATASETS:
        analyze_dataset(dataset)

    print("\n" + "=" * 75)
    print("ALL RISK ANALYSES COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
