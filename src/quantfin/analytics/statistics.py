
"""Statistical analysis of equity and benchmark returns.

Calculates annualized return, CAGR, volatility, distribution statistics,
normality diagnostics, quantiles and rolling performance measures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

TRADING_DAYS = 252

ROLLING_VOLATILITY_WINDOWS = [21, 63]
ROLLING_RETURN_WINDOW = 252

DATASETS = ["us_equities", "us_benchmarks"]

logger = logging.getLogger(__name__)


def load_returns(dataset: str) -> pd.DataFrame:
    """Load validated simple returns for a dataset."""

    path = PROCESSED_DIR / f"{dataset}_simple_returns.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Return dataset not found: {path}"
        )

    returns = pd.read_parquet(path)

    if returns.empty:
        raise ValueError(f"{dataset}: return dataset is empty.")

    if returns.index.has_duplicates:
        raise ValueError(f"{dataset}: duplicate dates detected.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(f"{dataset}: expected a DatetimeIndex.")

    return returns.sort_index()


def calculate_asset_statistics(
    returns: pd.Series,
) -> dict[str, float | int | str]:
    """Calculate descriptive and annualized statistics for one asset."""

    clean = returns.dropna().astype(float)

    if len(clean) < 2:
        raise ValueError(
            f"{returns.name}: insufficient observations."
        )

    values = clean.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            f"{returns.name}: infinite or invalid returns detected."
        )

    if (clean <= -1).any():
        raise ValueError(
            f"{returns.name}: returns at or below -100% detected."
        )

    observations = len(clean)

    arithmetic_daily_mean = float(clean.mean())
    daily_volatility = float(clean.std(ddof=1))

    annualized_arithmetic_return = (
        arithmetic_daily_mean * TRADING_DAYS
    )

    annualized_volatility = (
        daily_volatility * np.sqrt(TRADING_DAYS)
    )

    compounded_growth = float(np.prod(1.0 + values))

    cagr = (
        compounded_growth ** (TRADING_DAYS / observations) - 1.0
        if compounded_growth > 0
        else -1.0
    )

    skewness = float(stats.skew(values, bias=False))
    excess_kurtosis = float(
        stats.kurtosis(values, fisher=True, bias=False)
    )

    jb_result = stats.jarque_bera(values)

    return {
        "observations": observations,
        "start_date": clean.index.min().strftime("%Y-%m-%d"),
        "end_date": clean.index.max().strftime("%Y-%m-%d"),
        "daily_mean": arithmetic_daily_mean,
        "daily_volatility": daily_volatility,
        "annualized_arithmetic_return": annualized_arithmetic_return,
        "annualized_cagr": float(cagr),
        "annualized_volatility": float(annualized_volatility),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_statistic": float(jb_result.statistic),
        "jarque_bera_p_value": float(jb_result.pvalue),
        "minimum_daily_return": float(clean.min()),
        "maximum_daily_return": float(clean.max()),
        "best_daily_return": float(clean.max()),
        "worst_daily_return": float(clean.min()),
        "positive_return_days": int((clean > 0).sum()),
        "negative_return_days": int((clean < 0).sum()),
        "positive_day_percentage": float((clean > 0).mean() * 100),
    }


def calculate_return_quantiles(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate daily return distribution quantiles."""

    probabilities = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    quantiles = returns.quantile(probabilities)

    quantiles.index = [
        f"Q{int(probability * 100):02d}"
        for probability in probabilities
    ]

    return quantiles


def calculate_rolling_metrics(
    returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate rolling annualized volatility and rolling returns."""

    results = {}

    for window in ROLLING_VOLATILITY_WINDOWS:
        rolling_volatility = (
            returns
            .rolling(window=window, min_periods=window)
            .std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )

        results[f"rolling_{window}d_volatility"] = (
            rolling_volatility
        )

    rolling_return = (
        (1.0 + returns)
        .rolling(
            window=ROLLING_RETURN_WINDOW,
            min_periods=ROLLING_RETURN_WINDOW,
        )
        .apply(np.prod, raw=True)
        - 1.0
    )

    results["rolling_252d_return"] = rolling_return

    return results


def save_table(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save an analytical table in CSV and Parquet formats."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(TABLES_DIR / f"{filename}.csv")

    dataframe.to_parquet(
        TABLES_DIR / f"{filename}.parquet"
    )


def analyze_dataset(dataset: str) -> None:
    """Run the complete statistical analysis for one dataset."""

    logger.info("Starting statistical analysis: %s", dataset)

    returns = load_returns(dataset)

    # Exclude the expected initial NaN return row.
    clean_returns = returns.dropna(how="all")

    if clean_returns.isna().any().any():
        raise ValueError(
            f"{dataset}: unexpected missing return observations."
        )

    # 1. Descriptive and annualized statistics.
    statistics = {}

    for ticker in clean_returns.columns:
        statistics[ticker] = calculate_asset_statistics(
            clean_returns[ticker]
        )

    statistics_table = pd.DataFrame.from_dict(
        statistics,
        orient="index",
    )

    statistics_table.index.name = "ticker"

    save_table(
        statistics_table,
        f"{dataset}_statistical_summary",
    )

    # 2. Return distribution quantiles.
    quantiles = calculate_return_quantiles(clean_returns)

    save_table(
        quantiles,
        f"{dataset}_return_quantiles",
    )

    # 3. Rolling metrics.
    rolling_results = calculate_rolling_metrics(clean_returns)

    for metric_name, metric_data in rolling_results.items():
        metric_data.to_parquet(
            PROCESSED_DIR / f"{dataset}_{metric_name}.parquet"
        )

        metric_data.to_csv(
            PROCESSED_DIR / f"{dataset}_{metric_name}.csv"
        )

    # 4. Print a concise report.
    print("\n" + "=" * 75)
    print(f"STATISTICAL ANALYSIS: {dataset.upper()}")
    print("=" * 75)

    print("Instruments:", len(clean_returns.columns))
    print("Valid observations:", len(clean_returns))
    print(
        "Analysis period:",
        clean_returns.index.min().date(),
        "to",
        clean_returns.index.max().date(),
    )

    display_columns = [
        "annualized_arithmetic_return",
        "annualized_cagr",
        "annualized_volatility",
        "skewness",
        "excess_kurtosis",
        "jarque_bera_p_value",
    ]

    print("\nKey statistical measures:")
    print(
        statistics_table[display_columns].round(4).to_string()
    )

    print("\nReturn quantiles:")
    print(quantiles.round(4).to_string())

    print("\nSaved analytical outputs:")
    print(
        "-",
        TABLES_DIR / f"{dataset}_statistical_summary.csv",
    )
    print(
        "-",
        TABLES_DIR / f"{dataset}_return_quantiles.csv",
    )
    print(
        "- Rolling volatility and return files in data/processed/",
    )

    logger.info(
        "Statistical analysis completed: %s",
        dataset,
    )


def main() -> None:
    """Run statistical analysis across equities and benchmarks."""

    for dataset in DATASETS:
        analyze_dataset(dataset)

    print("\n" + "=" * 75)
    print("ALL STATISTICAL ANALYSES COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
