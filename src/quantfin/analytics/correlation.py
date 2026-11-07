
"""Correlation, covariance, and diversification analytics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.covariance import LedoitWolf

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"

TRADING_DAYS = 252
ROLLING_WINDOW = 63
BENCHMARK_TICKER = "SPY"

DATASETS = ("us_equities", "us_benchmarks")


def load_returns(dataset: str) -> pd.DataFrame:
    """Load and validate a wide-format simple-return dataset."""
    path = PROCESSED_DIR / f"{dataset}_simple_returns.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Return dataset not found: {path}")

    returns = pd.read_parquet(path)

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError(f"Return dataset is empty or invalid: {path}")

    if not returns.index.is_unique:
        raise ValueError(f"Duplicate dates found in {path}")

    if not returns.columns.is_unique:
        raise ValueError(f"Duplicate instrument columns found in {path}")

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")

    # Use a common observation window for all pairwise estimates.
    returns = returns.dropna(axis=0, how="any")

    if len(returns) < 2:
        raise ValueError(f"Insufficient complete observations in {path}")

    if (returns <= -1.0).any().any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    return returns.sort_index()


def calculate_correlation_matrices(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson and Spearman correlation matrices."""
    return returns.corr(method="pearson"), returns.corr(method="spearman")


def calculate_covariance_matrices(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate sample, annualized, and Ledoit-Wolf covariance matrices."""
    sample_covariance = returns.cov()
    annualized_covariance = sample_covariance * TRADING_DAYS

    estimator = LedoitWolf()
    estimator.fit(returns.to_numpy())

    shrinkage_covariance = pd.DataFrame(
        estimator.covariance_,
        index=returns.columns,
        columns=returns.columns,
    )

    annualized_shrinkage_covariance = (
        shrinkage_covariance * TRADING_DAYS
    )

    return (
        sample_covariance,
        annualized_covariance,
        annualized_shrinkage_covariance,
    )


def calculate_diversification_metrics(
    returns: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate equal-weight portfolio diversification statistics."""
    weights = np.full(returns.shape[1], 1.0 / returns.shape[1])

    asset_volatility = returns.std(ddof=1).to_numpy() * np.sqrt(
        TRADING_DAYS
    )

    covariance = returns.cov().to_numpy() * TRADING_DAYS
    portfolio_volatility = float(
        np.sqrt(weights @ covariance @ weights)
    )

    weighted_average_volatility = float(weights @ asset_volatility)

    diversification_ratio = (
        weighted_average_volatility / portfolio_volatility
        if portfolio_volatility > 0
        else np.nan
    )

    correlation = returns.corr().to_numpy()
    upper_triangle = correlation[np.triu_indices_from(correlation, k=1)]

    average_pairwise_correlation = (
        float(np.mean(upper_triangle))
        if upper_triangle.size
        else np.nan
    )

    return {
        "instrument_count": int(returns.shape[1]),
        "observations": int(returns.shape[0]),
        "equal_weight_portfolio_volatility": portfolio_volatility,
        "weighted_average_asset_volatility": weighted_average_volatility,
        "diversification_ratio": float(diversification_ratio),
        "average_pairwise_correlation": average_pairwise_correlation,
    }


def calculate_matrix_diagnostics(
    matrix: pd.DataFrame,
    matrix_name: str,
) -> dict[str, object]:
    """Check symmetry, eigenvalues, positive definiteness, and conditioning."""
    values = matrix.to_numpy(dtype=float)
    symmetric = bool(np.allclose(values, values.T, atol=1e-10))

    eigenvalues = np.linalg.eigvalsh(values)
    minimum_eigenvalue = float(eigenvalues.min())
    positive_definite = bool(minimum_eigenvalue > 0)

    condition_number = float(np.linalg.cond(values))

    return {
        "matrix": matrix_name,
        "dimension": int(values.shape[0]),
        "symmetric": symmetric,
        "minimum_eigenvalue": minimum_eigenvalue,
        "positive_definite": positive_definite,
        "condition_number": condition_number,
        "finite_values": bool(np.isfinite(values).all()),
    }


def save_matrix(matrix: pd.DataFrame, name: str) -> None:
    """Save a matrix in CSV and Parquet formats."""
    matrix.to_csv(TABLE_DIR / f"{name}.csv", index=True)
    matrix.to_parquet(TABLE_DIR / f"{name}.parquet", index=True)


def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    filename: str,
    *,
    center: float | None = None,
    correlation: bool = False,
) -> None:
    """Save a readable heatmap for an analytical matrix."""
    size = max(9, min(18, len(matrix.columns) * 0.85))

    plt.figure(figsize=(size, size * 0.78))

    sns.heatmap(
        matrix,
        cmap="RdBu_r" if correlation else "viridis",
        center=center,
        vmin=-1 if correlation else None,
        vmax=1 if correlation else None,
        annot=len(matrix.columns) <= 15,
        fmt=".2f",
        linewidths=0.35,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(title, fontsize=15, pad=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_rolling_correlations(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rolling 63-day correlations against SPY."""
    if BENCHMARK_TICKER not in returns.columns:
        LOGGER.warning(
            "SPY is not present. Rolling benchmark correlations skipped."
        )
        return pd.DataFrame(index=returns.index)

    benchmark = returns[BENCHMARK_TICKER]
    rolling = pd.DataFrame(index=returns.index)

    for ticker in returns.columns:
        if ticker == BENCHMARK_TICKER:
            continue

        rolling[f"{ticker}_vs_SPY"] = (
            returns[ticker]
            .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
            .corr(benchmark)
        )

    return rolling


def calculate_correlation_stability(
    rolling_correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the distribution and stability of rolling correlations."""
    records: list[dict[str, object]] = []

    for column in rolling_correlations.columns:
        values = rolling_correlations[column].dropna()

        if values.empty:
            continue

        records.append(
            {
                "comparison": column,
                "observations": int(values.count()),
                "mean_rolling_correlation": float(values.mean()),
                "std_rolling_correlation": float(values.std()),
                "minimum_rolling_correlation": float(values.min()),
                "maximum_rolling_correlation": float(values.max()),
                "latest_rolling_correlation": float(values.iloc[-1]),
            }
        )

    return pd.DataFrame(records).set_index("comparison") if records else pd.DataFrame()


def analyze_dataset(dataset: str) -> None:
    """Run and save the complete correlation analysis for one dataset."""
    LOGGER.info("Starting correlation analysis: %s", dataset)

    returns = load_returns(dataset)

    pearson, spearman = calculate_correlation_matrices(returns)

    (
        sample_covariance,
        annualized_covariance,
        ledoit_wolf_covariance,
    ) = calculate_covariance_matrices(returns)

    prefix = dataset

    matrices = {
        f"{prefix}_pearson_correlation_matrix": pearson,
        f"{prefix}_spearman_correlation_matrix": spearman,
        f"{prefix}_sample_covariance_matrix": sample_covariance,
        f"{prefix}_annualized_covariance_matrix": annualized_covariance,
        f"{prefix}_ledoit_wolf_covariance_matrix": ledoit_wolf_covariance,
    }

    for name, matrix in matrices.items():
        save_matrix(matrix, name)

    diagnostics = pd.DataFrame(
        [
            calculate_matrix_diagnostics(pearson, "pearson_correlation"),
            calculate_matrix_diagnostics(spearman, "spearman_correlation"),
            calculate_matrix_diagnostics(sample_covariance, "sample_covariance"),
            calculate_matrix_diagnostics(
                annualized_covariance, "annualized_covariance"
            ),
            calculate_matrix_diagnostics(
                ledoit_wolf_covariance, "ledoit_wolf_covariance"
            ),
        ]
    ).set_index("matrix")

    diagnostics.to_csv(
        TABLE_DIR / f"{prefix}_matrix_diagnostics.csv"
    )
    diagnostics.to_parquet(
        TABLE_DIR / f"{prefix}_matrix_diagnostics.parquet"
    )

    diversification = pd.DataFrame(
        [calculate_diversification_metrics(returns)],
        index=[dataset],
    )

    diversification.to_csv(
        TABLE_DIR / f"{prefix}_diversification_metrics.csv"
    )
    diversification.to_parquet(
        TABLE_DIR / f"{prefix}_diversification_metrics.parquet"
    )

    rolling = calculate_rolling_correlations(returns)
    stability = calculate_correlation_stability(rolling)

    if not rolling.empty:
        rolling.to_csv(
            PROCESSED_DIR / f"{prefix}_rolling_63d_correlations.csv"
        )
        rolling.to_parquet(
            PROCESSED_DIR / f"{prefix}_rolling_63d_correlations.parquet"
        )

    if not stability.empty:
        stability.to_csv(
            TABLE_DIR / f"{prefix}_correlation_stability.csv"
        )
        stability.to_parquet(
            TABLE_DIR / f"{prefix}_correlation_stability.parquet"
        )

        plt.figure(figsize=(14, 7))
        for column in rolling.columns:
            plt.plot(rolling.index, rolling[column], label=column, linewidth=1)

        plt.axhline(0, linewidth=0.8, linestyle="--")
        plt.axhline(1, linewidth=0.5, linestyle=":")
        plt.axhline(-1, linewidth=0.5, linestyle=":")
        plt.ylim(-1.05, 1.05)
        plt.title(f"{dataset}: 63-Day Rolling Correlation Against SPY")
        plt.xlabel("Date")
        plt.ylabel("Rolling correlation")
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"{prefix}_rolling_63d_correlation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    save_heatmap(
        pearson,
        f"{dataset}: Pearson Correlation",
        f"{prefix}_pearson_correlation_heatmap.png",
        center=0,
        correlation=True,
    )
    save_heatmap(
        spearman,
        f"{dataset}: Spearman Rank Correlation",
        f"{prefix}_spearman_correlation_heatmap.png",
        center=0,
        correlation=True,
    )
    save_heatmap(
        annualized_covariance,
        f"{dataset}: Annualized Covariance",
        f"{prefix}_annualized_covariance_heatmap.png",
    )
    save_heatmap(
        ledoit_wolf_covariance * TRADING_DAYS,
        f"{dataset}: Annualized Ledoit-Wolf Covariance",
        f"{prefix}_ledoit_wolf_covariance_heatmap.png",
    )

    print(f"\nCORRELATION ANALYSIS: {dataset.upper()}")
    print("=" * 78)
    print(f"Observations: {len(returns):,}")
    print(f"Instruments:  {len(returns.columns)}")
    print("\nDiversification metrics:")
    print(diversification.round(4).to_string())
    print("\nMatrix diagnostics:")
    print(diagnostics.round(6).to_string())
    print("\nPearson correlation matrix:")
    print(pearson.round(3).to_string())

    LOGGER.info("Correlation analysis completed: %s", dataset)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        analyze_dataset(dataset)

    print("\nALL CORRELATION ANALYSES COMPLETED.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
