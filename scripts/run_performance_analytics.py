
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DAILY_RESULTS_PATH = (
    ROOT
    / "reports"
    / "walk_forward_comparison"
    / "strategy_daily_results.csv"
)

BENCHMARK_PATH = (
    ROOT
    / "data"
    / "processed"
    / "us_benchmarks_simple_returns.csv"
)

OUTPUT_DIR = ROOT / "reports" / "performance_analytics"
FIGURE_DIR = OUTPUT_DIR / "figures"

INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
ROLLING_WINDOW = 252


# ============================================================
# 2. INPUT LOADING AND VALIDATION
# ============================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate strategy and benchmark datasets."""

    if not DAILY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Strategy results not found: {DAILY_RESULTS_PATH}"
        )

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark data not found: {BENCHMARK_PATH}"
        )

    daily = pd.read_csv(
        DAILY_RESULTS_PATH,
        parse_dates=["date"],
    )

    benchmark = pd.read_csv(
        BENCHMARK_PATH,
        parse_dates=["Date"],
    )

    required_daily = {
        "date",
        "strategy",
        "portfolio_value",
        "gross_daily_return",
        "net_daily_return",
        "trading_cost",
        "turnover",
        "drawdown",
        "is_rebalance",
    }

    missing = required_daily.difference(daily.columns)

    if missing:
        raise ValueError(
            f"Missing strategy columns: {sorted(missing)}"
        )

    if not {"Date", "SPY"}.issubset(benchmark.columns):
        raise ValueError(
            "Benchmark file must contain Date and SPY columns."
        )

    numeric_columns = [
        "portfolio_value",
        "gross_daily_return",
        "net_daily_return",
        "trading_cost",
        "turnover",
        "drawdown",
    ]

    for column in numeric_columns:
        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce",
        )

    benchmark["SPY"] = pd.to_numeric(
        benchmark["SPY"],
        errors="coerce",
    )

    # Validate dates and strategy names.
    if daily["date"].isna().any():
        raise ValueError("Invalid dates found in strategy data.")

    if benchmark["Date"].isna().any():
        raise ValueError("Invalid dates found in benchmark data.")

    if daily["strategy"].isna().any():
        raise ValueError("Missing strategy names detected.")

    if daily["strategy"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty strategy names detected.")

    # Validate strategy numeric values.
    if daily[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values found in strategy data."
        )

    if not np.isfinite(
        daily[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values found in strategy data."
        )

    # IMPORTANT:
    # The original benchmark file can contain an initial NaN.
    # This is expected because the first return has no previous
    # trading-day price.
    #
    # Remove missing benchmark observations here.
    # The actual out-of-sample dates will be validated separately.
    benchmark = benchmark[["Date", "SPY"]].copy()

    benchmark = benchmark.dropna(subset=["SPY"])

    if not np.isfinite(
        benchmark["SPY"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite SPY returns found after removing missing values."
        )

    # Validate uniqueness.
    if daily.duplicated(["strategy", "date"]).any():
        raise ValueError(
            "Duplicate strategy/date observations detected."
        )

    if benchmark["Date"].duplicated().any():
        raise ValueError(
            "Duplicate benchmark dates detected."
        )

    # Validate portfolio values and returns.
    if (daily["portfolio_value"] <= 0).any():
        raise ValueError("Portfolio values must be positive.")

    if daily["net_daily_return"].le(-1).any():
        raise ValueError(
            "Net returns must be greater than -100%."
        )

    if daily["gross_daily_return"].le(-1).any():
        raise ValueError(
            "Gross returns must be greater than -100%."
        )

    if benchmark["SPY"].le(-1).any():
        raise ValueError(
            "SPY returns must be greater than -100%."
        )

    if daily["trading_cost"].lt(0).any():
        raise ValueError("Trading costs cannot be negative.")

    if daily["turnover"].lt(0).any():
        raise ValueError("Turnover cannot be negative.")

    if not daily["is_rebalance"].isin(
        [True, False, 0, 1]
    ).all():
        raise ValueError(
            "Invalid values found in is_rebalance."
        )

    daily["is_rebalance"] = daily["is_rebalance"].astype(bool)

    daily = (
        daily
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )

    benchmark = benchmark.rename(
        columns={
            "Date": "date",
            "SPY": "benchmark_return",
        }
    )

    benchmark = (
        benchmark
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        "Benchmark observations after removing initial missing values:",
        f"{len(benchmark):,}",
    )

    return daily, benchmark


# ============================================================
# 3. PERFORMANCE AND RISK METRICS
# ============================================================

def calculate_return_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate portfolio performance and risk metrics."""

    returns = returns.dropna().astype(float)

    if len(returns) < 2:
        raise ValueError(
            "At least two daily observations are required."
        )

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Non-finite portfolio returns detected.")

    if returns.le(-1).any():
        raise ValueError(
            "Portfolio returns must be greater than -100%."
        )

    daily_rf = (
        (1.0 + RISK_FREE_RATE)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    excess = returns - daily_rf

    # Annualized compounded return.
    annualized_return = (
        (1.0 + returns).prod()
        ** (TRADING_DAYS / len(returns))
        - 1.0
    )

    # Annualized volatility.
    daily_volatility = returns.std(ddof=1)

    annualized_volatility = (
        daily_volatility * np.sqrt(TRADING_DAYS)
    )

    # Sharpe ratio.
    sharpe = (
        excess.mean()
        / daily_volatility
        * np.sqrt(TRADING_DAYS)
        if daily_volatility > 0
        else np.nan
    )

    # Sortino ratio.
    downside = np.minimum(
        excess.to_numpy(),
        0.0,
    )

    downside_deviation = (
        np.sqrt(np.mean(downside**2))
        * np.sqrt(TRADING_DAYS)
    )

    sortino = (
        excess.mean() * TRADING_DAYS / downside_deviation
        if downside_deviation > 0
        else np.nan
    )

    # Maximum drawdown.
    # Include initial wealth before the first daily return.
    wealth = pd.concat(
        [
            pd.Series([1.0]),
            (1.0 + returns.reset_index(drop=True)),
        ],
        ignore_index=True,
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown = wealth / running_peak - 1.0

    max_drawdown = float(drawdown.min())

    # Calmar ratio.
    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    metrics = {
        "observations": int(len(returns)),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "best_daily_return": float(returns.max()),
        "worst_daily_return": float(returns.min()),
        "positive_day_ratio": float((returns > 0).mean()),
    }

    # Benchmark-relative analytics.
    if benchmark_returns is not None:

        aligned = pd.concat(
            [
                returns.rename("portfolio"),
                benchmark_returns.rename("benchmark"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < 2:
            raise ValueError(
                "Insufficient overlapping benchmark observations."
            )

        portfolio = aligned["portfolio"]
        benchmark = aligned["benchmark"]

        active = portfolio - benchmark

        benchmark_variance = benchmark.var(ddof=1)

        tracking_error = (
            active.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )

        beta = (
            portfolio.cov(benchmark) / benchmark_variance
            if benchmark_variance > 0
            else np.nan
        )

        information_ratio = (
            active.mean() * TRADING_DAYS / tracking_error
            if tracking_error > 0
            else np.nan
        )

        # Annualized CAPM alpha.
        alpha = (
            (
                portfolio.mean() - daily_rf
            )
            - beta * (
                benchmark.mean() - daily_rf
            )
        ) * TRADING_DAYS

        benchmark_annualized_return = (
            (1.0 + benchmark).prod()
            ** (TRADING_DAYS / len(benchmark))
            - 1.0
        )

        metrics.update({
            "benchmark_annualized_return": float(
                benchmark_annualized_return
            ),
            "tracking_error": float(tracking_error),
            "information_ratio": float(information_ratio),
            "beta_vs_spy": float(beta),
            "annualized_capm_alpha": float(alpha),
            "benchmark_correlation": float(
                portfolio.corr(benchmark)
            ),
            "benchmark_observations": int(len(aligned)),
        })

    return metrics


# ============================================================
# 4. DRAWDOWN EPISODE ANALYSIS
# ============================================================

def calculate_drawdown_periods(
    dates: pd.Series,
    portfolio_values: pd.Series,
    strategy: str,
) -> list[dict]:
    """Identify drawdown episodes, troughs and recovery dates."""

    values = portfolio_values.reset_index(drop=True).astype(float)

    dates = pd.to_datetime(
        dates
    ).reset_index(drop=True)

    if len(values) != len(dates):
        raise ValueError(
            "Dates and portfolio values must have equal lengths."
        )

    if values.empty:
        return []

    augmented_values = pd.concat(
        [
            pd.Series([INITIAL_CAPITAL]),
            values,
        ],
        ignore_index=True,
    )

    peaks = augmented_values.cummax()

    drawdowns = augmented_values / peaks - 1.0

    underwater = drawdowns < -1e-12

    episodes = []
    start = None

    for i, is_underwater in enumerate(underwater):

        if is_underwater and start is None:
            start = i

        is_last = i == len(underwater) - 1

        if start is not None and (
            not is_underwater or is_last
        ):

            end = (
                i
                if is_underwater and is_last
                else i - 1
            )

            segment = drawdowns.iloc[start:end + 1]

            trough = int(segment.idxmin())

            prior_peak_value = float(peaks.iloc[start])

            peak_candidates = augmented_values.iloc[
                :start + 1
            ]

            peak_index = int(
                np.flatnonzero(
                    np.isclose(
                        peak_candidates.to_numpy(),
                        prior_peak_value,
                    )
                )[-1]
            )

            recovered = (
                i
                if (
                    not is_underwater
                    and augmented_values.iloc[i] >= prior_peak_value
                )
                else None
            )

            peak_date = (
                pd.NaT
                if peak_index == 0
                else dates.iloc[peak_index - 1]
            )

            trough_date = dates.iloc[trough - 1]

            recovery_date = (
                dates.iloc[recovered - 1]
                if recovered is not None
                else pd.NaT
            )

            episodes.append({
                "strategy": strategy,
                "peak_date": peak_date,
                "peak_type": (
                    "Initial capital"
                    if peak_index == 0
                    else "Previous portfolio high"
                ),
                "trough_date": trough_date,
                "recovery_date": recovery_date,
                "drawdown": float(drawdowns.iloc[trough]),
                "days_to_trough": int(trough - peak_index),
                "days_to_recovery": (
                    int(recovered - peak_index)
                    if recovered is not None
                    else np.nan
                ),
                "recovered": recovered is not None,
            })

            start = None

    return episodes


# ============================================================
# 5. ROLLING PERFORMANCE ANALYSIS
# ============================================================

def build_rolling_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Calculate trailing 252-session performance metrics."""

    aligned = pd.concat(
        [
            returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    daily_rf = (
        (1.0 + RISK_FREE_RATE)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    portfolio = aligned["portfolio"]
    benchmark = aligned["benchmark"]

    rolling_mean = portfolio.rolling(
        ROLLING_WINDOW
    ).mean()

    rolling_std = portfolio.rolling(
        ROLLING_WINDOW
    ).std(ddof=1)

    rolling_sharpe = (
        (rolling_mean - daily_rf)
        / rolling_std
        * np.sqrt(TRADING_DAYS)
    )

    rolling_volatility = (
        rolling_std * np.sqrt(TRADING_DAYS)
    )

    rolling_covariance = (
        portfolio.rolling(ROLLING_WINDOW)
        .cov(benchmark)
    )

    rolling_benchmark_variance = (
        benchmark.rolling(ROLLING_WINDOW)
        .var(ddof=1)
    )

    rolling_beta = (
        rolling_covariance
        / rolling_benchmark_variance
    )

    return pd.DataFrame(
        {
            "rolling_annualized_volatility": rolling_volatility,
            "rolling_sharpe_ratio": rolling_sharpe,
            "rolling_beta_vs_spy": rolling_beta,
        },
        index=aligned.index,
    )


# ============================================================
# 6. CSV AND PARQUET EXPORT
# ============================================================

def save_table(
    data: pd.DataFrame,
    name: str,
) -> None:
    """Save analytics in CSV and Parquet formats."""

    csv_path = OUTPUT_DIR / f"{name}.csv"
    parquet_path = OUTPUT_DIR / f"{name}.parquet"

    data.to_csv(
        csv_path,
        index=False,
    )

    data.to_parquet(
        parquet_path,
        index=False,
    )

    print(f"Saved: {csv_path.name}")
    print(f"Saved: {parquet_path.name}")


# ============================================================
# 7. CHART GENERATION
# ============================================================

def create_charts(
    strategy_data: dict[str, pd.DataFrame],
    benchmark: pd.Series,
    summary: pd.DataFrame,
) -> None:
    """Generate high-resolution performance charts."""

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Chart 1: Portfolio growth.
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, frame in strategy_data.items():
        ax.plot(
            frame["date"],
            frame["portfolio_value"],
            label=name.replace("_", " ").title(),
            linewidth=1.6,
        )

    benchmark_wealth = (
        INITIAL_CAPITAL
        * (1.0 + benchmark).cumprod()
    )

    ax.plot(
        benchmark_wealth.index,
        benchmark_wealth,
        label="SPY Buy & Hold",
        linewidth=2.0,
        linestyle="--",
    )

    ax.set_title(
        "Out-of-Sample Portfolio Growth",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_ylabel("Portfolio value ($)")
    ax.set_xlabel("Date")
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cumulative_performance.png")
    plt.close(fig)

    # Chart 2: Drawdown comparison.
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, frame in strategy_data.items():
        ax.plot(
            frame["date"],
            frame["drawdown"] * 100,
            label=name.replace("_", " ").title(),
            linewidth=1.2,
        )

    ax.set_title(
        "Portfolio Drawdown Comparison",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "drawdown_comparison.png")
    plt.close(fig)

    # Chart 3: Risk-return profile.
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        summary["annualized_volatility"] * 100,
        summary["annualized_return"] * 100,
        s=90,
    )

    for _, row in summary.iterrows():
        ax.annotate(
            row["strategy"].replace("_", " ").title(),
            (
                row["annualized_volatility"] * 100,
                row["annualized_return"] * 100,
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(
        "Annualized Risk–Return Profile",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_return_comparison.png")
    plt.close(fig)

    # Chart 4: Sharpe ratio.
    fig, ax = plt.subplots(figsize=(11, 6))

    ordered = summary.sort_values(
        "sharpe_ratio",
        ascending=True,
    )

    ax.barh(
        ordered["strategy"],
        ordered["sharpe_ratio"],
    )

    ax.set_title(
        "Risk-Adjusted Performance — Sharpe Ratio",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Sharpe ratio")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sharpe_comparison.png")
    plt.close(fig)

    print("All four performance charts generated successfully.")


# ============================================================
# 8. MAIN ANALYTICS PIPELINE
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("COMPREHENSIVE OUT-OF-SAMPLE PERFORMANCE ANALYTICS")
    print("=" * 80)

    daily, benchmark_data = load_inputs()

    strategy_names = sorted(
        daily["strategy"].unique()
    )

    all_dates = pd.DatetimeIndex(
        sorted(daily["date"].unique()),
        name="date",
    )

    # Align benchmark to the exact out-of-sample dates.
    benchmark = (
        benchmark_data
        .set_index("date")["benchmark_return"]
        .reindex(all_dates)
    )

    # Only missing values inside the actual test period are errors.
    if benchmark.isna().any():

        missing_dates = benchmark.index[
            benchmark.isna()
        ]

        raise ValueError(
            "SPY benchmark is missing observations inside "
            "the out-of-sample testing period. "
            f"First missing dates: {list(missing_dates[:10])}"
        )

    if not np.isfinite(
        benchmark.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Non-finite SPY returns found in the testing period."
        )

    if benchmark.le(-1).any():
        raise ValueError(
            "SPY returns must be greater than -100%."
        )

    summary_rows = []
    drawdown_rows = []
    calendar_rows = []
    rolling_frames = []

    strategy_data = {}

    print(f"Strategies: {len(strategy_names)}")
    print("Benchmark: SPY buy-and-hold")

    print(
        f"Test period: {all_dates.min().date()} "
        f"to {all_dates.max().date()}"
    )

    print(
        f"Observations per strategy: {len(all_dates):,}"
    )

    # --------------------------------------------------------
    # Process each strategy.
    # --------------------------------------------------------

    for strategy in strategy_names:

        frame = (
            daily.loc[daily["strategy"] == strategy]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(frame) != len(all_dates):
            raise ValueError(
                f"{strategy}: observation count differs "
                "from common test dates."
            )

        if not pd.DatetimeIndex(frame["date"]).equals(all_dates):
            raise ValueError(
                f"{strategy}: dates do not match "
                "the common testing period."
            )

        returns = (
            frame
            .set_index("date")["net_daily_return"]
        )

        benchmark_aligned = benchmark.reindex(
            returns.index
        )

        # Performance metrics.
        metrics = calculate_return_metrics(
            returns,
            benchmark_aligned,
        )

        metrics["strategy"] = strategy

        metrics["final_value"] = float(
            frame["portfolio_value"].iloc[-1]
        )

        metrics["total_return"] = float(
            (1.0 + returns).prod() - 1.0
        )

        metrics["total_trading_cost"] = float(
            frame["trading_cost"].sum()
        )

        metrics["total_turnover"] = float(
            frame["turnover"].sum()
        )

        metrics["rebalance_count"] = int(
            frame["is_rebalance"].sum()
        )

        summary_rows.append(metrics)

        strategy_data[strategy] = frame

        # Drawdown episodes.
        drawdown_rows.extend(
            calculate_drawdown_periods(
                frame["date"],
                frame["portfolio_value"],
                strategy,
            )
        )

        # Rolling performance metrics.
        rolling = build_rolling_metrics(
            returns,
            benchmark_aligned,
        )

        rolling = rolling.reset_index()

        rolling.insert(
            1,
            "strategy",
            strategy,
        )

        rolling_frames.append(rolling)

        # Calendar-year returns.
        for year, group in returns.groupby(
            returns.index.year
        ):

            calendar_rows.append({
                "strategy": strategy,
                "year": int(year),
                "annual_return": float(
                    (1.0 + group).prod() - 1.0
                ),
                "observations": int(len(group)),
            })

        print(
            f"Processed {strategy}: "
            f"Sharpe={metrics['sharpe_ratio']:.4f}, "
            f"Sortino={metrics['sortino_ratio']:.4f}, "
            f"Calmar={metrics['calmar_ratio']:.4f}"
        )

    # --------------------------------------------------------
    # Create analytical datasets.
    # --------------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    summary = (
        summary
        .sort_values(
            "sharpe_ratio",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary.insert(
        0,
        "sharpe_rank",
        range(1, len(summary) + 1),
    )

    drawdowns = pd.DataFrame(drawdown_rows)

    if not drawdowns.empty:
        drawdowns = (
            drawdowns
            .sort_values(
                ["strategy", "drawdown"]
            )
            .reset_index(drop=True)
        )

    calendar = pd.DataFrame(calendar_rows)

    rolling_all = pd.concat(
        rolling_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Export strategy analytics.
    # --------------------------------------------------------

    print("\nEXPORTING STRATEGY ANALYTICS")

    save_table(summary, "performance_summary")
    save_table(drawdowns, "drawdown_episodes")
    save_table(calendar, "calendar_year_returns")
    save_table(rolling_all, "rolling_performance_metrics")

    # --------------------------------------------------------
    # Benchmark reference analytics.
    # --------------------------------------------------------

    benchmark_metrics = calculate_return_metrics(
        benchmark
    )

    benchmark_summary = pd.DataFrame([{
        "benchmark": "SPY",
        **benchmark_metrics,
        "initial_capital": INITIAL_CAPITAL,
        "final_value": (
            INITIAL_CAPITAL
            * (1.0 + benchmark).prod()
        ),
        "total_return": float(
            (1.0 + benchmark).prod() - 1.0
        ),
        "cost_treatment": (
            "No explicit benchmark transaction costs"
        ),
    }])

    save_table(
        benchmark_summary,
        "benchmark_summary",
    )

    # --------------------------------------------------------
    # Generate charts.
    # --------------------------------------------------------

    print("\nGENERATING PERFORMANCE CHARTS")

    create_charts(
        strategy_data,
        benchmark,
        summary,
    )

    # --------------------------------------------------------
    # Final strategy report.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY — STRATEGIES")
    print("=" * 80)

    display_columns = [
        "sharpe_rank",
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "information_ratio",
        "beta_vs_spy",
        "total_trading_cost",
    ]

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Benchmark report.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY — SPY")
    print("=" * 80)

    benchmark_columns = [
        "benchmark",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "final_value",
        "total_return",
    ]

    print(
        benchmark_summary[
            benchmark_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Completion.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUTPUT DIRECTORY")
    print("=" * 80)

    print(OUTPUT_DIR)

    print("\nPERFORMANCE ANALYTICS COMPLETED SUCCESSFULLY.")


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
