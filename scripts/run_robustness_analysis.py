
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantfin.backtesting import (
    BacktestConfig,
    run_walk_forward_backtest,
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RETURNS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "us_equities_simple_returns.csv"
)

OUTPUT_DIR = ROOT / "reports" / "robustness_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

INITIAL_CAPITAL = 100_000.0
BASELINE_LOOKBACK = 252
BASELINE_TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 2.0
BASELINE_REBALANCE_FREQUENCY = "M"
BASELINE_MAX_WEIGHT = 0.50
RISK_FREE_RATE = 0.04

TRADING_DAYS = 252

# Minimum observations required before reporting
# conditional market-regime performance statistics.
MIN_REGIME_OBSERVATIONS = 63

ASSETS = [
    "AAPL", "MSFT", "NVDA", "GOOGL",
    "AMZN", "META", "JPM", "XOM",
    "JNJ", "COST", "CAT", "KO",
]

STRATEGIES = [
    "equal_weight",
    "minimum_volatility",
    "maximum_sharpe",
    "hrp",
    "risk_parity",
]

LOOKBACK_VALUES = [126, 252, 504]
TRANSACTION_COST_VALUES = [0, 5, 10, 20, 50]
REBALANCE_FREQUENCIES = ["M", "Q", "Y"]
MAX_WEIGHT_VALUES = [0.30, 0.40, 0.50, 0.60]


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def validate_finite(
    values: np.ndarray,
    description: str,
) -> None:
    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values detected in {description}."
        )


def save_table(
    frame: pd.DataFrame,
    name: str,
) -> None:
    csv_path = OUTPUT_DIR / f"{name}.csv"
    parquet_path = OUTPUT_DIR / f"{name}.parquet"

    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    print(f"Saved: {csv_path.name}")
    print(f"Saved: {parquet_path.name}")


def normalize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize walk-forward output to a date-indexed DataFrame."""

    frame = results.copy()

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date")

    elif "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date")
        frame.index.name = "date"

    else:
        frame.index = pd.to_datetime(
            frame.index,
            errors="coerce",
        )
        frame.index.name = "date"

    if frame.index.isna().any():
        raise ValueError(
            "Invalid dates in walk-forward result."
        )

    frame = frame.sort_index()

    if frame.index.has_duplicates:
        raise ValueError(
            "Duplicate dates in walk-forward result."
        )

    required = {
        "portfolio_value",
        "net_daily_return",
        "gross_daily_return",
        "trading_cost",
        "turnover",
        "drawdown",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing walk-forward result columns: {sorted(missing)}"
        )

    for column in required:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[list(required)].isna().any().any():
        raise ValueError(
            "Missing numeric observations in walk-forward output."
        )

    validate_finite(
        frame[list(required)].to_numpy(dtype=float),
        "walk-forward results",
    )

    if (frame["portfolio_value"] <= 0).any():
        raise ValueError(
            "Portfolio values must be positive."
        )

    if (frame["net_daily_return"] <= -1).any():
        raise ValueError(
            "Net returns must be greater than -100%."
        )

    if "is_rebalance" in frame.columns:
        frame["is_rebalance"] = (
            frame["is_rebalance"].fillna(False).astype(bool)
        )

    return frame


# ============================================================
# 3. INPUT DATA
# ============================================================

def load_returns() -> pd.DataFrame:
    print("\nLoading processed equity returns...")

    if not RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"Return dataset not found: {RETURNS_PATH}"
        )

    returns = pd.read_csv(RETURNS_PATH)

    date_columns = [
        column
        for column in returns.columns
        if str(column).strip().lower() == "date"
    ]

    if not date_columns:
        raise ValueError(
            "Equity return dataset must contain a Date column."
        )

    date_column = date_columns[0]

    returns[date_column] = pd.to_datetime(
        returns[date_column],
        errors="coerce",
    )

    if returns[date_column].isna().any():
        raise ValueError(
            "Invalid dates found in equity returns."
        )

    returns = returns.rename(
        columns={date_column: "date"}
    )

    returns = (
        returns
        .set_index("date")
        .sort_index()
    )

    if returns.index.has_duplicates:
        raise ValueError(
            "Duplicate dates found in equity returns."
        )

    missing_assets = sorted(
        set(ASSETS).difference(returns.columns)
    )

    if missing_assets:
        raise ValueError(
            f"Missing project assets: {missing_assets}"
        )

    returns = returns[ASSETS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # The first return observation may be missing.
    returns = returns.dropna(how="all")

    # Use complete observations for consistent covariance estimation.
    returns = returns.dropna(subset=ASSETS)

    if len(returns) < max(LOOKBACK_VALUES) + 252:
        raise ValueError(
            "Insufficient observations for the requested "
            "lookback and out-of-sample analysis."
        )

    validate_finite(
        returns.to_numpy(dtype=float),
        "processed equity returns",
    )

    if (returns <= -1).any().any():
        raise ValueError(
            "Asset returns must be greater than -100%."
        )

    print(f"Assets: {len(ASSETS)}")
    print(f"Observations: {len(returns):,}")
    print(
        f"Period: {returns.index.min().date()} "
        f"to {returns.index.max().date()}"
    )

    return returns


# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Calculate performance statistics for a continuous return series."""

    returns = frame["net_daily_return"].astype(float)

    if len(returns) < 2:
        raise ValueError(
            "At least two out-of-sample returns are required."
        )

    daily_rf = (
        (1.0 + RISK_FREE_RATE)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    daily_volatility = returns.std(ddof=1)

    annualized_return = (
        (1.0 + returns).prod()
        ** (TRADING_DAYS / len(returns))
        - 1.0
    )

    annualized_volatility = (
        daily_volatility * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        (returns.mean() - daily_rf)
        / daily_volatility
        * np.sqrt(TRADING_DAYS)
        if daily_volatility > 0
        else np.nan
    )

    excess = returns - daily_rf

    downside = np.minimum(
        excess.to_numpy(),
        0.0,
    )

    downside_deviation = (
        np.sqrt(np.mean(downside**2))
        * np.sqrt(TRADING_DAYS)
    )

    sortino = (
        excess.mean() * TRADING_DAYS
        / downside_deviation
        if downside_deviation > 0
        else np.nan
    )

    wealth = pd.concat(
        [
            pd.Series([1.0]),
            (1.0 + returns.reset_index(drop=True)),
        ],
        ignore_index=True,
    ).cumprod()

    drawdown = wealth / wealth.cummax() - 1.0

    max_drawdown = float(drawdown.min())

    calmar = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    return {
        "observations": int(len(returns)),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "total_return": float(
            (1.0 + returns).prod() - 1.0
        ),
        "final_portfolio_value": float(
            frame["portfolio_value"].iloc[-1]
        ),
        "total_trading_cost": float(
            frame["trading_cost"].sum()
        ),
        "total_turnover": float(
            frame["turnover"].sum()
        ),
        "rebalance_count": (
            int(frame["is_rebalance"].sum())
            if "is_rebalance" in frame.columns
            else np.nan
        ),
    }


# ============================================================
# 5. WALK-FORWARD EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    returns: pd.DataFrame,
    strategy: str,
    experiment: str,
    lookback_days: int = BASELINE_LOOKBACK,
    transaction_cost_bps: float = BASELINE_TRANSACTION_COST_BPS,
    rebalance_frequency: str = BASELINE_REBALANCE_FREQUENCY,
    max_weight: float = BASELINE_MAX_WEIGHT,
) -> tuple[dict, pd.DataFrame]:
    """Execute one controlled walk-forward experiment."""

    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=float(transaction_cost_bps),
        slippage_bps=SLIPPAGE_BPS,
        rebalance_frequency=rebalance_frequency,
        charge_initial_transaction=True,
    )

    results, _ = run_walk_forward_backtest(
        returns=returns,
        strategy=strategy,
        lookback_days=int(lookback_days),
        config=config,
        risk_free_rate=RISK_FREE_RATE,
        max_weight=float(max_weight),
    )

    frame = normalize_results(results)

    metrics = calculate_metrics(frame)

    record = {
        "experiment": experiment,
        "strategy": strategy,
        "lookback_days": int(lookback_days),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": SLIPPAGE_BPS,
        "rebalance_frequency": rebalance_frequency,
        "max_weight": float(max_weight),
        "initial_capital": INITIAL_CAPITAL,
        "risk_free_rate": RISK_FREE_RATE,
        "test_start": frame.index.min(),
        "test_end": frame.index.max(),
        **metrics,
    }

    return record, frame


# ============================================================
# 6. RUN BASELINE AND SENSITIVITY EXPERIMENTS
# ============================================================

def run_all_experiments(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records = []
    baseline_frames = {}

    # Cache successful configurations to avoid rerunning
    # identical parameter combinations.
    cache = {}

    def execute(
        strategy: str,
        experiment: str,
        lookback: int,
        cost: float,
        frequency: str,
        max_weight: float,
    ) -> None:

        key = (
            strategy,
            lookback,
            cost,
            frequency,
            max_weight,
        )

        try:
            if key in cache:
                record, frame = cache[key]

            else:
                record, frame = run_experiment(
                    returns=returns,
                    strategy=strategy,
                    experiment=experiment,
                    lookback_days=lookback,
                    transaction_cost_bps=cost,
                    rebalance_frequency=frequency,
                    max_weight=max_weight,
                )

                cache[key] = (
                    record.copy(),
                    frame.copy(),
                )

            output_record = record.copy()
            output_record["experiment"] = experiment
            output_record["status"] = "success"
            output_record["error"] = None

            records.append(output_record)

            print(
                f"{strategy:<22} "
                f"{experiment:<22} "
                f"Sharpe={record['sharpe_ratio']:.4f}"
            )

            if experiment == "baseline":
                baseline_frames[strategy] = frame.copy()

        except Exception as exc:
            failed_record = {
                "experiment": experiment,
                "strategy": strategy,
                "lookback_days": int(lookback),
                "transaction_cost_bps": float(cost),
                "slippage_bps": SLIPPAGE_BPS,
                "rebalance_frequency": frequency,
                "max_weight": float(max_weight),
                "initial_capital": INITIAL_CAPITAL,
                "risk_free_rate": RISK_FREE_RATE,
                "test_start": pd.NaT,
                "test_end": pd.NaT,
                "observations": 0,
                "annualized_return": np.nan,
                "annualized_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "max_drawdown": np.nan,
                "calmar_ratio": np.nan,
                "total_return": np.nan,
                "final_portfolio_value": np.nan,
                "total_trading_cost": np.nan,
                "total_turnover": np.nan,
                "rebalance_count": np.nan,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

            records.append(failed_record)

            print(
                f"FAILED: {strategy} | {experiment} | "
                f"lookback={lookback} | "
                f"cost={cost} bps | "
                f"frequency={frequency} | "
                f"max_weight={max_weight:.2f}"
            )

            print(
                f"Reason: {type(exc).__name__}: {exc}"
            )

    print("\nRunning baseline configurations...")

    for strategy in STRATEGIES:
        execute(
            strategy,
            "baseline",
            BASELINE_LOOKBACK,
            BASELINE_TRANSACTION_COST_BPS,
            BASELINE_REBALANCE_FREQUENCY,
            BASELINE_MAX_WEIGHT,
        )

    print("\nRunning lookback sensitivity...")

    for strategy in STRATEGIES:
        for lookback in LOOKBACK_VALUES:
            execute(
                strategy,
                "lookback",
                lookback,
                BASELINE_TRANSACTION_COST_BPS,
                BASELINE_REBALANCE_FREQUENCY,
                BASELINE_MAX_WEIGHT,
            )

    print("\nRunning transaction-cost sensitivity...")

    for strategy in STRATEGIES:
        for cost in TRANSACTION_COST_VALUES:
            execute(
                strategy,
                "transaction_cost",
                BASELINE_LOOKBACK,
                cost,
                BASELINE_REBALANCE_FREQUENCY,
                BASELINE_MAX_WEIGHT,
            )

    print("\nRunning rebalancing-frequency sensitivity...")

    for strategy in STRATEGIES:
        for frequency in REBALANCE_FREQUENCIES:
            execute(
                strategy,
                "rebalance_frequency",
                BASELINE_LOOKBACK,
                BASELINE_TRANSACTION_COST_BPS,
                frequency,
                BASELINE_MAX_WEIGHT,
            )

    print("\nRunning maximum-weight sensitivity...")

    for strategy in STRATEGIES:
        for max_weight in MAX_WEIGHT_VALUES:
            execute(
                strategy,
                "maximum_weight",
                BASELINE_LOOKBACK,
                BASELINE_TRANSACTION_COST_BPS,
                BASELINE_REBALANCE_FREQUENCY,
                max_weight,
            )

    return pd.DataFrame(records), baseline_frames


# ============================================================
# 7. ROLLING PERFORMANCE STABILITY
# ============================================================

def build_rolling_metrics(
    baseline_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print("\nCalculating rolling performance stability...")

    records = []

    daily_rf = (
        (1.0 + RISK_FREE_RATE)
        ** (1.0 / TRADING_DAYS)
        - 1.0
    )

    for strategy, frame in baseline_frames.items():

        returns = frame["net_daily_return"]

        rolling_mean = returns.rolling(
            TRADING_DAYS
        ).mean()

        rolling_std = returns.rolling(
            TRADING_DAYS
        ).std(ddof=1)

        rolling_sharpe = (
            (rolling_mean - daily_rf)
            / rolling_std
            * np.sqrt(TRADING_DAYS)
        )

        rolling_volatility = (
            rolling_std * np.sqrt(TRADING_DAYS)
        )

        rolling_return = (
            (1.0 + returns)
            .rolling(TRADING_DAYS)
            .apply(np.prod, raw=True)
            - 1.0
        )

        rolling = pd.DataFrame({
            "date": frame.index,
            "strategy": strategy,
            "rolling_252d_return": rolling_return,
            "rolling_annualized_volatility": rolling_volatility,
            "rolling_sharpe_ratio": rolling_sharpe,
        })

        records.append(rolling)

    if not records:
        raise ValueError(
            "No baseline frames available for rolling analysis."
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


# ============================================================
# 8. MARKET REGIME CLASSIFICATION
# ============================================================

def classify_market_regimes(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify four market regimes using an equal-weight
    project-universe proxy.

    Signals are shifted by one trading session so that
    the regime assigned to date t uses information available
    through date t-1.

    This is a project-universe proxy, not the SPY benchmark.
    """

    print(
        "\nClassifying market regimes using lagged information..."
    )

    # Equal-weight daily return of the 12-stock universe.
    universe_proxy_return = returns.mean(axis=1)

    # Trailing 126-session cumulative proxy return.
    trailing_return = (
        (1.0 + universe_proxy_return)
        .rolling(126)
        .apply(np.prod, raw=True)
        - 1.0
    )

    # Trailing 63-session annualized volatility.
    trailing_volatility = (
        universe_proxy_return
        .rolling(63)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    # Lag both indicators by one session.
    trailing_return = trailing_return.shift(1)
    trailing_volatility = trailing_volatility.shift(1)

    # Expanding 75th percentile threshold.
    volatility_threshold = (
        trailing_volatility
        .expanding(min_periods=252)
        .quantile(0.75)
    )

    valid = (
        trailing_return.notna()
        & trailing_volatility.notna()
        & volatility_threshold.notna()
    )

    bear = trailing_return < 0
    high_volatility = (
        trailing_volatility > volatility_threshold
    )

    regime = pd.Series(
        "unclassified",
        index=returns.index,
        dtype="object",
    )

    regime.loc[
        valid & bear & high_volatility
    ] = "bear_high_volatility"

    regime.loc[
        valid & bear & ~high_volatility
    ] = "bear_normal_volatility"

    regime.loc[
        valid & ~bear & high_volatility
    ] = "bull_high_volatility"

    regime.loc[
        valid & ~bear & ~high_volatility
    ] = "bull_normal_volatility"

    classification = pd.DataFrame({
        "date": returns.index,
        "universe_proxy_daily_return": universe_proxy_return,
        "universe_proxy_trailing_126d_return": trailing_return,
        "universe_proxy_trailing_63d_volatility": trailing_volatility,
        "volatility_threshold": volatility_threshold,
        "market_regime": regime,
    })

    counts = (
        classification["market_regime"]
        .value_counts()
        .reindex(
            [
                "bear_high_volatility",
                "bear_normal_volatility",
                "bull_high_volatility",
                "bull_normal_volatility",
                "unclassified",
            ],
            fill_value=0,
        )
    )

    print("\nMarket-regime observation counts:")

    for regime_name, count in counts.items():
        print(f"{regime_name:<28} {count:>6,}")

    return classification


# ============================================================
# 9. CONDITIONAL MARKET REGIME PERFORMANCE
# ============================================================

def calculate_regime_performance(
    baseline_frames: dict[str, pd.DataFrame],
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate conditional performance within each regime.

    Regime observations are not necessarily consecutive.
    Therefore, this function does not calculate a regime-specific
    compounded annual return or drawdown from filtered daily data.

    Annualized return is the conditional arithmetic estimate:
        mean daily return * 252

    Annualized volatility:
        daily standard deviation * sqrt(252)
    """

    print(
        "\nCalculating conditional market-regime performance..."
    )

    regime_data = (
        regimes
        .set_index("date")
        [["market_regime"]]
    )

    records = []

    for strategy, frame in baseline_frames.items():

        aligned = frame.join(
            regime_data,
            how="inner",
        )

        aligned = aligned[
            aligned["market_regime"] != "unclassified"
        ]

        total_classified = len(aligned)

        for regime_name, group in aligned.groupby(
            "market_regime",
            sort=True,
        ):

            daily_returns = (
                group["net_daily_return"].astype(float)
            )

            observations = len(daily_returns)

            record = {
                "strategy": strategy,
                "market_regime": regime_name,
                "observations": observations,
                "share_of_classified_observations": (
                    observations / total_classified
                    if total_classified > 0
                    else np.nan
                ),
                "mean_daily_return": daily_returns.mean(),
                "median_daily_return": daily_returns.median(),
                "hit_rate": (
                    (daily_returns > 0).mean()
                ),
                "conditional_annualized_return": np.nan,
                "annualized_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "sample_status": "insufficient_sample",
            }

            if observations >= MIN_REGIME_OBSERVATIONS:

                daily_rf = (
                    (1.0 + RISK_FREE_RATE)
                    ** (1.0 / TRADING_DAYS)
                    - 1.0
                )

                daily_mean = daily_returns.mean()
                daily_std = daily_returns.std(ddof=1)

                conditional_annualized_return = (
                    daily_mean * TRADING_DAYS
                )

                annualized_volatility = (
                    daily_std * np.sqrt(TRADING_DAYS)
                )

                sharpe = (
                    (daily_mean - daily_rf)
                    / daily_std
                    * np.sqrt(TRADING_DAYS)
                    if daily_std > 0
                    else np.nan
                )

                record.update({
                    "conditional_annualized_return": (
                        conditional_annualized_return
                    ),
                    "annualized_volatility": (
                        annualized_volatility
                    ),
                    "sharpe_ratio": sharpe,
                    "sample_status": "sufficient_for_reporting",
                })

            records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 10. ROBUSTNESS SCORECARD
# ============================================================

def build_robustness_scorecard(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank strategies under identical experiment settings.

    Strategies are not ranked across different lookbacks,
    transaction costs, rebalance frequencies, or weight limits.
    """

    sensitivity = results[
        (results["experiment"] != "baseline")
        & (results["status"] == "success")
    ].copy()

    if sensitivity.empty:
        raise ValueError(
            "No successful sensitivity experiments are available."
        )

    sensitivity["comparison_setting"] = ""

    lookback_mask = (
        sensitivity["experiment"] == "lookback"
    )
    sensitivity.loc[
        lookback_mask,
        "comparison_setting",
    ] = (
        "lookback="
        + sensitivity.loc[
            lookback_mask,
            "lookback_days",
        ].astype(str)
    )

    cost_mask = (
        sensitivity["experiment"] == "transaction_cost"
    )
    sensitivity.loc[
        cost_mask,
        "comparison_setting",
    ] = (
        "cost="
        + sensitivity.loc[
            cost_mask,
            "transaction_cost_bps",
        ].astype(str)
    )

    frequency_mask = (
        sensitivity["experiment"] == "rebalance_frequency"
    )
    sensitivity.loc[
        frequency_mask,
        "comparison_setting",
    ] = (
        "frequency="
        + sensitivity.loc[
            frequency_mask,
            "rebalance_frequency",
        ].astype(str)
    )

    weight_mask = (
        sensitivity["experiment"] == "maximum_weight"
    )
    sensitivity.loc[
        weight_mask,
        "comparison_setting",
    ] = (
        "max_weight="
        + sensitivity.loc[
            weight_mask,
            "max_weight",
        ].round(4).astype(str)
    )

    # Rank strategies only within matching settings.
    sensitivity["sharpe_rank"] = (
        sensitivity
        .groupby(
            ["experiment", "comparison_setting"]
        )["sharpe_ratio"]
        .rank(
            ascending=False,
            method="average",
        )
    )

    scorecard = (
        sensitivity
        .groupby("strategy")
        .agg(
            mean_sharpe=("sharpe_ratio", "mean"),
            median_sharpe=("sharpe_ratio", "median"),
            sharpe_std=("sharpe_ratio", "std"),
            mean_sharpe_rank=("sharpe_rank", "mean"),
            best_sharpe=("sharpe_ratio", "max"),
            worst_sharpe=("sharpe_ratio", "min"),
            mean_annualized_return=(
                "annualized_return",
                "mean",
            ),
            mean_max_drawdown=(
                "max_drawdown",
                "mean",
            ),
            experiment_count=("experiment", "count"),
            rank_observations=("sharpe_rank", "count"),
        )
        .reset_index()
    )

    scorecard["sharpe_range"] = (
        scorecard["best_sharpe"]
        - scorecard["worst_sharpe"]
    )

    return scorecard.sort_values(
        "mean_sharpe_rank"
    ).reset_index(drop=True)


# ============================================================
# 11. CHART GENERATION
# ============================================================

def create_charts(
    results: pd.DataFrame,
    rolling: pd.DataFrame,
    regime_performance: pd.DataFrame,
    scorecard: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:

    print("\nGenerating robustness charts...")

    results = results[
        results["status"] == "success"
    ].copy()

    if results.empty:
        raise ValueError(
            "No successful experiments are available for chart generation."
        )

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    chart_count = 0

    # --------------------------------------------------------
    # Chart 1: Lookback sensitivity.
    # --------------------------------------------------------

    lookback = results[
        results["experiment"] == "lookback"
    ]

    if not lookback.empty:

        fig, ax = plt.subplots(figsize=(12, 7))

        for strategy, frame in lookback.groupby("strategy"):
            frame = frame.sort_values("lookback_days")

            ax.plot(
                frame["lookback_days"],
                frame["sharpe_ratio"],
                marker="o",
                label=strategy,
            )

        ax.set_title("Lookback Window Sensitivity")
        ax.set_xlabel("Lookback sessions")
        ax.set_ylabel("Out-of-sample Sharpe ratio")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "lookback_sensitivity.png"
        )
        plt.close(fig)
        chart_count += 1

    # --------------------------------------------------------
    # Chart 2: Transaction-cost sensitivity.
    # --------------------------------------------------------

    costs = results[
        results["experiment"] == "transaction_cost"
    ]

    if not costs.empty:

        fig, ax = plt.subplots(figsize=(12, 7))

        for strategy, frame in costs.groupby("strategy"):
            frame = frame.sort_values(
                "transaction_cost_bps"
            )

            ax.plot(
                frame["transaction_cost_bps"],
                frame["annualized_return"] * 100,
                marker="o",
                label=strategy,
            )

        ax.set_title("Transaction-Cost Sensitivity")
        ax.set_xlabel("Transaction cost (bps)")
        ax.set_ylabel("Annualized net return (%)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "transaction_cost_sensitivity.png"
        )
        plt.close(fig)
        chart_count += 1

    # --------------------------------------------------------
    # Chart 3: Rebalancing frequency.
    # --------------------------------------------------------

    frequency = results[
        results["experiment"] == "rebalance_frequency"
    ]

    if not frequency.empty:

        frequency_order = ["M", "Q", "Y"]

        pivot = frequency.pivot(
            index="rebalance_frequency",
            columns="strategy",
            values="sharpe_ratio",
        ).reindex(frequency_order)

        ax = pivot.plot(
            kind="bar",
            figsize=(12, 7),
        )

        ax.set_title("Rebalancing Frequency Sensitivity")
        ax.set_xlabel(
            "Frequency: M = monthly, Q = quarterly, Y = yearly"
        )
        ax.set_ylabel("Out-of-sample Sharpe ratio")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Strategy")

        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / "rebalance_frequency_sensitivity.png"
        )
        plt.close()
        chart_count += 1

    # --------------------------------------------------------
    # Chart 4: Maximum asset weight.
    # --------------------------------------------------------

    weights = results[
        results["experiment"] == "maximum_weight"
    ]

    if not weights.empty:

        fig, ax = plt.subplots(figsize=(12, 7))

        for strategy, frame in weights.groupby("strategy"):
            frame = frame.sort_values("max_weight")

            ax.plot(
                frame["max_weight"] * 100,
                frame["sharpe_ratio"],
                marker="o",
                label=strategy,
            )

        ax.set_title("Maximum Asset Weight Sensitivity")
        ax.set_xlabel("Maximum asset weight (%)")
        ax.set_ylabel("Out-of-sample Sharpe ratio")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "maximum_weight_sensitivity.png"
        )
        plt.close(fig)
        chart_count += 1

    # --------------------------------------------------------
    # Chart 5: Rolling Sharpe.
    # --------------------------------------------------------

    if not rolling.empty:

        fig, ax = plt.subplots(figsize=(14, 7))

        for strategy, frame in rolling.groupby("strategy"):
            ax.plot(
                frame["date"],
                frame["rolling_sharpe_ratio"],
                label=strategy,
                linewidth=1.1,
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_title("252-Session Rolling Sharpe Ratio")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Sharpe ratio")
        ax.legend(ncol=2)

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "rolling_sharpe_stability.png"
        )
        plt.close(fig)
        chart_count += 1

    # --------------------------------------------------------
    # Chart 6: Conditional regime Sharpe.
    # --------------------------------------------------------

    if not regime_performance.empty:

        regime_chart_data = regime_performance[
            (regime_performance["sample_status"] == "sufficient_for_reporting")
            & regime_performance["sharpe_ratio"].notna()
        ]

        if not regime_chart_data.empty:

            regime_pivot = regime_chart_data.pivot(
                index="strategy",
                columns="market_regime",
                values="sharpe_ratio",
            )

            ax = regime_pivot.plot(
                kind="bar",
                figsize=(14, 7),
            )

            ax.set_title(
                "Conditional Strategy Sharpe Across Market Regimes"
            )
            ax.set_xlabel("Strategy")
            ax.set_ylabel("Conditional Sharpe ratio")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Market regime")

            plt.tight_layout()
            plt.savefig(
                FIGURE_DIR / "market_regime_comparison.png"
            )
            plt.close()
            chart_count += 1

    # --------------------------------------------------------
    # Chart 7: Robustness scorecard.
    # --------------------------------------------------------

    if not scorecard.empty:

        ordered = scorecard.sort_values(
            "mean_sharpe_rank",
            ascending=False,
        )

        fig, ax = plt.subplots(figsize=(11, 7))

        ax.barh(
            ordered["strategy"],
            ordered["mean_sharpe_rank"],
        )

        ax.set_title(
            "Average Strategy Rank Across Comparable Settings"
        )
        ax.set_xlabel("Mean Sharpe rank (lower is better)")
        ax.set_ylabel("Strategy")

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "strategy_robustness_scorecard.png"
        )
        plt.close(fig)
        chart_count += 1

    # --------------------------------------------------------
    # Chart 8: Regime distribution.
    # --------------------------------------------------------

    regime_counts = (
        regimes["market_regime"]
        .value_counts()
        .reindex(
            [
                "bear_high_volatility",
                "bear_normal_volatility",
                "bull_high_volatility",
                "bull_normal_volatility",
                "unclassified",
            ],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    regime_counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Market-Regime Observation Distribution")
    ax.set_xlabel("Market regime")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "market_regime_distribution.png"
    )
    plt.close(fig)
    chart_count += 1

    print(f"Generated {chart_count} charts.")


# ============================================================
# 12. MAIN PIPELINE
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
    print("PORTFOLIO ROBUSTNESS AND SENSITIVITY ANALYSIS")
    print("=" * 80)

    returns = load_returns()

    # --------------------------------------------------------
    # Run all experiments.
    # --------------------------------------------------------

    results, baseline_frames = run_all_experiments(
        returns
    )

    failures = results[
        results["status"] == "failed"
    ].copy()

    if not failures.empty:
        save_table(
            failures,
            "failed_robustness_experiments",
        )

        print(
            f"\nWARNING: {len(failures)} robustness experiments failed."
        )

    else:
        print("\nAll robustness experiments succeeded.")

        # Remove stale failure reports from previous runs.
        for extension in ("csv", "parquet"):
            stale_report = (
                OUTPUT_DIR
                / f"failed_robustness_experiments.{extension}"
            )

            if stale_report.exists():
                stale_report.unlink()

                print(
                    f"Removed stale report: {stale_report.name}"
                )

    # A missing baseline invalidates downstream comparisons.
    missing_baselines = sorted(
        set(STRATEGIES).difference(baseline_frames)
    )

    if missing_baselines:
        raise RuntimeError(
            "Baseline optimization failed for required strategies: "
            f"{missing_baselines}. Review "
            "'failed_robustness_experiments.csv'."
        )

    # --------------------------------------------------------
    # Rolling performance.
    # --------------------------------------------------------

    print("\nCalculating rolling metrics...")

    rolling = build_rolling_metrics(
        baseline_frames
    )

    # --------------------------------------------------------
    # Regime classification and analysis.
    # --------------------------------------------------------

    regimes = classify_market_regimes(
        returns
    )

    regime_performance = calculate_regime_performance(
        baseline_frames,
        regimes,
    )

    # --------------------------------------------------------
    # Robustness scorecard.
    # --------------------------------------------------------

    scorecard = build_robustness_scorecard(
        results
    )

    # --------------------------------------------------------
    # Save datasets.
    # --------------------------------------------------------

    print("\nExporting robustness datasets...")

    save_table(
        results,
        "sensitivity_experiment_results",
    )

    save_table(
        rolling,
        "rolling_robustness_metrics",
    )

    save_table(
        regimes.reset_index(drop=True),
        "market_regime_classification",
    )

    save_table(
        regime_performance,
        "market_regime_performance",
    )

    save_table(
        scorecard,
        "strategy_robustness_scorecard",
    )

    # --------------------------------------------------------
    # Generate charts.
    # --------------------------------------------------------

    create_charts(
        results,
        rolling,
        regime_performance,
        scorecard,
        regimes,
    )

    # --------------------------------------------------------
    # Print baseline results.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("BASELINE STRATEGY RESULTS")
    print("=" * 80)

    baseline = results[
        (results["experiment"] == "baseline")
        & (results["status"] == "success")
    ].sort_values(
        "sharpe_ratio",
        ascending=False,
    )

    print(
        baseline[
            [
                "strategy",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "total_trading_cost",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Print scorecard.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ROBUSTNESS SCORECARD")
    print("=" * 80)

    print(
        scorecard.to_string(index=False)
    )

    # --------------------------------------------------------
    # Print regime performance.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONDITIONAL MARKET REGIME PERFORMANCE")
    print("=" * 80)

    if regime_performance.empty:
        print("No classified regime observations available.")

    else:
        print(
            regime_performance[
                [
                    "strategy",
                    "market_regime",
                    "observations",
                    "share_of_classified_observations",
                    "conditional_annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "hit_rate",
                    "sample_status",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Completion summary.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS COMPLETED")
    print("=" * 80)

    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    print(
        "Sensitivity experiment rows: "
        f"{len(results)}"
    )

    print(
        "Successful experiments: "
        f"{(results['status'] == 'success').sum()}"
    )

    print(
        "Failed experiments: "
        f"{(results['status'] == 'failed').sum()}"
    )

    print(
        "Generated CSV files: "
        f"{len(list(OUTPUT_DIR.glob('*.csv')))}"
    )

    print(
        "Generated Parquet files: "
        f"{len(list(OUTPUT_DIR.glob('*.parquet')))}"
    )

    print(
        "Generated charts: "
        f"{len(list(FIGURE_DIR.glob('*.png')))}"
    )


if __name__ == "__main__":
    main()
