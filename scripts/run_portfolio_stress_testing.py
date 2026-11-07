
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DAILY_RESULTS_PATH = (
    ROOT
    / "reports"
    / "walk_forward_comparison"
    / "strategy_daily_results.csv"
)

EQUITY_RETURNS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "us_equities_simple_returns.csv"
)

OUTPUT_DIR = ROOT / "reports" / "stress_testing"
FIGURE_DIR = OUTPUT_DIR / "figures"

INITIAL_CAPITAL = 100_000.0
TRADING_DAYS = 252
CONFIDENCE_LEVEL = 0.99
COVARIANCE_LOOKBACK = 252

BROAD_MARKET_SHOCKS = [-0.05, -0.10, -0.20, -0.30]
INDIVIDUAL_ASSET_SHOCK = -0.30
SECTOR_SHOCK = -0.20
COMBINED_MARKET_SHOCK = -0.10
COMBINED_INCREMENTAL_SECTOR_SHOCK = -0.15

CORRELATION_STRESS_WEIGHT = 0.75
VOLATILITY_MULTIPLIER = 1.50

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "JPM": "Financials",
    "XOM": "Energy",
    "JNJ": "Healthcare",
    "CAT": "Industrials",
}


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


def get_date_column(frame: pd.DataFrame) -> str:
    candidates = [
        column
        for column in frame.columns
        if str(column).strip().lower() in {
            "date",
            "datetime",
            "timestamp",
        }
    ]

    if not candidates:
        raise ValueError(
            "Could not identify a date column in the input dataset."
        )

    return candidates[0]


# ============================================================
# 3. INPUT LOADING AND VALIDATION
# ============================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nLoading stress-testing input datasets...")

    if not DAILY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Strategy results not found: {DAILY_RESULTS_PATH}"
        )

    if not EQUITY_RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"Equity returns not found: {EQUITY_RETURNS_PATH}"
        )

    daily = pd.read_csv(DAILY_RESULTS_PATH)
    equity = pd.read_csv(EQUITY_RETURNS_PATH)

    daily_date_column = get_date_column(daily)
    equity_date_column = get_date_column(equity)

    daily[daily_date_column] = pd.to_datetime(
        daily[daily_date_column],
        errors="coerce",
    )

    equity[equity_date_column] = pd.to_datetime(
        equity[equity_date_column],
        errors="coerce",
    )

    daily = daily.rename(
        columns={daily_date_column: "date"}
    )

    equity = equity.rename(
        columns={equity_date_column: "date"}
    )

    required_daily = {
        "date",
        "strategy",
        "portfolio_value",
        "net_daily_return",
    }

    missing = required_daily.difference(daily.columns)

    if missing:
        raise ValueError(
            f"Missing strategy columns: {sorted(missing)}"
        )

    if daily["date"].isna().any():
        raise ValueError(
            "Invalid dates found in strategy results."
        )

    if equity["date"].isna().any():
        raise ValueError(
            "Invalid dates found in equity returns."
        )

    if daily["strategy"].isna().any():
        raise ValueError("Missing strategy names.")

    if daily.duplicated(["date", "strategy"]).any():
        raise ValueError(
            "Duplicate strategy/date observations detected."
        )

    if equity["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found in equity returns."
        )

    daily["portfolio_value"] = pd.to_numeric(
        daily["portfolio_value"],
        errors="coerce",
    )

    daily["net_daily_return"] = pd.to_numeric(
        daily["net_daily_return"],
        errors="coerce",
    )

    if daily[
        ["portfolio_value", "net_daily_return"]
    ].isna().any().any():
        raise ValueError(
            "Missing portfolio values or daily returns."
        )

    if (daily["portfolio_value"] <= 0).any():
        raise ValueError(
            "Portfolio values must be positive."
        )

    if (daily["net_daily_return"] <= -1).any():
        raise ValueError(
            "Net returns must be greater than -100%."
        )

    # Keep only the supported equity universe.
    assets = [
        asset
        for asset in SECTOR_MAP
        if asset in equity.columns
    ]

    if len(assets) < 2:
        raise ValueError(
            "Fewer than two recognized equity return columns found."
        )

    for asset in assets:
        equity[asset] = pd.to_numeric(
            equity[asset],
            errors="coerce",
        )

    equity = (
        equity[["date", *assets]]
        .sort_values("date")
        .set_index("date")
    )

    # Initial missing return observations are expected.
    equity = equity.dropna(how="all")

    # Use complete observations for covariance estimation.
    equity = equity.dropna(subset=assets)

    if len(equity) < COVARIANCE_LOOKBACK:
        raise ValueError(
            "Insufficient complete equity return observations "
            f"for a {COVARIANCE_LOOKBACK}-session covariance window."
        )

    validate_finite(
        equity[assets].to_numpy(dtype=float),
        "equity returns",
    )

    if (equity[assets] <= -1).any().any():
        raise ValueError(
            "Equity returns must be greater than -100%."
        )

    daily = (
        daily
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )

    print(f"Strategies loaded: {daily['strategy'].nunique()}")
    print(f"Equities loaded: {len(assets)}")
    print(f"Equity observations: {len(equity):,}")
    print(
        f"Equity data period: {equity.index.min().date()} "
        f"to {equity.index.max().date()}"
    )

    return daily, equity


# ============================================================
# 4. LATEST PORTFOLIO WEIGHTS
# ============================================================

def extract_latest_weights(
    daily: pd.DataFrame,
    assets: list[str],
) -> dict[str, dict[str, float]]:
    """Extract each strategy's latest available portfolio weights."""

    print("\nExtracting latest portfolio weights...")

    weight_columns = {
        asset: f"weight_{asset}"
        for asset in assets
    }

    missing = [
        column
        for column in weight_columns.values()
        if column not in daily.columns
    ]

    if missing:
        raise ValueError(
            f"Missing portfolio weight columns: {missing}"
        )

    latest_weights = {}

    for strategy, frame in daily.groupby("strategy"):

        latest = frame.sort_values("date").iloc[-1]

        weights = {
            asset: float(latest[column])
            for asset, column in weight_columns.items()
        }

        values = np.array(
            list(weights.values()),
            dtype=float,
        )

        validate_finite(
            values,
            f"{strategy} portfolio weights",
        )

        if (values < -1e-8).any():
            raise ValueError(
                f"Negative weights found for {strategy}."
            )

        if not np.isclose(
            values.sum(),
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                f"{strategy} weights sum to {values.sum():.6f}, "
                "not 1.0."
            )

        latest_weights[strategy] = weights

        print(
            f"{strategy}: weights sum to {values.sum():.6f}"
        )

    return latest_weights


# ============================================================
# 5. HISTORICAL CRISIS STRESS TESTING
# ============================================================

HISTORICAL_PERIODS = {
    "COVID-19 Crash": (
        "2020-02-19",
        "2020-03-23",
    ),
    "COVID-19 Recovery": (
        "2020-03-24",
        "2020-12-31",
    ),
    "2022 Bear Market": (
        "2022-01-03",
        "2022-10-12",
    ),
}


def historical_metrics(
    returns: pd.Series,
) -> dict[str, float]:
    returns = returns.dropna().astype(float)

    if returns.empty:
        raise ValueError(
            "No returns available for historical stress period."
        )

    cumulative_return = float(
        (1.0 + returns).prod() - 1.0
    )

    annualized_volatility = float(
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    ) if len(returns) > 1 else np.nan

    wealth = (1.0 + returns).cumprod()

    wealth = pd.concat(
        [
            pd.Series([1.0]),
            wealth.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    drawdown = wealth / wealth.cummax() - 1.0

    var = float(
        returns.quantile(1.0 - CONFIDENCE_LEVEL)
    )

    tail = returns[
        returns <= var
    ]

    cvar = float(tail.mean())

    return {
        "observations": int(len(returns)),
        "cumulative_return": cumulative_return,
        "annualized_volatility": annualized_volatility,
        "maximum_drawdown": float(drawdown.min()),
        "worst_daily_return": float(returns.min()),
        "historical_var_99": var,
        "historical_cvar_99": cvar,
    }


def run_historical_stress(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    print("\nRunning historical crisis stress tests...")

    results = []

    for strategy, frame in daily.groupby("strategy"):

        frame = frame.set_index("date").sort_index()

        for period_name, (start, end) in HISTORICAL_PERIODS.items():

            start_date = pd.Timestamp(start)
            end_date = pd.Timestamp(end)

            period_returns = frame.loc[
                frame.index.to_series().between(
                    start_date,
                    end_date,
                ),
                "net_daily_return",
            ]

            if period_returns.empty:
                raise ValueError(
                    f"No data for {period_name}, strategy {strategy}."
                )

            metrics = historical_metrics(period_returns)

            results.append({
                "strategy": strategy,
                "historical_period": period_name,
                "start_date": start_date,
                "end_date": end_date,
                **metrics,
            })

    result = pd.DataFrame(results)

    print(
        f"Historical stress cases calculated: {len(result)}"
    )

    return result


# ============================================================
# 6. HYPOTHETICAL MARKET AND ASSET SCENARIOS
# ============================================================

def portfolio_scenario_return(
    weights: dict[str, float],
    asset_shocks: dict[str, float],
) -> float:
    """Calculate weighted one-period portfolio scenario return."""

    return float(
        sum(
            weights[asset] * asset_shocks.get(asset, 0.0)
            for asset in weights
        )
    )


def run_hypothetical_scenarios(
    weights_by_strategy: dict[str, dict[str, float]],
) -> pd.DataFrame:
    print("\nRunning hypothetical market and asset scenarios...")

    results = []

    for strategy, weights in weights_by_strategy.items():

        # Broad market shocks.
        for shock in BROAD_MARKET_SHOCKS:

            shocks = {
                asset: shock
                for asset in weights
            }

            portfolio_return = portfolio_scenario_return(
                weights,
                shocks,
            )

            results.append({
                "strategy": strategy,
                "scenario_type": "Broad market",
                "scenario": f"Market shock {shock:.0%}",
                "portfolio_return": portfolio_return,
                "estimated_pnl": (
                    INITIAL_CAPITAL * portfolio_return
                ),
                "stressed_portfolio_value": (
                    INITIAL_CAPITAL
                    * (1.0 + portfolio_return)
                ),
            })

        # Individual asset shocks.
        for asset in weights:

            shocks = {
                asset: INDIVIDUAL_ASSET_SHOCK
            }

            portfolio_return = portfolio_scenario_return(
                weights,
                shocks,
            )

            results.append({
                "strategy": strategy,
                "scenario_type": "Individual asset",
                "scenario": (
                    f"{asset} falls {abs(INDIVIDUAL_ASSET_SHOCK):.0%}"
                ),
                "portfolio_return": portfolio_return,
                "estimated_pnl": (
                    INITIAL_CAPITAL * portfolio_return
                ),
                "stressed_portfolio_value": (
                    INITIAL_CAPITAL
                    * (1.0 + portfolio_return)
                ),
            })

        # Sector-level shocks.
        sectors = sorted(
            {
                SECTOR_MAP[asset]
                for asset in weights
                if asset in SECTOR_MAP
            }
        )

        for sector in sectors:

            shocks = {
                asset: (
                    SECTOR_SHOCK
                    if SECTOR_MAP[asset] == sector
                    else 0.0
                )
                for asset in weights
            }

            portfolio_return = portfolio_scenario_return(
                weights,
                shocks,
            )

            results.append({
                "strategy": strategy,
                "scenario_type": "Sector",
                "scenario": (
                    f"{sector} falls {abs(SECTOR_SHOCK):.0%}"
                ),
                "portfolio_return": portfolio_return,
                "estimated_pnl": (
                    INITIAL_CAPITAL * portfolio_return
                ),
                "stressed_portfolio_value": (
                    INITIAL_CAPITAL
                    * (1.0 + portfolio_return)
                ),
            })

        # Combined market and incremental sector shock.
        for sector in sectors:

            shocks = {}

            for asset in weights:

                base_shock = COMBINED_MARKET_SHOCK

                if SECTOR_MAP[asset] == sector:
                    asset_shock = (
                        (1.0 + base_shock)
                        * (
                            1.0
                            + COMBINED_INCREMENTAL_SECTOR_SHOCK
                        )
                        - 1.0
                    )
                else:
                    asset_shock = base_shock

                shocks[asset] = asset_shock

            portfolio_return = portfolio_scenario_return(
                weights,
                shocks,
            )

            results.append({
                "strategy": strategy,
                "scenario_type": "Combined",
                "scenario": (
                    f"Market {COMBINED_MARKET_SHOCK:.0%} "
                    f"+ {sector} incremental "
                    f"{COMBINED_INCREMENTAL_SECTOR_SHOCK:.0%}"
                ),
                "portfolio_return": portfolio_return,
                "estimated_pnl": (
                    INITIAL_CAPITAL * portfolio_return
                ),
                "stressed_portfolio_value": (
                    INITIAL_CAPITAL
                    * (1.0 + portfolio_return)
                ),
            })

    result = pd.DataFrame(results)

    print(
        f"Hypothetical scenarios calculated: {len(result)}"
    )

    return result


# ============================================================
# 7. VOLATILITY AND CORRELATION STRESS
# ============================================================

def nearest_positive_definite(
    matrix: np.ndarray,
    minimum_eigenvalue: float = 1e-8,
) -> np.ndarray:
    """Symmetrize a matrix and floor its eigenvalues."""

    symmetric = (matrix + matrix.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        symmetric
    )

    eigenvalues = np.maximum(
        eigenvalues,
        minimum_eigenvalue,
    )

    adjusted = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    adjusted = (
        adjusted + adjusted.T
    ) / 2.0

    return adjusted


def portfolio_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    variance = float(
        weights @ covariance @ weights
    )

    return float(np.sqrt(max(variance, 0.0)))


def calculate_parametric_var_es(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float]:
    """Calculate normal-parametric VaR and expected shortfall."""

    from scipy.stats import norm

    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    z = float(
        norm.ppf(CONFIDENCE_LEVEL)
    )

    # Positive loss convention.
    var = z * volatility

    expected_shortfall = (
        norm.pdf(z)
        / (1.0 - CONFIDENCE_LEVEL)
        * volatility
    )

    return float(var), float(expected_shortfall)


def run_covariance_stress(
    equity: pd.DataFrame,
    weights_by_strategy: dict[str, dict[str, float]],
) -> pd.DataFrame:
    print("\nRunning volatility and correlation stress tests...")

    # Use recent 252 complete trading sessions.
    recent = equity.tail(COVARIANCE_LOOKBACK)

    if len(recent) != COVARIANCE_LOOKBACK:
        raise ValueError(
            "The trailing covariance window is incomplete."
        )

    assets = list(recent.columns)

    daily_covariance = recent.cov().to_numpy(
        dtype=float
    )

    validate_finite(
        daily_covariance,
        "historical covariance matrix",
    )

    daily_volatility = np.sqrt(
        np.diag(daily_covariance)
    )

    if (daily_volatility <= 0).any():
        raise ValueError(
            "Non-positive asset volatility detected."
        )

    historical_correlation = recent.corr().to_numpy(
        dtype=float
    )

    # Increase correlations toward a common-correlation matrix.
    stressed_correlation = (
        (1.0 - CORRELATION_STRESS_WEIGHT)
        * historical_correlation
        + CORRELATION_STRESS_WEIGHT
        * np.ones_like(historical_correlation)
    )

    np.fill_diagonal(
        stressed_correlation,
        1.0,
    )

    # Scale volatility and preserve a valid covariance matrix.
    stressed_volatility = (
        daily_volatility
        * VOLATILITY_MULTIPLIER
    )

    stressed_covariance = (
        np.outer(
            stressed_volatility,
            stressed_volatility,
        )
        * stressed_correlation
    )

    stressed_covariance = nearest_positive_definite(
        stressed_covariance
    )

    historical_covariance = nearest_positive_definite(
        daily_covariance
    )

    results = []

    for strategy, weight_dict in weights_by_strategy.items():

        weights = np.array(
            [
                weight_dict[asset]
                for asset in assets
            ],
            dtype=float,
        )

        historical_vol = portfolio_volatility(
            weights,
            historical_covariance,
        )

        stressed_vol = portfolio_volatility(
            weights,
            stressed_covariance,
        )

        historical_var, historical_es = (
            calculate_parametric_var_es(
                weights,
                historical_covariance,
            )
        )

        stressed_var, stressed_es = (
            calculate_parametric_var_es(
                weights,
                stressed_covariance,
            )
        )

        results.append({
            "strategy": strategy,
            "covariance_window_observations": len(recent),
            "covariance_window_start": recent.index.min(),
            "covariance_window_end": recent.index.max(),
            "volatility_multiplier": VOLATILITY_MULTIPLIER,
            "correlation_stress_weight": (
                CORRELATION_STRESS_WEIGHT
            ),
            "historical_annualized_volatility": (
                historical_vol * np.sqrt(TRADING_DAYS)
            ),
            "stressed_annualized_volatility": (
                stressed_vol * np.sqrt(TRADING_DAYS)
            ),
            "volatility_increase": (
                stressed_vol / historical_vol - 1.0
                if historical_vol > 0
                else np.nan
            ),
            "historical_parametric_var_99": historical_var,
            "stressed_parametric_var_99": stressed_var,
            "historical_parametric_es_99": historical_es,
            "stressed_parametric_es_99": stressed_es,
            "historical_var_capital_amount": (
                historical_var * INITIAL_CAPITAL
            ),
            "stressed_var_capital_amount": (
                stressed_var * INITIAL_CAPITAL
            ),
            "historical_es_capital_amount": (
                historical_es * INITIAL_CAPITAL
            ),
            "stressed_es_capital_amount": (
                stressed_es * INITIAL_CAPITAL
            ),
        })

    result = pd.DataFrame(results)

    print(
        "Covariance stress uses the trailing "
        f"{COVARIANCE_LOOKBACK} complete sessions."
    )

    return result


# ============================================================
# 8. CHARTS
# ============================================================

def create_charts(
    historical: pd.DataFrame,
    hypothetical: pd.DataFrame,
    covariance: pd.DataFrame,
) -> None:
    print("\nGenerating stress-testing charts...")

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Chart 1: Historical crisis performance.
    historical_pivot = historical.pivot(
        index="strategy",
        columns="historical_period",
        values="cumulative_return",
    ) * 100

    ax = historical_pivot.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Historical Crisis and Recovery Performance",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Strategy")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Historical period")

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "historical_crisis_returns.png"
    )
    plt.close()

    # Chart 2: Broad market scenarios.
    broad = hypothetical[
        hypothetical["scenario_type"] == "Broad market"
    ].copy()

    broad["shock"] = broad["scenario"].str.extract(
        r"(-?\d+)%"
    ).astype(float)

    broad_pivot = broad.pivot(
        index="shock",
        columns="strategy",
        values="portfolio_return",
    ) * 100

    ax = broad_pivot.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Portfolio Sensitivity to Broad Market Shocks",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel("Estimated portfolio loss / return (%)")
    ax.set_xlabel("Market shock (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Strategy")

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "broad_market_stress.png"
    )
    plt.close()

    # Chart 3: Asset and sector shocks.
    asset_sector = hypothetical[
        hypothetical["scenario_type"].isin(
            ["Individual asset", "Sector"]
        )
    ].copy()

    asset_sector = asset_sector.sort_values(
        "portfolio_return"
    )

    fig, ax = plt.subplots(figsize=(15, 9))

    for strategy, frame in asset_sector.groupby("strategy"):

        ax.plot(
            frame["portfolio_return"].to_numpy() * 100,
            np.arange(len(frame)),
            marker="o",
            linestyle="none",
            label=strategy,
        )

    ax.set_title(
        "Individual Asset and Sector Shock Impact",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Estimated portfolio return (%)")
    ax.set_ylabel("Scenario index")
    ax.legend(title="Strategy")

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "asset_sector_stress.png"
    )
    plt.close()

    # Chart 4: Volatility stress.
    fig, ax = plt.subplots(figsize=(12, 7))

    positions = np.arange(len(covariance))
    width = 0.36

    ax.bar(
        positions - width / 2,
        covariance["historical_annualized_volatility"] * 100,
        width,
        label="Historical",
    )

    ax.bar(
        positions + width / 2,
        covariance["stressed_annualized_volatility"] * 100,
        width,
        label="Stressed",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        covariance["strategy"],
        rotation=20,
        ha="right",
    )

    ax.set_title(
        "Historical vs Stressed Annualized Volatility",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel("Annualized volatility (%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "volatility_correlation_stress.png"
    )
    plt.close()

    print("All four stress-testing charts generated.")


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main() -> None:

    print("=" * 80)
    print("PORTFOLIO STRESS TESTING AND SCENARIO ANALYSIS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily, equity = load_inputs()

    assets = list(equity.columns)

    weights_by_strategy = extract_latest_weights(
        daily,
        assets,
    )

    historical = run_historical_stress(daily)

    hypothetical = run_hypothetical_scenarios(
        weights_by_strategy
    )

    covariance = run_covariance_stress(
        equity,
        weights_by_strategy,
    )

    # --------------------------------------------------------
    # Save analytical datasets.
    # --------------------------------------------------------

    print("\nExporting stress-testing datasets...")

    save_table(
        historical,
        "historical_stress_results",
    )

    save_table(
        hypothetical,
        "hypothetical_scenario_results",
    )

    save_table(
        covariance,
        "volatility_correlation_stress",
    )

    # --------------------------------------------------------
    # Generate charts.
    # --------------------------------------------------------

    create_charts(
        historical,
        hypothetical,
        covariance,
    )

    # --------------------------------------------------------
    # Print important results.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HISTORICAL STRESS TEST RESULTS")
    print("=" * 80)

    print(
        historical[
            [
                "strategy",
                "historical_period",
                "cumulative_return",
                "maximum_drawdown",
                "historical_var_99",
                "historical_cvar_99",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("HYPOTHETICAL SCENARIO RESULTS")
    print("=" * 80)

    broad = hypothetical[
        hypothetical["scenario_type"] == "Broad market"
    ]

    print(
        broad[
            [
                "strategy",
                "scenario",
                "portfolio_return",
                "estimated_pnl",
                "stressed_portfolio_value",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("VOLATILITY AND CORRELATION STRESS RESULTS")
    print("=" * 80)

    print(
        covariance[
            [
                "strategy",
                "historical_annualized_volatility",
                "stressed_annualized_volatility",
                "historical_parametric_var_99",
                "stressed_parametric_var_99",
                "historical_parametric_es_99",
                "stressed_parametric_es_99",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("STRESS TESTING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(
        "Generated data files: "
        f"{len(list(OUTPUT_DIR.glob('*.csv'))) + len(list(OUTPUT_DIR.glob('*.parquet')))}"
    )

    print(
        "Generated charts: "
        f"{len(list(FIGURE_DIR.glob('*.png')))}"
    )


if __name__ == "__main__":
    main()
