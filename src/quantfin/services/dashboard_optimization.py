"""Interactive, read-only portfolio optimization service for the Flask dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from quantfin.estimators.returns import estimate_market_inputs
from quantfin.optimization import (
    PortfolioConstraints,
    maximum_sharpe_weights,
    minimum_volatility_weights,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
RETURNS_FILE = ROOT_DIR / "data" / "processed" / "us_equities_simple_returns.csv"

TRADING_DAYS = 252
DEFAULT_LOOKBACK = 252
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 0.50
DEFAULT_FRONTIER_POINTS = 25


class DashboardOptimizationError(ValueError):
    """Raised when dashboard optimization inputs or calculations are invalid."""


def _finite_float(value: Any, name: str) -> float:
    """Convert a value to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DashboardOptimizationError(
            f"{name} must be a valid number."
        ) from exc

    if not np.isfinite(result):
        raise DashboardOptimizationError(f"{name} must be finite.")

    return result


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the historical returns matrix."""
    if not isinstance(returns, pd.DataFrame):
        raise DashboardOptimizationError(
            "The returns data must be a pandas DataFrame."
        )

    if returns.empty:
        raise DashboardOptimizationError("The returns dataset is empty.")

    cleaned = returns.copy()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.dropna(axis=0, how="any")

    if cleaned.empty:
        raise DashboardOptimizationError(
            "No complete observations remain after cleaning the returns data."
        )

    if cleaned.shape[1] < 2:
        raise DashboardOptimizationError(
            "At least two assets are required for portfolio optimization."
        )

    if cleaned.columns.duplicated().any():
        raise DashboardOptimizationError(
            "The returns dataset contains duplicate asset columns."
        )

    if not np.isfinite(cleaned.to_numpy(dtype=float)).all():
        raise DashboardOptimizationError(
            "The cleaned returns dataset contains non-finite values."
        )

    if (cleaned.var(axis=0) <= 0).any():
        invalid = cleaned.columns[cleaned.var(axis=0) <= 0].tolist()
        raise DashboardOptimizationError(
            f"These assets have zero variance: {', '.join(invalid)}."
        )

    return cleaned.astype(float)


def load_equity_returns() -> pd.DataFrame:
    """Load the project's processed daily equity returns."""
    if not RETURNS_FILE.exists():
        raise DashboardOptimizationError(
            "Processed equity returns were not found. Expected file: "
            f"{RETURNS_FILE}"
        )

    try:
        data = pd.read_csv(RETURNS_FILE)
    except (OSError, pd.errors.ParserError) as exc:
        raise DashboardOptimizationError(
            f"Could not read the processed returns file: {exc}"
        ) from exc

    if "Date" not in data.columns:
        raise DashboardOptimizationError(
            "The processed returns CSV must contain a Date column."
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    data = data.sort_values("Date").set_index("Date")

    returns = _clean_returns(data)

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise DashboardOptimizationError(
            "The returns index must contain valid dates."
        )

    return returns


def _validate_settings(
    returns: pd.DataFrame,
    objective: str,
    lookback: int,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    frontier_points: int,
) -> tuple[pd.DataFrame, PortfolioConstraints]:
    """Validate the requested optimization settings."""
    if objective not in {"maximum_sharpe", "minimum_volatility"}:
        raise DashboardOptimizationError(
            "Objective must be maximum_sharpe or minimum_volatility."
        )

    if isinstance(lookback, bool) or not isinstance(lookback, int):
        raise DashboardOptimizationError(
            "Lookback must be a whole number of trading days."
        )

    if lookback < 63:
        raise DashboardOptimizationError(
            "Lookback must be at least 63 trading days."
        )

    if lookback > len(returns):
        raise DashboardOptimizationError(
            f"Lookback cannot exceed the available {len(returns)} observations."
        )

    if isinstance(frontier_points, bool) or not isinstance(
        frontier_points, int
    ):
        raise DashboardOptimizationError(
            "Frontier points must be a whole number."
        )

    if not 10 <= frontier_points <= 60:
        raise DashboardOptimizationError(
            "Frontier points must be between 10 and 60."
        )

    if not -0.05 <= risk_free_rate <= 0.25:
        raise DashboardOptimizationError(
            "Risk-free rate must be between -5% and 25%."
        )

    if not 0.0 <= min_weight <= 1.0:
        raise DashboardOptimizationError(
            "Minimum weight must be between 0% and 100%."
        )

    if not 0.01 <= max_weight <= 1.0:
        raise DashboardOptimizationError(
            "Maximum weight must be between 1% and 100%."
        )

    if min_weight > max_weight:
        raise DashboardOptimizationError(
            "Minimum weight cannot exceed maximum weight."
        )

    selected_returns = returns.tail(lookback).copy()
    asset_count = selected_returns.shape[1]

    if asset_count * min_weight > 1.0 + 1e-9:
        raise DashboardOptimizationError(
            "The minimum weight is infeasible for the number of assets."
        )

    if asset_count * max_weight < 1.0 - 1e-9:
        raise DashboardOptimizationError(
            "The maximum weight is too low to fully invest the portfolio."
        )

    constraints = PortfolioConstraints(
        min_weight=min_weight,
        max_weight=max_weight,
        allow_short_selling=False,
        fully_invested=True,
        weight_sum=1.0,
    )

    return selected_returns, constraints


def _normalise_weights(weights: Any, assets: list[str]) -> pd.Series:
    """Convert optimizer output to a validated long-only weight series."""
    if isinstance(weights, pd.Series):
        result = weights.reindex(assets).astype(float)
    elif isinstance(weights, dict):
        result = pd.Series(weights, dtype=float).reindex(assets)
    else:
        result = pd.Series(np.asarray(weights, dtype=float), index=assets)

    if result.isna().any():
        raise DashboardOptimizationError(
            "The optimizer returned missing asset weights."
        )

    values = result.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise DashboardOptimizationError(
            "The optimizer returned non-finite asset weights."
        )

    if (values < -1e-7).any():
        raise DashboardOptimizationError(
            "The optimizer returned negative weights."
        )

    values = np.clip(values, 0.0, None)
    total = float(values.sum())

    if total <= 0:
        raise DashboardOptimizationError(
            "The optimizer returned an allocation with zero total weight."
        )

    result = pd.Series(values / total, index=assets, dtype=float)

    if not np.isclose(result.sum(), 1.0, atol=1e-7):
        raise DashboardOptimizationError(
            "The optimized portfolio is not fully invested."
        )

    return result


def _portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> dict[str, float]:
    """Calculate annualized portfolio return, volatility and Sharpe ratio."""
    annual_return = float(weights @ expected_returns)
    variance = float(weights @ covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))

    if volatility <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((annual_return - risk_free_rate) / volatility)

    return {
        "expected_annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
    }


def _calculate_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints,
    risk_free_rate: float,
    points: int,
) -> list[dict[str, float]]:
    """Calculate feasible long-only efficient-frontier portfolios."""
    assets = expected_returns.index.tolist()
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.loc[assets, assets].to_numpy(dtype=float)
    asset_count = len(assets)
    bounds = constraints.bounds(asset_count)

    objective = np.zeros(asset_count, dtype=float)

    minimum_return = linprog(
        c=mu,
        A_eq=np.ones((1, asset_count)),
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )

    maximum_return = linprog(
        c=-mu,
        A_eq=np.ones((1, asset_count)),
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )

    if not minimum_return.success or not maximum_return.success:
        raise DashboardOptimizationError(
            "Could not determine feasible portfolio return boundaries."
        )

    low_return = float(minimum_return.fun)
    high_return = float(-maximum_return.fun)

    if high_return - low_return <= 1e-10:
        raise DashboardOptimizationError(
            "The feasible return range is too narrow to calculate a frontier."
        )

    targets = np.linspace(low_return, high_return, points)
    frontier: list[dict[str, float]] = []

    equality_matrix = np.vstack([np.ones(asset_count), mu])

    for target in targets:
        equality_values = np.array([1.0, float(target)])

        initial = linprog(
            c=objective,
            A_eq=equality_matrix,
            b_eq=equality_values,
            bounds=bounds,
            method="highs",
        )

        if not initial.success:
            continue

        result = minimize(
            fun=lambda weights: float(weights @ sigma @ weights),
            x0=initial.x,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda weights: float(np.sum(weights) - 1.0),
                },
                {
                    "type": "eq",
                    "fun": lambda weights, target_value=float(target): float(
                        weights @ mu - target_value
                    ),
                },
            ],
            options={
                "maxiter": 2000,
                "ftol": 1e-11,
                "disp": False,
            },
        )

        if not result.success:
            continue

        weights = np.asarray(result.x, dtype=float)

        if not np.isfinite(weights).all():
            continue

        if not np.isclose(weights.sum(), 1.0, atol=1e-5):
            continue

        if (weights < -1e-6).any():
            continue

        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

        metrics = _portfolio_metrics(
            weights,
            mu,
            sigma,
            risk_free_rate,
        )

        frontier.append(
            {
                "expected_annual_return": metrics["expected_annual_return"],
                "annual_volatility": metrics["annual_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
            }
        )

    if len(frontier) < max(5, points // 2):
        raise DashboardOptimizationError(
            "Too few efficient-frontier portfolios converged. "
            "Try a different lookback period or weight limit."
        )

    return frontier


def _calculate_risk_contributions(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> list[dict[str, float | str]]:
    """Calculate each asset's percentage contribution to portfolio volatility."""
    assets = weights.index.tolist()
    weight_values = weights.to_numpy(dtype=float)
    sigma = covariance.loc[assets, assets].to_numpy(dtype=float)

    portfolio_variance = float(weight_values @ sigma @ weight_values)

    if portfolio_variance <= 0:
        raise DashboardOptimizationError(
            "Portfolio variance must be positive to calculate risk contributions."
        )

    portfolio_volatility = float(np.sqrt(portfolio_variance))
    marginal_risk = sigma @ weight_values / portfolio_volatility
    component_risk = weight_values * marginal_risk
    total_component_risk = float(component_risk.sum())

    if abs(total_component_risk) <= 1e-12:
        raise DashboardOptimizationError(
            "Could not calculate meaningful risk contributions."
        )

    contributions = component_risk / total_component_risk

    return [
        {
            "asset": asset,
            "weight": float(weights.loc[asset]),
            "risk_contribution_pct": float(contribution),
            "marginal_risk": float(marginal),
        }
        for asset, contribution, marginal in zip(
            assets,
            contributions,
            marginal_risk,
            strict=True,
        )
    ]


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""
    records = frame.reset_index().to_dict(orient="records")
    safe_records: list[dict[str, Any]] = []

    for record in records:
        safe_record: dict[str, Any] = {}

        for key, value in record.items():
            if pd.isna(value):
                safe_record[str(key)] = None
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                safe_record[str(key)] = pd.Timestamp(value).isoformat()
            elif isinstance(value, (np.integer,)):
                safe_record[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                safe_record[str(key)] = float(value)
            elif isinstance(value, (np.bool_,)):
                safe_record[str(key)] = bool(value)
            else:
                safe_record[str(key)] = value

        safe_records.append(safe_record)

    return safe_records


def optimize_dashboard_portfolio(
    objective: str = "maximum_sharpe",
    lookback: int = DEFAULT_LOOKBACK,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    frontier_points: int = DEFAULT_FRONTIER_POINTS,
) -> dict[str, Any]:
    """Run interactive historical portfolio optimization for the dashboard."""
    objective = str(objective).strip().lower()
    lookback = int(lookback)
    risk_free_rate = _finite_float(risk_free_rate, "Risk-free rate")
    min_weight = _finite_float(min_weight, "Minimum weight")
    max_weight = _finite_float(max_weight, "Maximum weight")
    frontier_points = int(frontier_points)

    all_returns = load_equity_returns()

    selected_returns, constraints = _validate_settings(
        returns=all_returns,
        objective=objective,
        lookback=lookback,
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        frontier_points=frontier_points,
    )

    try:
        expected_returns, covariance = estimate_market_inputs(
            selected_returns,
            return_method="historical_mean",
            covariance_method="ledoit_wolf",
            periods_per_year=TRADING_DAYS,
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        raise DashboardOptimizationError(
            f"Market-input estimation failed: {exc}"
        ) from exc

    assets = expected_returns.index.tolist()
    covariance = covariance.loc[assets, assets]

    if not np.isfinite(expected_returns.to_numpy(dtype=float)).all():
        raise DashboardOptimizationError(
            "Expected returns contain non-finite values."
        )

    if not np.isfinite(covariance.to_numpy(dtype=float)).all():
        raise DashboardOptimizationError(
            "Covariance matrix contains non-finite values."
        )

    try:
        maximum_sharpe = _normalise_weights(
            maximum_sharpe_weights(
                expected_returns,
                covariance,
                risk_free_rate=risk_free_rate,
                constraints=constraints,
            ),
            assets,
        )

        minimum_volatility = _normalise_weights(
            minimum_volatility_weights(
                expected_returns,
                covariance,
                constraints=constraints,
            ),
            assets,
        )
    except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise DashboardOptimizationError(
            f"Portfolio optimization failed: {exc}"
        ) from exc

    selected_weights = (
        maximum_sharpe
        if objective == "maximum_sharpe"
        else minimum_volatility
    )

    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    selected_metrics = _portfolio_metrics(
        selected_weights.to_numpy(dtype=float),
        mu,
        sigma,
        risk_free_rate,
    )

    max_sharpe_metrics = _portfolio_metrics(
        maximum_sharpe.to_numpy(dtype=float),
        mu,
        sigma,
        risk_free_rate,
    )

    min_volatility_metrics = _portfolio_metrics(
        minimum_volatility.to_numpy(dtype=float),
        mu,
        sigma,
        risk_free_rate,
    )

    frontier = _calculate_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        constraints=constraints,
        risk_free_rate=risk_free_rate,
        points=frontier_points,
    )

    allocation = [
        {
            "asset": asset,
            "weight": float(selected_weights.loc[asset]),
            "weight_pct": float(selected_weights.loc[asset] * 100.0),
            "expected_return": float(expected_returns.loc[asset]),
            "annual_volatility": float(
                np.sqrt(covariance.loc[asset, asset])
            ),
        }
        for asset in assets
    ]

    risk_contributions = _calculate_risk_contributions(
        selected_weights,
        covariance,
    )

    correlation = selected_returns.corr()

    return {
        "status": "success",
        "objective": objective,
        "estimation": {
            "method": "Historical arithmetic mean returns and Ledoit-Wolf covariance",
            "lookback_days": int(lookback),
            "observations": int(len(selected_returns)),
            "start_date": selected_returns.index.min().strftime("%Y-%m-%d"),
            "end_date": selected_returns.index.max().strftime("%Y-%m-%d"),
            "asset_count": int(len(assets)),
            "risk_free_rate": float(risk_free_rate),
            "min_weight": float(min_weight),
            "max_weight": float(max_weight),
            "rebalancing": "Single-period historical in-sample optimization",
        },
        "selected_metrics": selected_metrics,
        "maximum_sharpe_metrics": max_sharpe_metrics,
        "minimum_volatility_metrics": min_volatility_metrics,
        "allocation": allocation,
        "risk_contributions": risk_contributions,
        "efficient_frontier": frontier,
        "correlation": {
            "assets": assets,
            "values": correlation.to_numpy(dtype=float).tolist(),
        },
    }