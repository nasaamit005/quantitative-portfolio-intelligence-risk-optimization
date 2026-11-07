
"""Mean-variance portfolio optimization algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from quantfin.optimization.constraints import (
    PortfolioConstraints,
    validate_portfolio_weights,
)


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[float, float]]]:
    """Validate and align expected returns and covariance inputs."""

    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series.")

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")

    if expected_returns.empty:
        raise ValueError("Expected returns cannot be empty.")

    if expected_returns.index.has_duplicates:
        raise ValueError("Asset labels must be unique.")

    assets = expected_returns.index.tolist()

    if set(assets) != set(covariance.index) or set(assets) != set(
        covariance.columns
    ):
        raise ValueError(
            "Expected returns and covariance must contain identical assets."
        )

    covariance = covariance.loc[assets, assets]

    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Optimization inputs contain non-finite values.")

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(sigma).min())

    if minimum_eigenvalue <= 0:
        raise ValueError(
            "Covariance matrix must be positive definite. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.6e}"
        )

    bounds = constraints.bounds(len(assets))

    return mu, sigma, assets, bounds


def portfolio_performance(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Calculate annualized portfolio return, volatility and Sharpe ratio."""

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    assets = expected_returns.index.tolist()

    if not weights.index.equals(expected_returns.index):
        weights = weights.reindex(assets)

    if weights.isna().any():
        raise ValueError("Portfolio weights and expected returns are misaligned.")

    covariance = covariance.loc[assets, assets]

    weight_array = weights.to_numpy(dtype=float)
    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    annual_return = float(weight_array @ mu)
    variance = float(weight_array @ sigma @ weight_array)
    volatility = float(np.sqrt(max(variance, 0.0)))

    sharpe_ratio = (
        (annual_return - risk_free_rate) / volatility
        if volatility > 1e-12
        else np.nan
    )

    return {
        "expected_annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
    }


def _build_constraints(
    mu: np.ndarray,
    target_return: float | None = None,
    fully_invested: bool = True,
    weight_sum: float = 1.0,
) -> list[dict]:
    """Build equality constraints for the numerical optimizer."""

    conditions = []

    if fully_invested:
        conditions.append(
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - weight_sum,
            }
        )

    if target_return is not None:
        conditions.append(
            {
                "type": "eq",
                "fun": lambda weights: float(weights @ mu) - target_return,
            }
        )

    return conditions


def _initial_weights(
    n_assets: int,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Construct a feasible initial portfolio."""

    if constraints.fully_invested:
        return np.full(
            n_assets,
            constraints.weight_sum / n_assets,
            dtype=float,
        )

    return np.zeros(n_assets, dtype=float)


def _maximum_return_initial_weights(
    mu: np.ndarray,
    bounds: list[tuple[float, float]],
    weight_sum: float,
) -> np.ndarray:
    """Find a feasible maximum-expected-return starting portfolio."""

    result = linprog(
        c=-mu,
        A_eq=np.ones((1, len(mu))),
        b_eq=np.array([weight_sum]),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "Maximum-return initialization failed: "
            f"{result.message}"
        )

    weights = np.asarray(result.x, dtype=float)

    if not np.isfinite(weights).all():
        raise RuntimeError(
            "Maximum-return initialization returned non-finite weights."
        )

    return weights


def _validate_optimization_result(
    result: object,
    assets: list[str],
    constraints: PortfolioConstraints,
) -> pd.Series:
    """Validate optimizer status and independently check portfolio weights."""

    if not result.success:
        raise RuntimeError(str(result.message))

    weights_array = np.asarray(result.x, dtype=float)

    if weights_array.shape != (len(assets),):
        raise RuntimeError("Optimizer returned an unexpected weight shape.")

    if not np.isfinite(weights_array).all():
        raise RuntimeError("Optimizer returned non-finite weights.")

    weights = pd.Series(
        weights_array,
        index=assets,
        name="weight",
    )

    validation = validate_portfolio_weights(weights, constraints)

    if not validation["all_valid"]:
        raise RuntimeError(
            "Optimizer returned weights that failed independent validation: "
            f"{validation}"
        )

    return weights


def minimum_volatility_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Find the minimum-variance portfolio."""

    rules = constraints or PortfolioConstraints()
    _, sigma, assets, bounds = _validate_inputs(
        expected_returns, covariance, rules
    )

    initial = _initial_weights(len(assets), rules)

    result = minimize(
        fun=lambda weights: float(weights @ sigma @ weights),
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=_build_constraints(
            expected_returns.to_numpy(dtype=float),
            fully_invested=rules.fully_invested,
            weight_sum=rules.weight_sum,
        ),
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"Minimum-volatility optimization failed: {result.message}"
        )

    weights = pd.Series(result.x, index=assets, name="weight")

    validation = validate_portfolio_weights(weights, rules)

    if not validation["all_valid"]:
        raise RuntimeError("Minimum-volatility weights failed validation.")

    return weights


def maximum_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.04,
    constraints: PortfolioConstraints | None = None,
) -> pd.Series:
    """Find the portfolio with the maximum Sharpe ratio.

    The optimizer first attempts SLSQP from an equal-weight starting point.
    If that attempt fails or returns invalid weights, it retries from a
    feasible maximum-expected-return portfolio obtained using linear
    programming.

    The fallback is available for fully invested portfolios.
    """

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    rules = constraints or PortfolioConstraints()

    mu, sigma, assets, bounds = _validate_inputs(
        expected_returns,
        covariance,
        rules,
    )

    def negative_sharpe(weights: np.ndarray) -> float:
        """Calculate the negative portfolio Sharpe ratio."""

        annual_return = float(weights @ mu)
        variance = float(weights @ sigma @ weights)
        volatility = np.sqrt(max(variance, 0.0))

        if volatility <= 1e-12:
            return 1e10

        return -(annual_return - risk_free_rate) / volatility

    solver_constraints = _build_constraints(
        mu,
        fully_invested=rules.fully_invested,
        weight_sum=rules.weight_sum,
    )

    initial = _initial_weights(len(assets), rules)

    # Attempt 1: preserve the existing equal-weight initialization.
    primary_result = minimize(
        fun=negative_sharpe,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=solver_constraints,
        options={"maxiter": 3000, "ftol": 1e-12},
    )

    try:
        return _validate_optimization_result(
            primary_result,
            assets,
            rules,
        )

    except RuntimeError as primary_error:
        primary_error_message = str(primary_error)

    # Attempt 2: use a feasible maximum-return initialization.
    if not rules.fully_invested:
        raise RuntimeError(
            "Maximum-Sharpe optimization failed from the primary "
            "starting point. The maximum-return fallback currently "
            "requires fully invested portfolios. "
            f"Primary error: {primary_error_message}"
        )

    try:
        fallback_initial = _maximum_return_initial_weights(
            mu,
            bounds,
            rules.weight_sum,
        )

    except RuntimeError as fallback_initial_error:
        raise RuntimeError(
            "Maximum-Sharpe primary optimization and fallback "
            "initialization both failed. "
            f"Primary error: {primary_error_message}. "
            f"Fallback initialization error: {fallback_initial_error}"
        ) from fallback_initial_error

    fallback_result = minimize(
        fun=negative_sharpe,
        x0=fallback_initial,
        method="SLSQP",
        bounds=bounds,
        constraints=solver_constraints,
        options={"maxiter": 3000, "ftol": 1e-12},
    )

    try:
        return _validate_optimization_result(
            fallback_result,
            assets,
            rules,
        )

    except RuntimeError as fallback_error:
        raise RuntimeError(
            "Maximum-Sharpe optimization failed using both starting "
            "points. "
            f"Primary error: {primary_error_message}. "
            f"Fallback error: {fallback_error}"
        ) from fallback_error


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints | None = None,
    points: int = 50,
) -> pd.DataFrame:
    """Calculate minimum-volatility portfolios across target returns."""

    if points < 2:
        raise ValueError("At least two frontier points are required.")

    rules = constraints or PortfolioConstraints()

    if not rules.fully_invested:
        raise ValueError(
            "Efficient frontier currently requires fully invested portfolios."
        )

    mu, sigma, assets, bounds = _validate_inputs(
        expected_returns, covariance, rules
    )

    # Find the minimum feasible expected return.
    minimum_result = linprog(
        c=mu,
        A_eq=np.ones((1, len(assets))),
        b_eq=[rules.weight_sum],
        bounds=bounds,
        method="highs",
    )

    # Find the maximum feasible expected return.
    maximum_result = linprog(
        c=-mu,
        A_eq=np.ones((1, len(assets))),
        b_eq=[rules.weight_sum],
        bounds=bounds,
        method="highs",
    )

    if not minimum_result.success or not maximum_result.success:
        raise RuntimeError("Could not determine feasible return boundaries.")

    minimum_return = float(minimum_result.fun)
    maximum_return = float(-maximum_result.fun)

    target_returns = np.linspace(minimum_return, maximum_return, points)
    results = []

    for target in target_returns:
        initial = _initial_weights(len(assets), rules)

        optimization = minimize(
            fun=lambda weights: float(weights @ sigma @ weights),
            x0=initial,
            method="SLSQP",
            bounds=bounds,
            constraints=_build_constraints(
                mu,
                target_return=float(target),
                fully_invested=True,
                weight_sum=rules.weight_sum,
            ),
            options={"maxiter": 3000, "ftol": 1e-12},
        )

        if not optimization.success:
            raise RuntimeError(
                "Efficient frontier optimization failed at "
                f"target return {target:.6f}: {optimization.message}"
            )

        weights = pd.Series(optimization.x, index=assets)

        validation = validate_portfolio_weights(weights, rules)

        if not validation["all_valid"]:
            raise RuntimeError(
                f"Frontier portfolio failed validation at {target:.6f}."
            )

        performance = portfolio_performance(
            weights,
            expected_returns,
            covariance,
        )

        row = {
            "target_return": float(target),
            **performance,
        }

        row.update(
            {
                f"weight_{asset}": float(weights[asset])
                for asset in assets
            }
        )

        results.append(row)

    return pd.DataFrame(results)
