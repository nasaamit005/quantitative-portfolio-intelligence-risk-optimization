"""Equal Risk Contribution (Risk Parity) portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate historical simple returns."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("Returns data cannot be empty.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset labels must be unique.")

    data = returns.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.shape[0] < 2:
        raise ValueError("At least two complete observations are required.")

    if data.shape[1] < 2:
        raise ValueError("Risk parity requires at least two assets.")

    if (data <= -1.0).any().any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    if (data.std(ddof=1) <= 0).any():
        raise ValueError("Every asset must have positive volatility.")

    return data.astype(float)


def calculate_risk_contributions(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate marginal and total risk contributions.

    Risk contribution:
        RC_i = w_i * (Sigma @ w)_i / portfolio_volatility
    """

    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series.")

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")

    if weights.empty or covariance.empty:
        raise ValueError("Weights and covariance cannot be empty.")

    if weights.index.has_duplicates:
        raise ValueError("Weight asset labels must be unique.")

    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise ValueError("Covariance asset labels must be unique.")

    if set(weights.index) != set(covariance.index):
        raise ValueError("Weight and covariance asset labels must match.")

    covariance = covariance.loc[weights.index, weights.index]

    w = weights.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    if not np.isfinite(w).all() or not np.isfinite(sigma).all():
        raise ValueError("Inputs must contain finite values.")

    if (w < 0).any():
        raise ValueError("Risk contribution calculation requires long-only weights.")

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")

    portfolio_variance = float(w @ sigma @ w)

    if portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive.")

    portfolio_volatility = np.sqrt(portfolio_variance)

    marginal_risk = (sigma @ w) / portfolio_volatility

    risk_contribution = w * marginal_risk

    risk_contribution_pct = (
        risk_contribution / portfolio_volatility
    )

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_risk": marginal_risk,
            "risk_contribution": risk_contribution,
            "risk_contribution_pct": risk_contribution_pct,
        },
        index=weights.index,
    )


def equal_risk_contribution_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.50,
    tolerance: float = 1e-10,
    max_iterations: int = 2000,
) -> pd.Series:
    """
    Calculate equal risk contribution portfolio weights.

    The optimizer minimizes the squared difference between
    actual risk contribution percentages and equal risk budgets.
    """

    data = _validate_returns(returns)

    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1].")

    if max_weight * data.shape[1] < 1.0:
        raise ValueError(
            "Maximum weight is too restrictive for a fully invested portfolio."
        )

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    covariance = data.cov().to_numpy(dtype=float)

    if not np.isfinite(covariance).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(covariance)

    if eigenvalues.min() <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    asset_count = data.shape[1]

    risk_budget = np.full(asset_count, 1.0 / asset_count)

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(
            weights @ covariance @ weights
        )

        if portfolio_variance <= 0:
            return 1e10

        contributions = weights * (covariance @ weights)

        contribution_shares = (
            contributions / portfolio_variance
        )

        differences = contribution_shares - risk_budget

        return float(differences @ differences)

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
        }
    ]

    bounds = [(0.0, max_weight)] * asset_count

    # Equal-weight and inverse-volatility starting points.
    initial_weights = [
        np.full(asset_count, 1.0 / asset_count),
    ]

    inverse_volatility = 1.0 / np.sqrt(np.diag(covariance))
    inverse_volatility /= inverse_volatility.sum()

    if inverse_volatility.max() <= max_weight:
        initial_weights.append(inverse_volatility)

    best_result = None

    for initial in initial_weights:
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": tolerance,
                "maxiter": max_iterations,
                "disp": False,
            },
        )

        if (
            result.success
            and np.isfinite(result.fun)
            and (
                best_result is None
                or result.fun < best_result.fun
            )
        ):
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "Risk parity optimization failed to converge."
        )

    weights = pd.Series(
        best_result.x,
        index=data.columns,
        name="risk_parity_weight",
    )

    weights = weights.clip(lower=0.0)
    weights /= weights.sum()

    if not np.isfinite(weights.to_numpy()).all():
        raise RuntimeError("Risk parity produced non-finite weights.")

    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError("Risk parity weights do not sum to one.")

    if (weights > max_weight + 1e-7).any():
        raise RuntimeError("Risk parity exceeded the maximum weight.")

    return weights