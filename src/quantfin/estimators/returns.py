
"""Expected-return and covariance estimation for portfolio optimization."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

CovarianceMethod = Literal["sample", "ledoit_wolf"]
ReturnMethod = Literal["historical_mean"]


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a DataFrame of asset simple returns."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns DataFrame is empty.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset columns must be unique.")

    data = returns.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    if data.empty:
        raise ValueError("No complete finite return observations remain.")

    if data.shape[0] < 2:
        raise ValueError("At least two complete observations are required.")

    if (data <= -1.0).any().any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    return data.astype(float)


def estimate_expected_returns(
    returns: pd.DataFrame,
    method: ReturnMethod = "historical_mean",
    periods_per_year: int = 252,
) -> pd.Series:
    """Estimate annualized expected arithmetic returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    data = _validate_returns(returns)

    if method == "historical_mean":
        expected_returns = data.mean() * periods_per_year
    else:
        raise ValueError(f"Unsupported expected-return method: {method}")

    expected_returns.name = "expected_annual_return"
    return expected_returns


def estimate_covariance(
    returns: pd.DataFrame,
    method: CovarianceMethod = "ledoit_wolf",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate an annualized asset covariance matrix."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    data = _validate_returns(returns)

    if method == "sample":
        covariance = data.cov()
    elif method == "ledoit_wolf":
        estimator = LedoitWolf(assume_centered=False)
        estimator.fit(data.to_numpy())
        covariance = pd.DataFrame(
            estimator.covariance_,
            index=data.columns,
            columns=data.columns,
        )
    else:
        raise ValueError(f"Unsupported covariance method: {method}")

    annualized = covariance * periods_per_year
    annualized.index.name = "asset"
    annualized.columns.name = "asset"

    return annualized


def estimate_market_inputs(
    returns: pd.DataFrame,
    return_method: ReturnMethod = "historical_mean",
    covariance_method: CovarianceMethod = "ledoit_wolf",
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate aligned expected returns and annualized covariance."""
    data = _validate_returns(returns)

    expected_returns = estimate_expected_returns(
        data,
        method=return_method,
        periods_per_year=periods_per_year,
    )

    covariance = estimate_covariance(
        data,
        method=covariance_method,
        periods_per_year=periods_per_year,
    )

    covariance = covariance.loc[expected_returns.index, expected_returns.index]

    if not np.isfinite(expected_returns.to_numpy()).all():
        raise ValueError("Expected returns contain non-finite values.")

    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    if not np.allclose(covariance.to_numpy(), covariance.to_numpy().T, atol=1e-10):
        raise ValueError("Covariance matrix is not symmetric.")

    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance.to_numpy()).min())
    if minimum_eigenvalue <= 0:
        raise ValueError(
            "Covariance matrix is not positive definite. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.6e}"
        )

    return expected_returns, covariance
