
"""Black-Litterman portfolio return estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_covariance(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Validate an annualized asset covariance matrix."""

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")

    if covariance.empty:
        raise ValueError("Covariance matrix cannot be empty.")

    if covariance.index.has_duplicates:
        raise ValueError("Covariance index labels must be unique.")

    if covariance.columns.has_duplicates:
        raise ValueError("Covariance column labels must be unique.")

    if set(covariance.index) != set(covariance.columns):
        raise ValueError("Covariance row and column labels must match.")

    covariance = covariance.loc[
        covariance.index, covariance.index
    ].astype(float)

    values = covariance.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError("Covariance matrix contains non-finite values.")

    if not np.allclose(values, values.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(values)

    if eigenvalues.min() <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    return covariance


def _validate_market_weights(
    market_weights: pd.Series,
    assets: pd.Index,
) -> pd.Series:
    """Validate and align market capitalization weights."""

    if not isinstance(market_weights, pd.Series):
        raise TypeError("market_weights must be a pandas Series.")

    if market_weights.index.has_duplicates:
        raise ValueError("Market weight labels must be unique.")

    if set(market_weights.index) != set(assets):
        raise ValueError(
            "Market weights must contain exactly the covariance assets."
        )

    weights = market_weights.reindex(assets).astype(float)

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Market weights must be finite.")

    if (weights < 0).any():
        raise ValueError("Market weights cannot be negative.")

    if weights.sum() <= 0:
        raise ValueError("Market weights must have a positive sum.")

    return weights / weights.sum()


def market_implied_equilibrium_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """
    Calculate market-implied equilibrium excess returns.

    Pi = delta * Sigma * w_market

    The returned values are equilibrium excess returns.
    """

    covariance = _validate_covariance(covariance)

    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    weights = _validate_market_weights(
        market_weights,
        covariance.index,
    )

    equilibrium = (
        risk_aversion
        * covariance.to_numpy()
        @ weights.to_numpy()
    )

    return pd.Series(
        equilibrium,
        index=covariance.index,
        name="equilibrium_excess_return",
    )


def build_view_covariance(
    covariance: pd.DataFrame,
    view_matrix: pd.DataFrame,
    tau: float = 0.05,
    confidence: float | list[float] | np.ndarray = 0.50,
) -> pd.DataFrame:
    """
    Build diagonal view uncertainty using confidence levels.

    Omega_ii = tau * (P Sigma P.T)_ii * (1 - c_i) / c_i

    Confidence must be strictly between zero and one.
    """

    covariance = _validate_covariance(covariance)

    if not isinstance(view_matrix, pd.DataFrame):
        raise TypeError("view_matrix must be a pandas DataFrame.")

    if view_matrix.empty:
        raise ValueError("At least one investor view is required.")

    if view_matrix.columns.has_duplicates:
        raise ValueError("View asset labels must be unique.")

    if set(view_matrix.columns) != set(covariance.columns):
        raise ValueError(
            "View matrix columns must match covariance assets."
        )

    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and positive.")

    p = view_matrix.reindex(
        columns=covariance.columns
    ).to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise ValueError("View matrix must contain finite values.")

    if np.allclose(p, 0.0):
        raise ValueError("View matrix cannot contain only zero rows.")

    confidence_values = np.asarray(
        confidence,
        dtype=float,
    )

    if confidence_values.ndim == 0:
        confidence_values = np.full(
            len(view_matrix),
            float(confidence_values),
        )

    if confidence_values.shape != (len(view_matrix),):
        raise ValueError(
            "Provide one confidence value per investor view."
        )

    if (
        not np.isfinite(confidence_values).all()
        or (confidence_values <= 0).any()
        or (confidence_values >= 1).any()
    ):
        raise ValueError(
            "Confidence levels must be strictly between zero and one."
        )

    view_variance = (
        p @ covariance.to_numpy() @ p.T
    )

    diagonal = (
        tau
        * np.diag(view_variance)
        * (1.0 - confidence_values)
        / confidence_values
    )

    if (diagonal <= 0).any():
        raise ValueError(
            "Every investor view must have positive uncertainty."
        )

    omega = np.diag(diagonal)

    return pd.DataFrame(
        omega,
        index=view_matrix.index,
        columns=view_matrix.index,
    )


def black_litterman_posterior(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    view_matrix: pd.DataFrame,
    view_returns: pd.Series,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    confidence: float | list[float] | np.ndarray = 0.50,
) -> tuple[
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Estimate Black-Litterman posterior returns and covariance.

    Returns:
        posterior_returns:
            Posterior expected excess returns.

        posterior_covariance:
            Posterior predictive covariance estimate.

        equilibrium_returns:
            Market-implied equilibrium excess returns.

        omega:
            Investor view uncertainty covariance.
    """

    covariance = _validate_covariance(covariance)

    if not isinstance(view_matrix, pd.DataFrame):
        raise TypeError("view_matrix must be a pandas DataFrame.")

    if not isinstance(view_returns, pd.Series):
        raise TypeError("view_returns must be a pandas Series.")

    if view_matrix.empty:
        raise ValueError("At least one investor view is required.")

    if view_matrix.index.has_duplicates:
        raise ValueError("View identifiers must be unique.")

    if view_returns.index.has_duplicates:
        raise ValueError("View return identifiers must be unique.")

    if set(view_matrix.index) != set(view_returns.index):
        raise ValueError(
            "View matrix rows must match view return labels."
        )

    views = view_matrix.reindex(
        index=view_returns.index,
        columns=covariance.columns,
    )

    q = view_returns.to_numpy(dtype=float)

    if not np.isfinite(q).all():
        raise ValueError("View returns must be finite.")

    p = views.to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise ValueError("View matrix must contain finite values.")

    if np.allclose(p, 0.0):
        raise ValueError("View matrix cannot contain only zero rows.")

    equilibrium = market_implied_equilibrium_returns(
        covariance,
        market_weights,
        risk_aversion=risk_aversion,
    )

    omega = build_view_covariance(
        covariance,
        views,
        tau=tau,
        confidence=confidence,
    )

    sigma = covariance.to_numpy()
    pi = equilibrium.to_numpy()
    omega_values = omega.to_numpy()

    prior_covariance = tau * sigma

    # Posterior covariance of the latent expected return.
    prior_precision = np.linalg.solve(
        prior_covariance,
        np.eye(len(covariance)),
    )

    view_precision = np.linalg.solve(
        omega_values,
        np.eye(len(omega)),
    )

    posterior_precision = (
        prior_precision
        + p.T @ view_precision @ p
    )

    posterior_mean_covariance = np.linalg.solve(
        posterior_precision,
        np.eye(len(covariance)),
    )

    posterior_mean = posterior_mean_covariance @ (
        prior_precision @ pi
        + p.T @ view_precision @ q
    )

    # Predictive covariance includes the prior return covariance.
    posterior_predictive_covariance = (
        sigma + posterior_mean_covariance
    )

    posterior_predictive_covariance = (
        posterior_predictive_covariance
        + posterior_predictive_covariance.T
    ) / 2.0

    posterior_returns = pd.Series(
        posterior_mean,
        index=covariance.index,
        name="posterior_excess_return",
    )

    posterior_covariance = pd.DataFrame(
        posterior_predictive_covariance,
        index=covariance.index,
        columns=covariance.columns,
    )

    if not np.isfinite(posterior_returns.to_numpy()).all():
        raise RuntimeError(
            "Posterior returns contain non-finite values."
        )

    if not np.isfinite(posterior_covariance.to_numpy()).all():
        raise RuntimeError(
            "Posterior covariance contains non-finite values."
        )

    if np.linalg.eigvalsh(
        posterior_covariance.to_numpy()
    ).min() <= 0:
        raise RuntimeError(
            "Posterior covariance is not positive definite."
        )

    return (
        posterior_returns,
        posterior_covariance,
        equilibrium,
        omega,
    )



