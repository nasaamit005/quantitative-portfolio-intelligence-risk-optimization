
"""Portfolio investment strategy construction."""

import numpy as np
import pandas as pd

from quantfin.optimization import (
    PortfolioConstraints,
    black_litterman_posterior,
    equal_risk_contribution_weights,
    hierarchical_risk_parity_weights,
    maximum_sharpe_weights,
    minimum_volatility_weights,
)

TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_MAX_WEIGHT = 0.50


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean historical asset returns."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty or returns.shape[1] < 2:
        raise ValueError("At least two assets with observations are required.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset columns must be unique.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must be chronologically ordered.")

    clean = returns.dropna().astype(float)

    if clean.empty:
        raise ValueError("No complete return observations are available.")

    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    if (clean <= -1.0).any().any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    return clean


def _validate_weights(weights: pd.Series) -> pd.Series:
    """Validate a fully invested long-only portfolio."""
    if not isinstance(weights, pd.Series):
        raise TypeError("Portfolio weights must be a pandas Series.")

    if weights.empty or weights.index.has_duplicates:
        raise ValueError("Weights must have unique asset labels.")

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Portfolio weights must be finite.")

    if (values < -1e-8).any():
        raise ValueError("Short positions are not supported.")

    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("Portfolio weights must sum to one.")

    return weights.clip(lower=0.0) / weights.clip(lower=0.0).sum()


def equal_weight_weights(returns: pd.DataFrame) -> pd.Series:
    """Allocate equal capital to every asset."""
    clean = _validate_returns(returns)

    return pd.Series(
        1.0 / clean.shape[1],
        index=clean.columns,
        name="Equal Weight",
    )


def market_cap_weighted_weights(
    market_caps: pd.Series,
    assets: pd.Index | None = None,
) -> pd.Series:
    """Construct weights from supplied market capitalizations.

    Market capitalizations must be positive and expressed in a consistent
    currency and valuation date. This function does not download market data.
    """
    if not isinstance(market_caps, pd.Series):
        raise TypeError("market_caps must be a pandas Series.")

    if market_caps.empty or market_caps.index.has_duplicates:
        raise ValueError("Market capitalizations must have unique asset labels.")

    caps = market_caps.astype(float)

    if not np.isfinite(caps.to_numpy()).all() or (caps <= 0).any():
        raise ValueError("Market capitalizations must be finite and positive.")

    if assets is not None:
        missing = assets.difference(caps.index)
        if len(missing):
            raise ValueError(f"Missing market capitalizations: {list(missing)}")
        caps = caps.reindex(assets)

    weights = caps / caps.sum()
    weights.name = "Market Cap Weighted"

    return _validate_weights(weights)


def minimum_volatility_strategy(
    returns: pd.DataFrame,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Construct a minimum-volatility portfolio."""
    clean = _validate_returns(returns)

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=max_weight,
        allow_short_selling=False,
        fully_invested=True,
    )

    return minimum_volatility_weights(
        expected_returns=clean.mean() * TRADING_DAYS,
        covariance=clean.cov() * TRADING_DAYS,
        constraints=constraints,
    )


def maximum_sharpe_strategy(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Construct a maximum-Sharpe portfolio."""
    clean = _validate_returns(returns)

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=max_weight,
        allow_short_selling=False,
        fully_invested=True,
    )

    return maximum_sharpe_weights(
        expected_returns=clean.mean() * TRADING_DAYS,
        covariance=clean.cov() * TRADING_DAYS,
        risk_free_rate=risk_free_rate,
        constraints=constraints,
    )


def hrp_strategy(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Construct a hierarchical risk parity portfolio."""
    clean = _validate_returns(returns)

    return hierarchical_risk_parity_weights(
        returns=clean,
        linkage_method=linkage_method,
    )


def risk_parity_strategy(
    returns: pd.DataFrame,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Construct an equal-risk-contribution portfolio."""
    clean = _validate_returns(returns)

    return equal_risk_contribution_weights(
        returns=clean,
        max_weight=max_weight,
    )


def black_litterman_strategy(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    view_matrix: pd.DataFrame,
    view_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    confidence: float = 0.60,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Construct a Black-Litterman maximum-Sharpe portfolio."""
    clean = _validate_returns(returns)
    covariance = clean.cov() * TRADING_DAYS

    (
        posterior_excess_returns,
        posterior_covariance,
        _,
        _,
    ) = black_litterman_posterior(
        covariance=covariance,
        market_weights=market_weights.reindex(clean.columns),
        view_matrix=view_matrix.reindex(columns=clean.columns),
        view_returns=view_returns,
        risk_aversion=risk_aversion,
        tau=tau,
        confidence=confidence,
    )

    posterior_total_returns = posterior_excess_returns + risk_free_rate

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=max_weight,
        allow_short_selling=False,
        fully_invested=True,
    )

    return maximum_sharpe_weights(
        expected_returns=posterior_total_returns,
        covariance=posterior_covariance,
        risk_free_rate=risk_free_rate,
        constraints=constraints,
    )


def buy_and_hold_weights(
    initial_weights: pd.Series,
) -> pd.Series:
    """Validate and return initial buy-and-hold portfolio weights.

    The backtesting engine must allow weights to drift naturally between
    trades; this function does not imply periodic rebalancing.
    """
    weights = _validate_weights(initial_weights.copy())
    weights.name = "Buy and Hold"

    return weights
