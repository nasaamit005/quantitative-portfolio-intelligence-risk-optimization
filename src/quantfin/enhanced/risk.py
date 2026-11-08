"""Risk analytics: VaR/CVaR (historical, parametric, Cornish-Fisher, Monte Carlo) and risk decomposition.

All VaR/CVaR numbers are positive loss fractions over `horizon` periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def horizon_returns(r: pd.Series, horizon: int) -> pd.Series:
    """Overlapping compounded returns over `horizon` periods."""
    if horizon == 1:
        return r
    return np.expm1(np.log1p(r).rolling(horizon).sum()).dropna()


def var_cvar_historical(r: pd.Series, alpha: float, horizon: int = 1) -> tuple[float, float]:
    h = horizon_returns(r, horizon)
    q = float(np.quantile(h, 1 - alpha))
    tail = h[h <= q]
    return -q, float(-tail.mean())


def var_cvar_parametric(r: pd.Series, alpha: float, horizon: int = 1) -> tuple[float, float]:
    """Gaussian VaR/CVaR with sqrt-time scaling."""
    mu, sd = r.mean() * horizon, r.std(ddof=1) * np.sqrt(horizon)
    z = stats.norm.ppf(1 - alpha)
    return float(-(mu + z * sd)), float(-(mu - sd * stats.norm.pdf(z) / (1 - alpha)))


def var_cornish_fisher(r: pd.Series, alpha: float, horizon: int = 1) -> float:
    """Cornish-Fisher VaR: Gaussian quantile adjusted for skewness and excess kurtosis."""
    mu, sd = r.mean() * horizon, r.std(ddof=1) * np.sqrt(horizon)
    s, k = stats.skew(r, bias=False), stats.kurtosis(r, bias=False)
    z = stats.norm.ppf(1 - alpha)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(mu + zcf * sd))


def simulate_portfolio(
    weights: np.ndarray,
    mu_daily: np.ndarray,
    cov_daily: np.ndarray,
    horizon: int,
    sims: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo horizon returns of a fixed-weight portfolio under multivariate normality."""
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu_daily * horizon, cov_daily * horizon, size=sims)
    return draws @ weights


def var_cvar_monte_carlo(sim: np.ndarray, alpha: float) -> tuple[float, float]:
    q = float(np.quantile(sim, 1 - alpha))
    return -q, float(-sim[sim <= q].mean())


def risk_contributions(weights: pd.Series, cov_ann: pd.DataFrame) -> pd.DataFrame:
    """Marginal and component contributions to portfolio volatility (sum of components = vol)."""
    w = weights.reindex(cov_ann.index).to_numpy()
    sigma = cov_ann.to_numpy()
    vol = float(np.sqrt(w @ sigma @ w))
    marginal = sigma @ w / vol
    comp = w * marginal
    return pd.DataFrame(
        {
            "Weight": w,
            "Marginal contribution": marginal,
            "Component contribution": comp,
            "% of portfolio risk": comp / vol,
        },
        index=cov_ann.index,
    )


def tail_stats(r: pd.Series, alpha: float) -> dict[str, float]:
    q = np.quantile(r, 1 - alpha)
    below = r[r < 0]
    return {
        "Downside deviation (ann., vs 0)": float(np.sqrt(np.mean(np.minimum(r, 0) ** 2) * 252)),
        "Worst day": float(r.min()),
        f"Days beyond {int(alpha * 100)}% VaR": float((r <= q).sum()),
        "Average loss on down days": float(below.mean()) if len(below) else float("nan"),
    }
