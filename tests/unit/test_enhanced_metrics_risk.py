"""Unit tests for the reporting layer's metric and risk definitions (synthetic inputs with known answers)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantfin.enhanced import metrics as m
from quantfin.enhanced import risk as rk


def test_constant_return_cagr_and_no_drawdown():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    assert m.cagr(r, 252) == pytest.approx(1.001**252 - 1)
    assert m.max_drawdown(r) == 0


def test_drawdown_known_path():
    r = pd.Series([0.10, -0.20, 0.10, 0.10, 0.10], index=pd.bdate_range("2021-01-01", periods=5))
    eq = (1 + r).cumprod()
    assert m.max_drawdown(r) == pytest.approx(eq.iloc[1] / eq.iloc[0] - 1)
    ep = m.drawdown_episodes(r, 3)
    assert ep.iloc[0]["Depth"] == pytest.approx(-0.20)
    assert ep.iloc[0]["Days to trough"] == 1


def test_sharpe_matches_definition():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 800))
    ex = r - 0.02 / 252
    assert m.sharpe(r, 0.02, 252) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))


def test_beta_alpha_recovers_known_values():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0.0004, 0.01, 3000))
    r = 0.0002 + 1.5 * b + pd.Series(rng.normal(0, 0.002, 3000))
    beta, alpha, r2 = m.beta_alpha(r, b, 0.0, 252)
    assert beta == pytest.approx(1.5, abs=0.02)
    assert alpha == pytest.approx(0.0002 * 252, abs=0.02)
    assert r2 > 0.9


def test_var_hist_matches_quantile_and_cvar_exceeds_var():
    r = pd.Series(np.random.default_rng(2).standard_t(4, 5000) * 0.01)
    v, c = rk.var_cvar_historical(r, 0.95)
    assert v == pytest.approx(-np.quantile(r, 0.05))
    assert c >= v


def test_parametric_var_on_gaussian_sample():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 200000))
    v, c = rk.var_cvar_parametric(r, 0.95)
    assert v == pytest.approx(0.01 * stats.norm.ppf(0.95), rel=0.02)
    assert c > v


def test_monte_carlo_var_close_to_parametric():
    cov = np.array([[0.0001, 0.00004], [0.00004, 0.0002]])
    w = np.array([0.6, 0.4])
    sim = rk.simulate_portfolio(w, np.zeros(2), cov, 1, 200000, 5)
    v, _ = rk.var_cvar_monte_carlo(sim, 0.95)
    assert v == pytest.approx(stats.norm.ppf(0.95) * np.sqrt(w @ cov @ w), rel=0.02)


def test_risk_contributions_sum_to_portfolio_volatility():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(500, 4))
    cov = pd.DataFrame(np.cov(a.T) * 252, index=list("ABCD"), columns=list("ABCD"))
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    rc = rk.risk_contributions(w, cov)
    vol = np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    assert rc["Component contribution"].sum() == pytest.approx(vol)
    assert rc["% of portfolio risk"].sum() == pytest.approx(1.0)
