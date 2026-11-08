"""Window analytics: one code path for the full sample and for dashboard date filters."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import metrics as m
from . import risk as rk
from .config import Config

BENCH = "Benchmark"


@dataclass
class WindowAnalytics:
    start: pd.Timestamp
    end: pd.Timestamp
    returns: pd.DataFrame
    perf: pd.DataFrame  # rows = metrics, columns = series
    equity: pd.DataFrame
    drawdown: pd.DataFrame
    rolling: dict[str, pd.DataFrame]  # metric -> columns per strategy
    monthly: dict[str, pd.DataFrame]
    annual: pd.DataFrame
    excess_cum: pd.DataFrame
    risk_table: pd.DataFrame
    episodes: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def analyze_window(sr: pd.DataFrame, cfg: Config) -> WindowAnalytics:
    """Compute every return-based analytic for the given strategy-return table (must include 'Benchmark')."""
    if BENCH not in sr.columns:
        raise ValueError("strategy return table must contain a 'Benchmark' column")
    if len(sr) < 30:
        raise ValueError("Selected window has fewer than 30 observations")
    ppy, rf = cfg.periods_per_year, cfg.risk_free_rate
    bench = sr[BENCH]
    perf = pd.DataFrame(
        {c: m.perf_summary(sr[c], None if c == BENCH else bench, rf, ppy) for c in sr.columns}
    )
    win = min(cfg.rolling_window, max(len(sr) // 3, 20))
    roll = {
        k: pd.DataFrame({c: m.rolling_metrics(sr[c], bench, win, rf, ppy)[k] for c in sr.columns})
        for k in ("Return", "Volatility", "Sharpe", "Beta")
    }
    risk_rows = {}
    for c in sr.columns:
        hv, hc = rk.var_cvar_historical(sr[c], cfg.confidence, cfg.var_horizon)
        pv, pc = rk.var_cvar_parametric(sr[c], cfg.confidence, cfg.var_horizon)
        risk_rows[c] = {
            "Historical VaR": hv,
            "Historical CVaR": hc,
            "Parametric VaR": pv,
            "Parametric CVaR": pc,
            "Cornish-Fisher VaR": rk.var_cornish_fisher(sr[c], cfg.confidence, cfg.var_horizon),
            **rk.tail_stats(sr[c], cfg.confidence),
        }
    excess = (1 + sr.drop(columns=BENCH)).cumprod().div((1 + bench).cumprod(), axis=0) - 1
    return WindowAnalytics(
        start=sr.index[0],
        end=sr.index[-1],
        returns=sr,
        perf=perf,
        equity=m.equity_curve(sr),
        drawdown=m.drawdown_series(sr),
        rolling=roll,
        monthly={c: m.monthly_table(sr[c]) for c in sr.columns},
        annual=pd.DataFrame({c: m.annual_returns(sr[c]) for c in sr.columns}),
        excess_cum=excess,
        risk_table=pd.DataFrame(risk_rows).T,
        episodes={c: m.drawdown_episodes(sr[c], cfg.top_drawdowns) for c in sr.columns},
        diagnostics=pd.DataFrame({c: m.stat_diagnostics(sr[c], rf, ppy) for c in sr.columns}).T,
    )
