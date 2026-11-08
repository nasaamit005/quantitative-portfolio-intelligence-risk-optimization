"""Performance metrics. All inputs are simple periodic returns; outputs are fractions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def equity_curve(r: pd.Series | pd.DataFrame, start: float = 1.0) -> pd.Series | pd.DataFrame:
    """Growth of `start` invested at the beginning."""
    return start * (1.0 + r).cumprod()


def drawdown_series(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Underwater curve: equity / running peak - 1 (peak includes the starting value 1)."""
    eq = equity_curve(r)
    peak = np.maximum(eq.cummax(), 1.0)
    return eq / peak - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def cagr(r: pd.Series, ppy: int) -> float:
    n = len(r)
    if n == 0:
        return float("nan")
    return float(equity_curve(r).iloc[-1] ** (ppy / n) - 1.0)


def annual_vol(r: pd.Series, ppy: int) -> float:
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(r: pd.Series, rf: float, ppy: int) -> float:
    ex = r - rf / ppy
    sd = ex.std(ddof=1)
    return float(ex.mean() * ppy / (sd * np.sqrt(ppy))) if sd > 0 else float("nan")


def sortino(r: pd.Series, rf: float, ppy: int) -> float:
    ex = r - rf / ppy
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(ex.mean() * ppy / (dd * np.sqrt(ppy))) if dd > 0 else float("nan")


def beta_alpha(r: pd.Series, b: pd.Series, rf: float, ppy: int) -> tuple[float, float, float]:
    """OLS of excess strategy returns on excess benchmark returns -> (beta, annualised alpha, R^2)."""
    ex_r = r - rf / ppy
    ex_b = b - rf / ppy
    var_b = ex_b.var(ddof=1)
    if var_b <= 0:
        return float("nan"), float("nan"), float("nan")
    beta = float(ex_r.cov(ex_b) / var_b)
    alpha_d = float(ex_r.mean() - beta * ex_b.mean())
    corr = float(ex_r.corr(ex_b))
    return beta, alpha_d * ppy, corr**2


def perf_summary(r: pd.Series, bench: pd.Series | None, rf: float, ppy: int) -> dict[str, float]:
    """Full metric set for one return series (optionally relative to a benchmark)."""
    n = len(r)
    mdd = max_drawdown(r)
    c = cagr(r, ppy)
    out: dict[str, float] = {
        "Total return": float(equity_curve(r).iloc[-1] - 1.0),
        "CAGR": c,
        "Volatility": annual_vol(r, ppy),
        "Sharpe": sharpe(r, rf, ppy),
        "Sortino": sortino(r, rf, ppy),
        "Max drawdown": mdd,
        "Calmar": c / abs(mdd) if mdd < 0 else float("nan"),
        "Skewness": float(stats.skew(r, bias=False)) if n > 3 else float("nan"),
        "Excess kurtosis": float(stats.kurtosis(r, bias=False)) if n > 4 else float("nan"),
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "Hit rate": float((r > 0).mean()),
        "Observations": float(n),
    }
    if bench is not None:
        b = bench.loc[r.index]
        beta, alpha, r2 = beta_alpha(r, b, rf, ppy)
        active = r - b
        te = float(active.std(ddof=1) * np.sqrt(ppy))
        up, dn = b > 0, b < 0
        out.update(
            {
                "Beta": beta,
                "Alpha (ann.)": alpha,
                "R-squared": r2,
                "Tracking error": te,
                "Information ratio": float(active.mean() * ppy / te) if te > 0 else float("nan"),
                "Excess CAGR vs benchmark": c - cagr(b, ppy),
                "Up capture": float(r[up].mean() / b[up].mean()) if up.any() else float("nan"),
                "Down capture": float(r[dn].mean() / b[dn].mean()) if dn.any() else float("nan"),
            }
        )
    return out


def drawdown_episodes(r: pd.Series, top: int = 5) -> pd.DataFrame:
    """Largest drawdown episodes with peak, trough, recovery dates and durations."""
    dd = drawdown_series(r)
    idx = r.index
    episodes: list[tuple[int, int | None]] = []
    start: int | None = None
    for i, below in enumerate((dd < 0).to_numpy()):
        if below and start is None:
            start = max(i - 1, 0)
        elif not below and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, None))
    rows = []
    for s, e in episodes:
        seg = dd.iloc[s : (e if e is not None else len(dd))]
        trough_pos = s + int(np.argmin(seg.to_numpy()))
        rows.append(
            {
                "Peak date": idx[s],
                "Trough date": idx[trough_pos],
                "Recovery date": idx[e] if e is not None else pd.NaT,
                "Depth": float(seg.min()),
                "Days to trough": int(trough_pos - s),
                "Days to recover": int(e - trough_pos) if e is not None else np.nan,
                "Total days": int((e if e is not None else len(dd) - 1) - s),
                "Recovered": e is not None,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("Depth").head(top).reset_index(drop=True) if len(df) else df


def rolling_metrics(
    r: pd.Series, bench: pd.Series, window: int, rf: float, ppy: int
) -> pd.DataFrame:
    """Rolling annualised return, volatility, Sharpe and beta."""
    ex = r - rf / ppy
    roll = r.rolling(window)
    vol = roll.std(ddof=1) * np.sqrt(ppy)
    ret = np.exp(np.log1p(r).rolling(window).sum() * ppy / window) - 1.0
    sh = ex.rolling(window).mean() * ppy / vol
    exb = bench - rf / ppy
    beta = ex.rolling(window).cov(exb) / exb.rolling(window).var(ddof=1)
    return pd.DataFrame({"Return": ret, "Volatility": vol, "Sharpe": sh, "Beta": beta})


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Year x month compounded returns, plus a calendar-year column."""
    g = r.groupby([r.index.year, r.index.month]).apply(lambda x: float((1 + x).prod() - 1))
    tbl = g.unstack()
    tbl.columns = [pd.Timestamp(2000, int(m), 1).strftime("%b") for m in tbl.columns]
    tbl["Year"] = r.groupby(r.index.year).apply(lambda x: float((1 + x).prod() - 1))
    tbl.index.name = "Calendar year"
    return tbl


def annual_returns(r: pd.Series) -> pd.Series:
    return r.groupby(r.index.year).apply(lambda x: float((1 + x).prod() - 1))


def probabilistic_sharpe(r: pd.Series, benchmark_sr_ann: float, ppy: int) -> float:
    """Bailey & Lopez de Prado PSR: P(true Sharpe > benchmark) given skew/kurtosis and sample length."""
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    sk = stats.skew(r, bias=False)
    ku = stats.kurtosis(r, bias=False, fisher=False)
    denom = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - benchmark_sr_ann / np.sqrt(ppy)) * np.sqrt(n - 1) / denom))


def stat_diagnostics(r: pd.Series, rf: float, ppy: int) -> dict[str, float]:
    """Distribution and significance diagnostics for one return series."""
    n = len(r)
    ex = r - rf / ppy
    t_stat, p_val = stats.ttest_1samp(ex, 0.0)
    jb_stat, jb_p = stats.jarque_bera(r)
    lags = min(10, n // 5)
    ac = [r.autocorr(k) for k in range(1, lags + 1)]
    q = n * (n + 2) * sum(a**2 / (n - k) for k, a in enumerate(ac, start=1))
    lb_p = float(1 - stats.chi2.cdf(q, lags))
    sr_ann = sharpe(r, rf, ppy)
    se = np.sqrt((1 + 0.5 * sr_ann**2) / (n / ppy)) if n > 0 else float("nan")
    return {
        "Mean excess return t-stat": float(t_stat),
        "Mean excess return p-value": float(p_val),
        "Jarque-Bera stat": float(jb_stat),
        "Jarque-Bera p-value": float(jb_p),
        "Ljung-Box(10) p-value": lb_p,
        "Lag-1 autocorrelation": float(ac[0]),
        "Sharpe 95% CI low": float(sr_ann - 1.96 * se),
        "Sharpe 95% CI high": float(sr_ann + 1.96 * se),
        "Prob. Sharpe > 0 (PSR)": probabilistic_sharpe(r, 0.0, ppy),
    }
