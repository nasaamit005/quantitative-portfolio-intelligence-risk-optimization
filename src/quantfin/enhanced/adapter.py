"""Build the reporting `Results` from this project's own saved outputs.

Nothing is re-optimised or re-backtested: strategy returns, weights, turnover, costs, frontier, stress and
robustness tables are read from ``reports/`` and ``data/`` exactly as the existing scripts wrote them. The only
quantities computed here are derived from those files (risk contributions, VaR variants, diagnostics, regimes
aggregated by trend and volatility).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics as m
from .analytics import BENCH, analyze_window
from .config import Config
from .insights import generate_findings, recommendations
from .results import Results
from .risk import risk_contributions, simulate_portfolio, var_cvar_monte_carlo

logger = logging.getLogger(__name__)

STRATEGY_ORDER = ("equal_weight", "minimum_volatility", "maximum_sharpe", "risk_parity", "hrp")
REBAL_LABEL = {
    "W": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "A": "annual",
    "Y": "annual",
    "YE": "annual",
}
REBAL_ORDER = ["weekly", "monthly", "quarterly", "annual"]


class InputError(RuntimeError):
    """Raised when a required project output is missing or inconsistent."""


def _read(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        raise InputError(
            f"Missing required file: {path}. Run the corresponding pipeline script first."
        )
    return pd.read_csv(path, **kw)


def cap_weights(w: np.ndarray, hi: float) -> np.ndarray:
    """Project a weight vector onto 0 <= w <= hi, redistributing the excess pro rata."""
    w = np.asarray(w, float) / np.sum(w)
    for _ in range(200):
        over = w > hi + 1e-12
        if not over.any():
            return w
        excess = float((w[over] - hi).sum())
        w[over] = hi
        free = (~over) & (w < hi - 1e-12)
        w[free] += excess * w[free] / w[free].sum()
    raise InputError("could not satisfy the weight cap")


def walk_forward_yearly(sr: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Calendar-year metrics of the (already out-of-sample) strategy returns."""
    rows = []
    for year, g in sr.groupby(sr.index.year):
        if len(g) < 40:
            continue
        for col in g.columns:
            rows.append(
                {
                    "Year": int(year),
                    "Strategy": col,
                    "Return": float((1 + g[col]).prod() - 1),
                    "Volatility": m.annual_vol(g[col], cfg.periods_per_year),
                    "Sharpe": m.sharpe(g[col], cfg.risk_free_rate, cfg.periods_per_year),
                    "Max drawdown": m.max_drawdown(g[col]),
                    "Trading days": len(g),
                }
            )
    return pd.DataFrame(rows)


def regime_performance(sr: pd.DataFrame, labels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Performance aggregated by each label dimension (trend, volatility)."""
    rows = []
    for kind in labels.columns:
        lab = labels[kind].reindex(sr.index)
        for regime in sorted(lab.dropna().unique()):
            mask = lab == regime
            for col in sr.columns:
                r = sr.loc[mask, col]
                if len(r) < 20:
                    continue
                rows.append(
                    {
                        "Regime type": kind,
                        "Regime": regime,
                        "Strategy": col,
                        "Days": len(r),
                        "% of days": float(mask.mean()),
                        "Annualised return": float(
                            (1 + r).prod() ** (cfg.periods_per_year / len(r)) - 1
                        ),
                        "Volatility": m.annual_vol(r, cfg.periods_per_year),
                        "Sharpe": m.sharpe(r, cfg.risk_free_rate, cfg.periods_per_year),
                        "Hit rate": float((r > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _sensitivity(root: Path, years_of: float) -> pd.DataFrame:
    s = _read(root / "reports/robustness_analysis/sensitivity_experiment_results.csv")
    s = s[s["status"] == "success"]
    spec = {
        "lookback": ("Lookback (days)", "lookback_days"),
        "transaction_cost": ("Transaction cost (bps)", "transaction_cost_bps"),
        "rebalance_frequency": ("Rebalancing frequency", "rebalance_frequency"),
        "maximum_weight": ("Max weight", "max_weight"),
    }
    base = s[s["experiment"] == "baseline"]
    rows = []
    for exp, (label, col) in spec.items():
        grp = pd.concat([s[s["experiment"] == exp], base]).drop_duplicates(subset=["strategy", col])
        for r in grp.itertuples(index=False):
            val = getattr(r, col)
            val = REBAL_LABEL.get(val, val) if col == "rebalance_frequency" else val
            rows.append(
                {
                    "Parameter": label,
                    "Value": str(val),
                    "Strategy": r.strategy,
                    "CAGR": r.annualized_return,
                    "Volatility": r.annualized_volatility,
                    "Sharpe": r.sharpe_ratio,
                    "Max drawdown": r.max_drawdown,
                    "Annual turnover": r.total_turnover / (r.observations / 252),
                    "_sort": float(getattr(r, col))
                    if col != "rebalance_frequency"
                    else REBAL_ORDER.index(REBAL_LABEL[r.rebalance_frequency]),
                }
            )
    d = pd.DataFrame(rows)
    d = (
        d.sort_values(["Parameter", "Strategy", "_sort"], kind="stable")
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return d


def build_results(root: str | Path = ".", mc_sims: int = 20000, seed: int = 42) -> Results:
    """Assemble `Results` from reports/ and data/ under the project root."""
    root = Path(root)
    rep = root / "reports"
    settings = yaml.safe_load((root / "config/settings.yaml").read_text(encoding="utf-8-sig"))

    daily = _read(rep / "walk_forward_comparison/strategy_daily_results.csv", parse_dates=["date"])
    names = [s for s in STRATEGY_ORDER if s in set(daily["strategy"])] + sorted(
        set(daily["strategy"]) - set(STRATEGY_ORDER)
    )
    net = daily.pivot(index="date", columns="strategy", values="net_daily_return")[names]
    gross = daily.pivot(index="date", columns="strategy", values="gross_daily_return")[names]
    bench_all = _read(
        root / "data/processed/us_benchmarks_simple_returns.csv", index_col=0, parse_dates=True
    )["SPY"]
    bench = bench_all.reindex(net.index)
    if bench.isna().any() or net.isna().any().any():
        raise InputError("Strategy and benchmark dates do not align (NaN after reindexing)")
    sr = net.assign(**{BENCH: bench})

    cost_frac = (gross - net).clip(lower=0)
    paid = daily[daily["turnover"] > 0]
    implied_bps = (
        float(
            ((paid["gross_daily_return"] - paid["net_daily_return"]) / paid["turnover"]).median()
            * 1e4
        )
        if len(paid)
        else 0.0
    )
    hist_windows = _read(rep / "stress_testing/historical_stress_results.csv")
    stress_windows = {
        r.historical_period: (r.start_date, r.end_date)
        for r in hist_windows.drop_duplicates("historical_period").itertuples()
    }
    p = settings["portfolio"]
    cfg = Config(
        benchmark="SPY",
        risk_free_rate=float(p["risk_free_rate"]),
        periods_per_year=int(p["trading_days_per_year"]),
        lookback=252,
        rebalance="monthly",
        min_weight=float(p["min_weight"]),
        max_weight=float(p["max_weight"]),
        cost_bps=round(implied_bps, 2),
        confidence=float(settings["risk"]["confidence_level"]),
        mc_sims=mc_sims,
        seed=seed,
        strategies=tuple(names),
        stress_windows=stress_windows,
        output_dir=str(rep / "enhanced"),
    )
    cfg.validate()

    # weights and turnover at actual trade dates
    wcols = [c for c in daily.columns if c.startswith("weight_")]
    weights, turnover = {}, {}
    for s in names:
        d = daily[(daily["strategy"] == s) & (daily["turnover"] > 0)].set_index("date")
        weights[s] = d[wcols].rename(columns=lambda c: c.removeprefix("weight_"))
        turnover[s] = d["turnover"].rename(s)
    tickers = list(weights[names[0]].columns)

    # market inputs as estimated by the project
    exp_ret = (
        _read(rep / "tables/estimated_annual_returns.csv", index_col=0).iloc[:, 0].reindex(tickers)
    )
    lw_cov = _read(rep / "tables/us_equities_ledoit_wolf_covariance_matrix.csv", index_col=0).loc[
        tickers, tickers
    ]
    corr = _read(rep / "tables/us_equities_pearson_correlation_matrix.csv", index_col=0).loc[
        tickers, tickers
    ]
    cov = _read(rep / "tables/us_equities_annualized_covariance_matrix.csv", index_col=0).loc[
        tickers, tickers
    ]
    eq_ret = _read(
        root / "data/processed/us_equities_simple_returns.csv", index_col=0, parse_dates=True
    )[tickers]
    asset_stats = pd.DataFrame(
        {
            "Annual return": exp_ret,
            "Volatility": pd.Series(np.sqrt(np.diag(lw_cov.to_numpy())), index=tickers),
        }
    )
    asset_stats["Sharpe"] = (asset_stats["Annual return"] - cfg.risk_free_rate) / asset_stats[
        "Volatility"
    ]

    # static optimisation + frontier (project's own)
    ow = _read(rep / "tables/optimized_portfolio_weights.csv", index_col=0)
    ow.index.name = None
    op = _read(rep / "tables/optimized_portfolio_performance.csv")
    op["Strategy"] = op["portfolio"].str.lower().str.replace(" ", "_")
    op = op.set_index("Strategy")
    static_stats = pd.DataFrame(
        {
            "Expected return": op["expected_annual_return"],
            "Volatility": op["annual_volatility"],
            "Sharpe": op["sharpe_ratio"],
            "Effective N": {k: float(1 / np.sum(ow[k].to_numpy() ** 2)) for k in op.index},
            "Max weight": {k: float(ow[k].max()) for k in op.index},
        }
    )
    ef = _read(rep / "tables/efficient_frontier.csv")
    frontier = pd.DataFrame(
        {
            "Return": ef["expected_annual_return"],
            "Volatility": ef["annual_volatility"],
            "Sharpe": ef["sharpe_ratio"],
        }
    ).join(
        ef[[c for c in ef.columns if c.startswith("weight_")]].rename(
            columns=lambda c: c.removeprefix("weight_")
        )
    )
    rng = np.random.default_rng(seed)
    mu_v, cov_v = exp_ret.to_numpy(), lw_cov.to_numpy()
    rp = []
    for _ in range(3000):
        w = cap_weights(rng.dirichlet(np.full(len(tickers), 0.6)), cfg.max_weight)
        r_, v_ = float(w @ mu_v), float(np.sqrt(w @ cov_v @ w))
        rp.append({"Return": r_, "Volatility": v_, "Sharpe": (r_ - cfg.risk_free_rate) / v_})
    random_pf = pd.DataFrame(rp)

    # risk contributions and Monte Carlo at the latest target weights
    latest = pd.DataFrame({s: weights[s].iloc[-1] for s in names})
    contrib = {s: risk_contributions(latest[s], lw_cov) for s in names}
    mc_rows, mc_sample = {}, {}
    for i, s in enumerate(names):
        sim = simulate_portfolio(
            latest[s].to_numpy(),
            (mu_v / cfg.periods_per_year),
            (cov_v / cfg.periods_per_year),
            cfg.var_horizon,
            mc_sims,
            seed + i,
        )
        v, c = var_cvar_monte_carlo(sim, cfg.confidence)
        mc_rows[s] = {"Monte Carlo VaR": v, "Monte Carlo CVaR": c}
        mc_sample[s] = sim
    mc = pd.DataFrame(mc_rows).T

    years = len(net) / cfg.periods_per_year
    full = analyze_window(sr, cfg)
    cost = pd.DataFrame(
        {
            s: {
                "Gross CAGR": float((1 + gross[s]).prod() ** (1 / years) - 1),
                "Net CAGR": float(full.perf.loc["CAGR", s]),
                "Annual cost drag": float(cost_frac[s].sum() / years),
                "Annual turnover": float(turnover[s].sum() / years),
                "Rebalances": int(len(turnover[s])),
                "Total costs paid": float(cost_frac[s].sum()),
            }
            for s in names
        }
    ).T
    cost["CAGR lost to costs"] = cost["Gross CAGR"] - cost["Net CAGR"]

    # regimes: project's own classification, aggregated by trend and volatility
    cls = _read(
        rep / "robustness_analysis/market_regime_classification.csv",
        index_col="date",
        parse_dates=True,
    )
    reg = cls["market_regime"].astype(str)
    labels = pd.DataFrame(
        {
            "Trend": np.where(
                reg.str.startswith("bear"),
                "Bear (below MA)",
                np.where(reg.str.startswith("bull"), "Bull (above MA)", None),
            ),
            "Volatility": np.where(
                reg.str.contains("high_vol"),
                "High volatility",
                np.where(reg.str.contains("normal_vol"), "Normal volatility", None),
            ),
        },
        index=cls.index,
    )
    regimes = regime_performance(sr, labels, cfg)
    theirs = _read(rep / "robustness_analysis/market_regime_performance.csv")
    theirs = theirs[theirs["sample_status"] == "sufficient_for_reporting"]
    combined = pd.DataFrame(
        {
            "Regime type": "Trend x volatility (project classification)",
            "Regime": theirs["market_regime"],
            "Strategy": theirs["strategy"],
            "Days": theirs["observations"],
            "% of days": theirs["share_of_classified_observations"],
            "Annualised return": theirs["conditional_annualized_return"],
            "Volatility": theirs["annualized_volatility"],
            "Sharpe": theirs["sharpe_ratio"],
            "Hit rate": theirs["hit_rate"],
        }
    )
    regimes = pd.concat([regimes, combined], ignore_index=True)

    # stress
    wide = hist_windows.pivot(
        index="historical_period", columns="strategy", values="cumulative_return"
    )
    order = list(hist_windows.drop_duplicates("historical_period")["historical_period"])
    stress_hist = pd.DataFrame({"Scenario": order, "Type": "Named crisis window"})
    stress_hist["Start"] = [pd.Timestamp(stress_windows[k][0]) for k in order]
    stress_hist["End"] = [pd.Timestamp(stress_windows[k][1]) for k in order]
    stress_hist["Covered by backtest"] = [
        (a >= net.index.min()) and (b <= net.index.max())
        for a, b in zip(stress_hist["Start"], stress_hist["End"], strict=True)
    ]
    for s in names:
        stress_hist[s] = wide.reindex(order)[s].to_numpy()
    stress_hist[BENCH] = [
        float((1 + bench.loc[a:b]).prod() - 1)
        for a, b in zip(stress_hist["Start"], stress_hist["End"], strict=True)
    ]
    hyp = _read(rep / "stress_testing/hypothetical_scenario_results.csv")
    stress_hypo = pd.DataFrame(
        {
            "Scenario": hyp["scenario"],
            "Strategy": hyp["strategy"],
            "Impact": hyp["portfolio_return"],
            "Method": "Hypothetical: " + hyp["scenario_type"],
        }
    )

    quality = _read(rep / "tables/data_quality_summary.csv")
    desc = _read(rep / "tables/us_equities_statistical_summary.csv", index_col=0)
    meta = {
        "start": str(net.index[0].date()),
        "end": str(net.index[-1].date()),
        "n_obs": int(len(net)),
        "assets": tickers,
        "synthetic": False,
        "data_source": "Project outputs under reports/ and data/processed (yfinance market data, quantfin walk-forward engine)",
        "benchmark_label": "SPY",
        "data_start": str(eq_ret.index[0].date()),
        "rf": cfg.risk_free_rate,
        "universe_note": (
            "The universe is 12 large US stocks selected with hindsight (e.g. NVDA, META, AAPL), so all strategies and the benchmark "
            "comparison carry survivorship and selection bias; absolute returns overstate what an investor could have known ex ante."
        ),
        "data_notes": [
            "Strategy returns, weights, turnover and costs are read unchanged from reports/walk_forward_comparison.",
            f"Implied round-trip cost per unit turnover in the backtest: {cfg.cost_bps:g} bps (transaction cost plus slippage).",
            "Sharpe and Sortino here use rf/252 per day; the project's own performance_summary compounds rf daily, giving values about 0.4% higher.",
            "Universe is a fixed list of 12 large US stocks chosen with hindsight: results carry survivorship / selection bias.",
            "Regime labels come from the project's own classification (universe proxy), aggregated here by trend and volatility.",
        ],
    }
    res = Results(
        cfg=cfg,
        meta=meta,
        assets=eq_ret.reindex(net.index),
        bench_full=bench_all,
        strategy_returns=sr,
        gross_returns=gross,
        weights=weights,
        turnover=turnover,
        backtest_diagnostics=[],
        full=full,
        static_weights=ow,
        static_stats=static_stats,
        frontier=frontier,
        random_pf=random_pf,
        asset_stats=asset_stats,
        corr=corr,
        cov=cov,
        desc=desc,
        quality=quality,
        sens=_sensitivity(root, years),
        wf_year=walk_forward_yearly(sr, cfg),
        wf_select=pd.DataFrame(
            columns=[
                "Strategy",
                "Year",
                "Selected return (OOS)",
                "Average of all lookbacks",
                "Selection beat average",
            ]
        ),
        regimes=regimes,
        regime_labels=labels,
        stress_hist=stress_hist,
        stress_hypo=stress_hypo,
        cost_table=cost,
        risk_contrib=contrib,
        mc_table=mc,
        mc_sample=mc_sample,
        findings=pd.DataFrame(),
        recs=pd.DataFrame(),
    )
    res.findings = generate_findings(res)
    res.recs = recommendations(res)
    return res
