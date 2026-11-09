"""Rule-based narrative insights generated from computed results (never hand-written numbers)."""

from __future__ import annotations

import re

import pandas as pd

HIST = "Historical observation"
MODEL = "Model estimate"
ASSUME = "Forward-looking assumption"


def _pct(x: float) -> str:
    return f"{x:.1%}"



def _prettify(df: pd.DataFrame, names: list[str], cols: list[str]) -> pd.DataFrame:
    """Replace snake_case strategy keys (e.g. maximum_sharpe) with readable labels (Maximum Sharpe)."""
    from .charts import pretty

    out = df.copy()
    for key in sorted(names, key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(key)}\b")
        for c in cols:
            out[c] = out[c].map(lambda t, pat=pat, key=key: pat.sub(pretty(key), t) if isinstance(t, str) else t)
    return out


def generate_findings(res) -> pd.DataFrame:  # noqa: ANN001 - Results (avoid circular import)
    cfg, wa = res.cfg, res.full
    perf = wa.perf
    strat = [c for c in perf.columns if c != "Benchmark"]
    rows: list[tuple[str, str, str, str]] = []

    def add(section: str, finding: str, kind: str, detail: str = "") -> None:
        rows.append((section, finding, kind, detail))

    best_sh = perf.loc["Sharpe", strat].idxmax()
    best_ret = perf.loc["CAGR", strat].idxmax()
    best_dd = perf.loc["Max drawdown", strat].idxmax()
    low_vol = perf.loc["Volatility", strat].idxmin()
    add(
        "Performance",
        f"Highest risk-adjusted return: {best_sh} (Sharpe {perf.loc['Sharpe', best_sh]:.2f}); "
        f"Benchmark Sharpe {perf.loc['Sharpe', 'Benchmark']:.2f}.",
        HIST,
        f"Backtest {wa.start.date()} to {wa.end.date()}, net of {cfg.cost_bps:g} bps costs, rf={cfg.risk_free_rate:.1%}.",
    )
    add(
        "Performance",
        f"Highest CAGR: {best_ret} ({_pct(perf.loc['CAGR', best_ret])}) vs benchmark {_pct(perf.loc['CAGR', 'Benchmark'])}.",
        HIST,
    )
    add(
        "Downside",
        f"Shallowest maximum drawdown: {best_dd} ({_pct(perf.loc['Max drawdown', best_dd])}); "
        f"benchmark {_pct(perf.loc['Max drawdown', 'Benchmark'])}. Lowest volatility: {low_vol} "
        f"({_pct(perf.loc['Volatility', low_vol])}).",
        HIST,
    )
    beat = [s for s in strat if perf.loc["Excess CAGR vs benchmark", s] > 0]
    add(
        "Benchmark",
        f"{len(beat)} of {len(strat)} strategies beat the benchmark on CAGR"
        + (f": {', '.join(beat)}." if beat else "."),
        HIST,
    )
    ir = perf.loc["Information ratio", strat]
    add(
        "Benchmark",
        f"Best information ratio: {ir.idxmax()} ({ir.max():.2f}); tracking error {_pct(perf.loc['Tracking error', ir.idxmax()])}.",
        HIST,
    )

    # concentration / risk contributions
    for s, rc in res.risk_contrib.items():
        top = rc["% of portfolio risk"].idxmax()
        add(
            "Risk contribution",
            f"{s}: largest risk contributor is {top} with {_pct(rc.loc[top, '% of portfolio risk'])} "
            f"of risk on {_pct(rc.loc[top, 'Weight'])} of capital.",
            MODEL,
            "Latest target weights, shrunk covariance.",
        )
    # diversification
    ss = res.static_stats
    add(
        "Diversification",
        "Effective number of holdings (1/sum w^2), full-sample optimisation: "
        + "; ".join(f"{i}={ss.loc[i, 'Effective N']:.1f}" for i in ss.index)
        + ".",
        MODEL,
        "In-sample and hypothetical; weights are not tradeable results.",
    )

    # cost
    ct = res.cost_table
    worst = ct["Annual cost drag"].idxmax()
    add(
        "Transaction costs",
        f"Cost drag is largest for {worst}: {ct.loc[worst, 'Annual cost drag']:.2%} per year at "
        f"{ct.loc[worst, 'Annual turnover']:.1f}x annual turnover.",
        HIST,
    )

    # regimes
    rp = res.regimes
    if len(rp):
        tr = rp[rp["Regime type"] == "Trend"]
        bear = tr[tr["Regime"].str.startswith("Bear")]
        if len(bear):
            b = (
                bear[bear["Strategy"].isin(strat)]
                .sort_values("Annualised return", ascending=False)
                .iloc[0]
            )
            add(
                "Regimes",
                f"Best strategy in bearish trend regimes: {b['Strategy']} "
                f"({_pct(b['Annualised return'])} annualised over {int(b['Days'])} days).",
                HIST,
                "Regime label uses only information up to the prior day.",
            )
        hv = rp[
            (rp["Regime type"] == "Volatility")
            & (rp["Regime"] == "High volatility")
            & rp["Strategy"].isin(strat)
        ]
        if len(hv):
            w = hv.sort_values("Sharpe").iloc[0]
            add(
                "Regimes",
                f"Weakest Sharpe in high-volatility regimes: {w['Strategy']} ({w['Sharpe']:.2f}).",
                HIST,
            )

    # robustness
    sens = res.sens
    if len(sens):
        spread = sens.groupby(["Parameter", "Strategy"])["Sharpe"].agg(lambda x: x.max() - x.min())
        p, s = spread.idxmax()
        add(
            "Robustness",
            f"Most parameter-sensitive result: {s} to '{p}' (Sharpe range {spread.max():.2f}). "
            f"Median Sharpe range across all cells: {spread.median():.2f}.",
            HIST,
            "Wide ranges signal estimation risk; prefer strategies whose ranking is stable.",
        )
    wf = res.wf_select
    if len(wf):
        share = wf["Selection beat average"].mean()
        add(
            "Walk-forward",
            f"Choosing the lookback with the best trailing Sharpe beat the average-lookback outcome in "
            f"{_pct(share)} of strategy-years (n={len(wf)}). Values near 50% imply no reliable benefit from tuning.",
            HIST,
        )

    # stress
    sh = res.stress_hist
    cov = sh[sh["Covered by backtest"]]
    if len(cov):
        for _, row in cov.iterrows():
            vals = row[strat].astype(float)
            add(
                "Stress (historical)",
                f"{row['Scenario']}: worst {vals.idxmin()} ({_pct(vals.min())}), "
                f"best {vals.idxmax()} ({_pct(vals.max())}), benchmark {_pct(row['Benchmark'])}.",
                HIST,
            )
    hs = res.stress_hypo
    if len(hs):
        # only scenarios whose impact differs across strategies carry information about the strategies
        var = hs.groupby("Scenario")["Impact"].transform(lambda x: x.round(8).nunique()) > 1
        sc = hs[var]
        if len(sc):
            r = sc.sort_values("Impact").iloc[0]
            add(
                "Stress (hypothetical)",
                f"Largest hypothetical loss among scenarios that differentiate strategies: {r['Strategy']} "
                f"under '{r['Scenario']}' ({_pct(r['Impact'])}).",
                MODEL,
                f"Method: {r['Method']}. Model estimate, not an observed outcome.",
            )
        flat = hs.groupby("Scenario")["Impact"].agg(lambda x: x.round(8).nunique() == 1)
        if flat.any():
            add(
                "Stress (hypothetical)",
                f"{int(flat.sum())} scenario(s) apply the same loss to every strategy (uniform broad-market shocks), "
                "so they do not rank strategies.",
                MODEL,
            )
    # diagnostics
    dg = wa.diagnostics
    nn = int((dg["Jarque-Bera p-value"] < 0.05).sum())
    add(
        "Diagnostics",
        f"{nn} of {len(dg)} return series reject normality (Jarque-Bera, 5%); parametric VaR may understate tail risk. "
        "Cornish-Fisher and historical VaR are reported alongside.",
        HIST,
    )
    sig = [s for s in strat if dg.loc[s, "Prob. Sharpe > 0 (PSR)"] > 0.95]
    add(
        "Diagnostics",
        f"Strategies with >95% probabilistic Sharpe (vs 0): {', '.join(sig) if sig else 'none'}.",
        HIST,
        "Sample-length dependent; not a forecast.",
    )
    note = res.meta.get("universe_note")
    if note:
        add(
            "Caveat",
            note,
            ASSUME,
            "Selection bias affects every strategy and the benchmark comparison; treat absolute returns with caution.",
        )
    add(
        "Caveat",
        "All performance figures are historical, in-sample for the choice of strategy set and parameters. "
        "Expected returns are estimated from past means and carry large estimation error. Past results do not guarantee future returns.",
        ASSUME,
    )
    if res.meta.get("synthetic"):
        add(
            "Caveat",
            "THIS RUN USES SYNTHETIC DEMO DATA. No figure here describes real markets.",
            ASSUME,
        )
    df = pd.DataFrame(rows, columns=["Section", "Finding", "Evidence type", "Detail"])
    return _prettify(df, strat, ["Finding", "Detail"])


def recommendations(res) -> pd.DataFrame:  # noqa: ANN001
    perf = res.full.perf
    strat = [c for c in perf.columns if c != "Benchmark"]
    sens = res.sens
    stable = None
    if len(sens):
        rank_std = (
            sens.assign(rank=sens.groupby(["Parameter", "Value"])["Sharpe"].rank(ascending=False))
            .groupby("Strategy")["rank"]
            .std()
        )
        stable = rank_std.idxmin()
    rows = [
        (
            "Strategy selection",
            f"Primary candidate on risk-adjusted basis: {perf.loc['Sharpe', strat].idxmax()}. "
            f"Defensive candidate (lowest volatility): {perf.loc['Volatility', strat].idxmin()}."
            + (
                " Maximum drawdowns are within "
                f"{perf.loc['Max drawdown', strat].max() - perf.loc['Max drawdown', strat].min():.1%} "
                "of each other, so volatility, not drawdown depth, separates the strategies."
                if perf.loc["Max drawdown", strat].max() - perf.loc["Max drawdown", strat].min() < 0.05
                else ""
            ),
            "Based on realised backtest; verify out-of-sample before capital allocation.",
        ),
        (
            "Stability",
            f"Most rank-stable strategy across parameter grids: {stable}."
            if stable
            else "Sweep not run.",
            "Rank stability is preferred over a single best-parameter result.",
        ),
        (
            "Costs",
            "Review rebalancing frequency and cost assumptions against actual execution costs.",
            "See Transaction Costs sheet for the net vs gross comparison.",
        ),
        (
            "Risk limits",
            f"Monitor {res.cfg.confidence:.0%} CVaR and component risk contributions at each rebalance.",
            "Concentration in risk can differ markedly from concentration in capital.",
        ),
        (
            "Governance",
            "Re-run the pipeline when new data arrives; compare against stored results for drift.",
            "All outputs regenerate from data + config.",
        ),
    ]
    df = pd.DataFrame(rows, columns=["Area", "Recommendation", "Rationale / caveat"])
    return _prettify(df, strat, ["Recommendation", "Rationale / caveat"])
