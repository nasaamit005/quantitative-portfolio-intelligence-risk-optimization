"""Plotly chart library. Every figure is built from Results/WindowAnalytics data, never hardcoded."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analytics import BENCH, WindowAnalytics

PALETTE = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#ea580c"]
BENCH_COLOR = "#64748b"
FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def pretty(name: str) -> str:
    return name if name == BENCH else name.replace("_", " ").title().replace("Hrp", "HRP")


@dataclass
class Ctx:
    res: object  # Results
    wa: WindowAnalytics
    strategies: list[str]  # selected (may include Benchmark)
    param: str | None = None

    @property
    def cols(self) -> list[str]:
        return [c for c in self.wa.returns.columns if c in self.strategies] or list(
            self.wa.returns.columns
        )

    def color(self, col: str) -> str:
        if col == BENCH:
            return BENCH_COLOR
        names = [c for c in self.res.strategy_returns.columns if c != BENCH]
        return PALETTE[names.index(col) % len(PALETTE)] if col in names else PALETTE[0]


def _base(
    fig: go.Figure,
    title: str,
    xt: str | None = None,
    yt: str | None = None,
    pct: bool = False,
    h: int = 430,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 16, "color": "#0f172a"}},
        template="plotly_white",
        height=h,
        font={"family": FONT, "size": 12, "color": "#334155"},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text=xt, gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(
        title_text=yt, gridcolor="#eef2f7", zeroline=False, tickformat=".0%" if pct else None
    )
    return fig


def _line(
    ctx: Ctx, df: pd.DataFrame, title: str, yt: str, pct: bool = False, hline: float | None = None
) -> go.Figure:
    fig = go.Figure()
    for c in ctx.cols:
        if c in df:
            fig.add_scatter(
                x=df.index,
                y=df[c],
                name=pretty(c),
                mode="lines",
                line={
                    "color": ctx.color(c),
                    "width": 2.4 if c != BENCH else 1.8,
                    "dash": "dash" if c == BENCH else "solid",
                },
                hovertemplate="%{y:.2%}" if pct else "%{y:.2f}",
            )
    if hline is not None:
        fig.add_hline(y=hline, line={"color": "#94a3b8", "width": 1, "dash": "dot"})
    return _base(fig, title, "Date", yt, pct)


# ---- individual charts: each returns a figure --------------------------------------------------


def equity(ctx: Ctx) -> go.Figure:
    return _line(
        ctx,
        ctx.wa.equity,
        "Cumulative performance (growth of 1.00, net of costs)",
        "Portfolio value (start = 1.00)",
    )


def drawdown(ctx: Ctx) -> go.Figure:
    fig = _line(
        ctx,
        ctx.wa.drawdown,
        "Underwater curve: drawdown from running peak",
        "Drawdown (%)",
        pct=True,
    )
    return fig


def excess(ctx: Ctx) -> go.Figure:
    df = ctx.wa.excess_cum
    fig = _line(
        ctx,
        df,
        "Cumulative relative performance vs benchmark",
        "Cumulative excess growth (%)",
        pct=True,
        hline=0,
    )
    return fig


def rolling_sharpe(ctx: Ctx) -> go.Figure:
    return _line(
        ctx, ctx.wa.rolling["Sharpe"], "Rolling Sharpe ratio", "Sharpe ratio (annualised)", hline=0
    )


def rolling_vol(ctx: Ctx) -> go.Figure:
    return _line(
        ctx,
        ctx.wa.rolling["Volatility"],
        "Rolling annualised volatility",
        "Volatility (% p.a.)",
        pct=True,
    )


def rolling_return(ctx: Ctx) -> go.Figure:
    return _line(
        ctx,
        ctx.wa.rolling["Return"],
        "Rolling annualised return",
        "Return (% p.a.)",
        pct=True,
        hline=0,
    )


def rolling_beta(ctx: Ctx) -> go.Figure:
    return _line(
        ctx,
        ctx.wa.rolling["Beta"].drop(columns=BENCH, errors="ignore"),
        "Rolling beta to benchmark",
        "Beta",
        hline=1,
    )


def metric_bars(ctx: Ctx) -> go.Figure:
    metrics = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=metrics, vertical_spacing=0.2)
    for i, mt in enumerate(metrics):
        vals = ctx.wa.perf.loc[mt, ctx.cols]
        pct = mt in ("CAGR", "Volatility", "Max drawdown")
        fig.add_bar(
            x=[pretty(c) for c in vals.index],
            y=vals.values,
            marker_color=[ctx.color(c) for c in vals.index],
            text=[f"{v:.1%}" if pct else f"{v:.2f}" for v in vals.values],
            textposition="outside",
            showlegend=False,
            row=i // 3 + 1,
            col=i % 3 + 1,
        )
        fig.update_yaxes(tickformat=".0%" if pct else None, row=i // 3 + 1, col=i % 3 + 1)
    return _base(fig, "Strategy comparison across key metrics", h=620).update_layout(
        hovermode="closest"
    )


def risk_return(ctx: Ctx) -> go.Figure:
    p = ctx.wa.perf
    fig = go.Figure()
    for c in ctx.cols:
        fig.add_scatter(
            x=[p.loc["Volatility", c]],
            y=[p.loc["CAGR", c]],
            mode="markers+text",
            text=[pretty(c)],
            textposition="top center",
            name=pretty(c),
            marker={
                "size": 16 + 20 * max(p.loc["Sharpe", c], 0),
                "color": ctx.color(c),
                "line": {"width": 1, "color": "white"},
            },
            hovertemplate=f"{pretty(c)}<br>Vol %{{x:.1%}}<br>CAGR %{{y:.1%}}<br>Sharpe {p.loc['Sharpe', c]:.2f}<extra></extra>",
        )
    return _base(
        fig, "Realised risk-return (marker size ~ Sharpe)", "Annualised volatility (%)", "CAGR (%)"
    ).update_layout(hovermode="closest", xaxis_tickformat=".0%", yaxis_tickformat=".0%")


def frontier(ctx: Ctx) -> go.Figure:
    r = ctx.res
    cfg = r.cfg
    fig = go.Figure()
    rp = r.random_pf
    fig.add_scatter(
        x=rp["Volatility"],
        y=rp["Return"],
        mode="markers",
        name="Random feasible portfolios",
        marker={
            "size": 4,
            "color": rp["Sharpe"],
            "colorscale": "Viridis",
            "opacity": 0.55,
            "colorbar": {"title": "Sharpe", "thickness": 12},
        },
        hovertemplate="Vol %{x:.1%}<br>Ret %{y:.1%}<extra></extra>",
    )
    f = r.frontier
    fig.add_scatter(
        x=f["Volatility"],
        y=f["Return"],
        mode="lines",
        name="Efficient frontier",
        line={"color": "#0f172a", "width": 3},
    )
    tv = f["Sharpe"].idxmax()
    top_vol = float(max(f["Volatility"].max(), r.asset_stats["Volatility"].max()) * 1.05)
    fig.add_scatter(
        x=[0, top_vol],
        y=[cfg.risk_free_rate, cfg.risk_free_rate + f.loc[tv, "Sharpe"] * top_vol],
        mode="lines",
        name="Capital allocation line",
        line={"color": "#dc2626", "dash": "dash", "width": 2},
    )
    fig.add_scatter(
        x=r.asset_stats["Volatility"],
        y=r.asset_stats["Annual return"],
        mode="markers+text",
        text=r.asset_stats.index,
        textposition="top right",
        name="Individual assets",
        marker={"symbol": "diamond", "size": 10, "color": "#475569"},
    )
    sym = {
        "min_vol": "triangle-down",
        "max_sharpe": "star",
        "minimum_volatility": "triangle-down",
        "maximum_sharpe": "star",
        "equal_weight": "square",
        "risk_parity": "pentagon",
        "hrp": "hexagon",
    }
    for i, (s, row) in enumerate(r.static_stats.iterrows()):
        fig.add_scatter(
            x=[row["Volatility"]],
            y=[row["Expected return"]],
            mode="markers",
            name=pretty(s),
            marker={
                "symbol": sym.get(s, "circle"),
                "size": 15,
                "color": PALETTE[i % len(PALETTE)],
                "line": {"width": 1.5, "color": "white"},
            },
        )
    return _base(
        fig,
        "Efficient frontier and capital allocation line (full-sample, hypothetical)",
        "Annualised volatility (%)",
        "Expected annual return (%)",
        h=560,
    ).update_layout(hovermode="closest", xaxis_tickformat=".0%", yaxis_tickformat=".0%")


def static_allocation(ctx: Ctx) -> go.Figure:
    w = ctx.res.static_weights
    fig = go.Figure()
    for i, a in enumerate(w.index):
        fig.add_bar(
            x=[pretty(c) for c in w.columns],
            y=w.loc[a],
            name=a,
            marker_color=PALETTE[i % len(PALETTE)],
        )
    return _base(
        fig,
        "Optimal asset allocation by method (full-sample, hypothetical)",
        "Method",
        "Portfolio weight (%)",
        pct=True,
    ).update_layout(barmode="stack", hovermode="closest")


def allocation_area(ctx: Ctx) -> go.Figure:
    s = ctx.param if ctx.param in ctx.res.weights else ctx.res.strategies[0]
    w = ctx.res.weights[s].loc[ctx.wa.start - pd.Timedelta(days=45) : ctx.wa.end]
    fig = go.Figure()
    for i, a in enumerate(w.columns):
        fig.add_scatter(
            x=w.index,
            y=w[a],
            stackgroup="one",
            name=a,
            mode="lines",
            line={"width": 0.5},
            marker_color=PALETTE[i % len(PALETTE)] if i < 8 else None,
        )
    return _base(
        fig, f"Target weights over time: {pretty(s)}", "Rebalance date", "Weight (%)", pct=True
    )


def weight_distribution(ctx: Ctx) -> go.Figure:
    s = ctx.param if ctx.param in ctx.res.weights else ctx.res.strategies[0]
    w = ctx.res.weights[s]
    fig = go.Figure()
    for i, a in enumerate(w.columns):
        fig.add_box(y=w[a], name=a, marker_color=PALETTE[i % len(PALETTE)], boxmean=True)
    return _base(
        fig,
        f"Distribution of weights across rebalances: {pretty(s)}",
        "Asset",
        "Weight (%)",
        pct=True,
    ).update_layout(hovermode="closest", showlegend=False)


def _heat(
    df: pd.DataFrame, title: str, zmin: float, zmax: float, fmt: str, scale: str = "RdBu"
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            zmin=zmin,
            zmax=zmax,
            colorscale=scale,
            text=np.round(df.values, 2),
            texttemplate=fmt,
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        )
    )
    return (
        _base(fig, title, h=520)
        .update_layout(hovermode="closest")
        .update_yaxes(autorange="reversed")
    )


def corr_heatmap(ctx: Ctx) -> go.Figure:
    return _heat(ctx.res.corr, "Correlation matrix of daily asset returns", -1, 1, "%{text}")


def cov_heatmap(ctx: Ctx) -> go.Figure:
    c = ctx.res.cov
    return _heat(c, "Annualised covariance matrix", 0, float(c.values.max()), "%{text}", "Blues")


def risk_contrib(ctx: Ctx) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("% of portfolio risk", "Weight (capital)"),
        shared_yaxes=True,
    )
    for _i, (s, rc) in enumerate(ctx.res.risk_contrib.items()):
        if s not in ctx.strategies:
            continue
        fig.add_bar(
            y=rc.index,
            x=rc["% of portfolio risk"],
            orientation="h",
            name=pretty(s),
            marker_color=ctx.color(s),
            row=1,
            col=1,
        )
        fig.add_bar(
            y=rc.index,
            x=rc["Weight"],
            orientation="h",
            name=pretty(s),
            marker_color=ctx.color(s),
            showlegend=False,
            row=1,
            col=2,
        )
    fig.update_xaxes(tickformat=".0%")
    return _base(
        fig, "Component risk contribution vs capital weight (latest rebalance)", h=520
    ).update_layout(barmode="group", hovermode="closest")


def var_bars(ctx: Ctx) -> go.Figure:
    rt = ctx.wa.risk_table.loc[ctx.cols]
    mc = ctx.res.mc_table
    fig = go.Figure()
    series = {
        "Historical VaR": "Historical VaR",
        "Parametric VaR": "Parametric VaR",
        "Cornish-Fisher VaR": "Cornish-Fisher VaR",
        "Historical CVaR": "Historical CVaR",
    }
    for i, (lab, col) in enumerate(series.items()):
        fig.add_bar(x=[pretty(c) for c in rt.index], y=rt[col], name=lab, marker_color=PALETTE[i])
    mcv = [mc.loc[c, "Monte Carlo VaR"] if c in mc.index else np.nan for c in rt.index]
    fig.add_bar(
        x=[pretty(c) for c in rt.index],
        y=mcv,
        name="Monte Carlo VaR (latest weights)",
        marker_color=PALETTE[4],
    )
    return _base(
        fig,
        f"Value at Risk and CVaR ({ctx.res.cfg.confidence:.0%}, {ctx.res.cfg.var_horizon}-day)",
        "Strategy",
        "Loss (% of portfolio)",
        pct=True,
    ).update_layout(barmode="group", hovermode="closest")


def return_dist(ctx: Ctx) -> go.Figure:
    c = ctx.param if ctx.param in ctx.wa.returns.columns else ctx.cols[0]
    r = ctx.wa.returns[c]
    cfg = ctx.res.cfg
    hv, hc = ctx.wa.risk_table.loc[c, ["Historical VaR", "Historical CVaR"]]
    x = np.linspace(r.min(), r.max(), 200)
    pdf = np.exp(-0.5 * ((x - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
    fig = go.Figure()
    fig.add_histogram(
        x=r,
        histnorm="probability density",
        nbinsx=80,
        name="Daily returns",
        marker_color=ctx.color(c),
        opacity=0.75,
    )
    fig.add_scatter(x=x, y=pdf, name="Normal fit", line={"color": "#0f172a", "dash": "dash"})
    fig.add_vline(x=-hv, line={"color": "#d97706", "width": 2}, annotation_text=f"VaR {hv:.2%}")
    fig.add_vline(
        x=-hc,
        line={"color": "#dc2626", "width": 2},
        annotation_text=f"CVaR {hc:.2%}",
        annotation_position="bottom left",
    )
    return _base(
        fig,
        f"Return distribution with {cfg.confidence:.0%} VaR/CVaR: {pretty(c)}",
        "Daily return",
        "Density",
    ).update_layout(hovermode="closest", xaxis_tickformat=".1%")


def mc_dist(ctx: Ctx) -> go.Figure:
    s = ctx.param if ctx.param in ctx.res.mc_sample else ctx.res.strategies[0]
    sim = ctx.res.mc_sample[s]
    v, c = ctx.res.mc_table.loc[s]
    fig = go.Figure(
        go.Histogram(x=sim, nbinsx=80, marker_color=PALETTE[0], opacity=0.8, histnorm="probability")
    )
    fig.add_vline(x=-v, line={"color": "#d97706", "width": 2}, annotation_text=f"VaR {v:.2%}")
    fig.add_vline(
        x=-c,
        line={"color": "#dc2626", "width": 2},
        annotation_text=f"CVaR {c:.2%}",
        annotation_position="bottom left",
    )
    return _base(
        fig,
        f"Monte Carlo simulated {ctx.res.cfg.var_horizon}-day return, latest weights: {pretty(s)} ({len(sim):,} paths)",
        "Simulated return",
        "Probability",
    ).update_layout(hovermode="closest", xaxis_tickformat=".1%")


def monthly_heatmap(ctx: Ctx) -> go.Figure:
    c = ctx.param if ctx.param in ctx.wa.returns.columns else ctx.cols[0]
    t = ctx.wa.monthly[c].drop(columns="Year")
    lim = float(np.nanmax(np.abs(t.values)))
    fig = go.Figure(
        go.Heatmap(
            z=t.values,
            x=list(t.columns),
            y=[str(i) for i in t.index],
            zmin=-lim,
            zmax=lim,
            colorscale="RdYlGn",
            text=t.values,
            texttemplate="%{text:.1%}",
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        )
    )
    return (
        _base(fig, f"Monthly returns: {pretty(c)}", h=max(320, 28 * len(t) + 120))
        .update_layout(hovermode="closest")
        .update_yaxes(autorange="reversed")
    )


def annual_bars(ctx: Ctx) -> go.Figure:
    a = ctx.wa.annual
    fig = go.Figure()
    for c in ctx.cols:
        fig.add_bar(x=[str(i) for i in a.index], y=a[c], name=pretty(c), marker_color=ctx.color(c))
    return _base(
        fig,
        "Calendar-year returns (first/last year may be partial)",
        "Year",
        "Return (%)",
        pct=True,
    ).update_layout(barmode="group", hovermode="closest")


def _sens(ctx: Ctx, label: str, metric: str, ytitle: str, pct: bool) -> go.Figure:
    df = ctx.res.sens
    d = df[df["Parameter"] == label]
    fig = go.Figure()
    order = list(dict.fromkeys(d["Value"]))
    for s in ctx.res.strategies:
        if s in ctx.strategies and s in set(d["Strategy"]):
            x = d[d["Strategy"] == s]
            fig.add_scatter(
                x=x["Value"],
                y=x[metric],
                mode="lines+markers",
                name=pretty(s),
                line={"color": ctx.color(s), "width": 2.4},
            )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    return _base(fig, f"Sensitivity of {metric} to {label.lower()}", label, ytitle, pct)


def sens_lookback(ctx: Ctx) -> go.Figure:
    return _sens(ctx, "Lookback (days)", "Sharpe", "Sharpe ratio", False)


def sens_cost(ctx: Ctx) -> go.Figure:
    return _sens(ctx, "Transaction cost (bps)", "CAGR", "Net CAGR (%)", True)


def sens_maxw(ctx: Ctx) -> go.Figure:
    return _sens(ctx, "Max weight", "Sharpe", "Sharpe ratio", False)


def sens_rebal(ctx: Ctx) -> go.Figure:
    return _sens(ctx, "Rebalancing frequency", "Sharpe", "Sharpe ratio", False)


def sens_heat(ctx: Ctx) -> go.Figure:
    d = ctx.res.sens.assign(Cell=lambda x: x["Parameter"].str.split(" ").str[0] + "=" + x["Value"])
    t = d.pivot_table(index="Strategy", columns="Cell", values="Sharpe", sort=False)
    return _heat(
        t,
        "Sharpe ratio across all parameter settings",
        float(t.values.min()),
        float(t.values.max()),
        "%{text}",
        "RdYlGn",
    ).update_layout(height=380)


def regimes(ctx: Ctx) -> go.Figure:
    d = ctx.res.regimes
    d = d[d["Strategy"].isin(ctx.cols) & d["Regime type"].isin(["Trend", "Volatility"])]
    fig = go.Figure()
    for c in ctx.cols:
        x = d[d["Strategy"] == c]
        fig.add_bar(
            x=x["Regime"],
            y=x["Annualised return"],
            name=pretty(c),
            marker_color=ctx.color(c),
            customdata=x[["Sharpe", "Days"]].values,
            hovertemplate="%{x}<br>Return %{y:.1%}<br>Sharpe %{customdata[0]:.2f}<br>Days %{customdata[1]}<extra></extra>",
        )
    return _base(
        fig, "Annualised return by market regime", "Regime", "Annualised return (%)", pct=True
    ).update_layout(barmode="group", hovermode="closest")


def regime_timeline(ctx: Ctx) -> go.Figure:
    lab = ctx.res.regime_labels.dropna()
    bench_eq = (1 + ctx.res.bench_full).cumprod().loc[ctx.wa.start : ctx.wa.end]
    fig = go.Figure(
        go.Scatter(
            x=bench_eq.index,
            y=bench_eq,
            name="Benchmark (growth of 1)",
            line={"color": BENCH_COLOR},
        )
    )
    bear = lab["Trend"].reindex(bench_eq.index) == "Bear (below MA)"
    start = None
    for t, flag in bear.items():
        if flag and start is None:
            start = t
        if not flag and start is not None:
            fig.add_vrect(x0=start, x1=t, fillcolor="#ef4444", opacity=0.12, line_width=0)
            start = None
    if start is not None:
        fig.add_vrect(
            x0=start, x1=bench_eq.index[-1], fillcolor="#ef4444", opacity=0.12, line_width=0
        )
    return _base(
        fig,
        "Benchmark with bearish-trend regimes shaded (below moving average)",
        "Date",
        "Growth of 1.00",
    )


def stress_hist(ctx: Ctx) -> go.Figure:
    d = ctx.res.stress_hist
    d = d[d["Covered by backtest"]]
    fig = go.Figure()
    for c in ctx.cols:
        fig.add_bar(x=d["Scenario"], y=d[c], name=pretty(c), marker_color=ctx.color(c))
    return _base(
        fig,
        "Historical stress windows: cumulative return",
        "Scenario",
        "Cumulative return (%)",
        pct=True,
    ).update_layout(barmode="group", hovermode="closest")


def stress_hypo(ctx: Ctx) -> go.Figure:
    d = ctx.res.stress_hypo
    d = d[d.groupby("Scenario")["Impact"].transform(lambda x: x.round(8).nunique()) > 1]
    t = d.pivot_table(index="Scenario", columns="Strategy", values="Impact", sort=False)
    t = t[[c for c in ctx.cols if c in t.columns]]
    lim = float(np.nanmax(np.abs(t.values)))
    fig = go.Figure(
        go.Heatmap(
            z=t.values,
            x=[pretty(c) for c in t.columns],
            y=list(t.index),
            zmin=-lim,
            zmax=0,
            colorscale="Reds_r",
            text=t.values,
            texttemplate="%{text:.1%}",
            hovertemplate="%{y}<br>%{x}: %{z:.2%}<extra></extra>",
        )
    )
    fig = _base(
        fig,
        "Hypothetical scenarios: portfolio impact by strategy (model estimates)",
        h=max(360, 26 * len(t) + 120),
    )
    return fig.update_layout(
        hovermode="closest", margin={"l": 330, "r": 20, "t": 60, "b": 50}
    ).update_yaxes(autorange="reversed")


def wf_year(ctx: Ctx) -> go.Figure:
    d = ctx.res.wf_year
    fig = go.Figure()
    for c in ctx.cols:
        x = d[d["Strategy"] == c]
        fig.add_scatter(
            x=x["Year"],
            y=x["Sharpe"],
            mode="lines+markers",
            name=pretty(c),
            line={"color": ctx.color(c)},
        )
    return _base(fig, "Walk-forward: out-of-sample Sharpe by calendar year", "Year", "Sharpe ratio")


def wf_select(ctx: Ctx) -> go.Figure:
    d = ctx.res.wf_select
    fig = go.Figure()
    for s in ctx.res.strategies:
        if s in ctx.strategies and len(d):
            x = d[d["Strategy"] == s]
            fig.add_bar(
                x=x["Year"],
                y=x["Selected return (OOS)"] - x["Average of all lookbacks"],
                name=pretty(s),
                marker_color=ctx.color(s),
            )
    return _base(
        fig,
        "Walk-forward lookback selection: OOS return minus average-lookback return",
        "Year",
        "Selection advantage (% pts)",
        pct=True,
    ).update_layout(barmode="group", hovermode="closest")


def cost_drag(ctx: Ctx) -> go.Figure:
    ct = ctx.res.cost_table
    ct = ct.loc[[c for c in ct.index if c in ctx.cols]]
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Gross vs net CAGR", "Annual turnover (one-way, x capital)")
    )
    fig.add_bar(
        x=[pretty(c) for c in ct.index],
        y=ct["Gross CAGR"],
        name="Gross",
        marker_color="#94a3b8",
        row=1,
        col=1,
    )
    fig.add_bar(
        x=[pretty(c) for c in ct.index],
        y=ct["Net CAGR"],
        name="Net of costs",
        marker_color=PALETTE[0],
        row=1,
        col=1,
    )
    fig.add_bar(
        x=[pretty(c) for c in ct.index],
        y=ct["Annual turnover"],
        name="Turnover",
        marker_color=PALETTE[2],
        row=1,
        col=2,
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    return _base(
        fig, f"Transaction-cost impact at {ctx.res.cfg.cost_bps:g} bps per unit turnover", h=400
    ).update_layout(barmode="group", hovermode="closest")


def turnover(ctx: Ctx) -> go.Figure:
    fig = go.Figure()
    for s, t in ctx.res.turnover.items():
        if s in ctx.strategies:
            t = t.loc[ctx.wa.start : ctx.wa.end]
            fig.add_scatter(
                x=t.index, y=t, name=pretty(s), mode="lines", line={"color": ctx.color(s)}
            )
    return _base(
        fig, "Turnover at each rebalance", "Rebalance date", "One-way turnover (x capital)"
    )


@dataclass(frozen=True)
class ChartSpec:
    fn: Callable[[Ctx], go.Figure]
    section: str
    caption: str
    needs_param: str | None = None  # "strategy" or "series"


REGISTRY: dict[str, ChartSpec] = {
    "equity": ChartSpec(
        equity,
        "performance",
        "Net-of-cost compounded growth of each strategy and the benchmark over the selected window.",
    ),
    "drawdown": ChartSpec(
        drawdown,
        "performance",
        "Depth of loss from the prior peak; longer, deeper troughs imply harder recovery.",
    ),
    "excess": ChartSpec(
        excess,
        "performance",
        "Strategy growth divided by benchmark growth, minus one. Rising = outperforming.",
    ),
    "metric_bars": ChartSpec(
        metric_bars,
        "performance",
        "Side-by-side CAGR, volatility, Sharpe, Sortino, drawdown and Calmar.",
    ),
    "risk_return": ChartSpec(
        risk_return,
        "performance",
        "Realised annualised return against volatility for the selected window.",
    ),
    "annual_bars": ChartSpec(
        annual_bars,
        "performance",
        "Calendar-year returns; partial first/last years are included as-is.",
    ),
    "monthly_heatmap": ChartSpec(
        monthly_heatmap,
        "performance",
        "Month-by-month compounded returns for one series.",
        "series",
    ),
    "rolling_sharpe": ChartSpec(
        rolling_sharpe, "rolling", "Rolling Sharpe over the configured window."
    ),
    "rolling_vol": ChartSpec(rolling_vol, "rolling", "Rolling annualised volatility."),
    "rolling_return": ChartSpec(rolling_return, "rolling", "Rolling annualised compounded return."),
    "rolling_beta": ChartSpec(
        rolling_beta, "rolling", "Rolling market beta against the benchmark."
    ),
    "frontier": ChartSpec(
        frontier,
        "optimization",
        "Full-sample, in-sample frontier with bounded weights. Hypothetical: not a tradeable result.",
    ),
    "static_allocation": ChartSpec(
        static_allocation,
        "optimization",
        "Full-sample optimal weights per method under the configured weight bounds.",
    ),
    "allocation_area": ChartSpec(
        allocation_area,
        "optimization",
        "Walk-forward target weights through time for one strategy.",
        "strategy",
    ),
    "weight_distribution": ChartSpec(
        weight_distribution,
        "optimization",
        "Spread of each asset's target weight across rebalances (stability of allocation).",
        "strategy",
    ),
    "turnover": ChartSpec(turnover, "optimization", "One-way turnover at each rebalance date."),
    "cost_drag": ChartSpec(
        cost_drag, "optimization", "Gross vs net CAGR and annual turnover: the cost of trading."
    ),
    "corr_heatmap": ChartSpec(
        corr_heatmap, "risk", "Pairwise correlation of daily asset returns over the full sample."
    ),
    "cov_heatmap": ChartSpec(
        cov_heatmap, "risk", "Annualised covariance matrix of asset returns (before shrinkage)."
    ),
    "risk_contrib": ChartSpec(
        risk_contrib,
        "risk",
        "Share of portfolio volatility from each asset vs its capital weight (latest rebalance).",
    ),
    "var_bars": ChartSpec(
        var_bars,
        "risk",
        "VaR by method; CVaR is the average loss beyond VaR. Monte Carlo uses latest weights.",
    ),
    "return_dist": ChartSpec(
        return_dist,
        "risk",
        "Histogram of daily returns with normal fit and historical VaR/CVaR markers.",
        "series",
    ),
    "mc_dist": ChartSpec(
        mc_dist,
        "risk",
        "Monte Carlo distribution under multivariate normality, at latest weights.",
        "strategy",
    ),
    "sens_heat": ChartSpec(
        sens_heat,
        "robustness",
        "Sharpe ratio for every strategy and every parameter setting tested.",
    ),
    "sens_lookback": ChartSpec(
        sens_lookback, "robustness", "Sharpe vs estimation lookback window."
    ),
    "sens_cost": ChartSpec(sens_cost, "robustness", "Net CAGR vs assumed transaction cost."),
    "sens_maxw": ChartSpec(sens_maxw, "robustness", "Sharpe vs maximum single-asset weight."),
    "sens_rebal": ChartSpec(sens_rebal, "robustness", "Sharpe vs rebalancing frequency."),
    "wf_year": ChartSpec(wf_year, "robustness", "Out-of-sample Sharpe by calendar year."),
    "wf_select": ChartSpec(
        wf_select,
        "robustness",
        "Value (or harm) of re-selecting the lookback each year using only past data.",
    ),
    "regimes": ChartSpec(
        regimes,
        "stress",
        "Annualised return in bull/bear (trend) and high/normal-volatility regimes, using the project's own regime classification.",
    ),
    "regime_timeline": ChartSpec(
        regime_timeline, "stress", "Benchmark path with bearish regimes shaded."
    ),
    "stress_hist": ChartSpec(
        stress_hist,
        "stress",
        "Realised cumulative return in crisis windows covered by the backtest.",
    ),
    "stress_hypo": ChartSpec(
        stress_hypo,
        "stress",
        "Modelled portfolio impact of hypothetical scenarios that differ across strategies. Model estimates, not observed outcomes.",
    ),
}


_NEEDS = {
    "wf_select": lambda r: len(r.wf_select) > 0,
    "wf_year": lambda r: len(r.wf_year) > 0,
    "regimes": lambda r: len(r.regimes) > 0,
    "regime_timeline": lambda r: len(r.regime_labels.dropna()) > 0,
    "stress_hist": lambda r: bool(r.stress_hist["Covered by backtest"].any()),
    "stress_hypo": lambda r: len(r.stress_hypo) > 0,
    "mc_dist": lambda r: bool(r.mc_sample),
    "frontier": lambda r: len(r.frontier) > 0,
    "static_allocation": lambda r: len(r.static_weights) > 0,
    **{
        k: (lambda r: len(r.sens) > 0)
        for k in ("sens_heat", "sens_lookback", "sens_cost", "sens_maxw", "sens_rebal")
    },
}


def is_available(name: str, res) -> bool:
    """False when the run has no data for a chart (e.g. an analysis that was not performed), so it is hidden, not shown empty."""
    return bool(_NEEDS.get(name, lambda r: True)(res))


def interpret(name: str, ctx: Ctx) -> str:
    """Data-driven one-line takeaway for a chart (computed from the same data as the figure)."""
    p, res = ctx.wa.perf, ctx.res
    s = [c for c in ctx.cols if c != BENCH] or ctx.cols
    try:
        if name == "equity":
            b = p.loc["Total return", s].idxmax()
            return f"{pretty(b)} compounded to {p.loc['Total return', b]:+.1%} total; benchmark {p.loc['Total return', BENCH]:+.1%}."
        if name == "drawdown":
            w = p.loc["Max drawdown", s].idxmin()
            return f"Deepest drawdown among selected strategies: {pretty(w)} at {p.loc['Max drawdown', w]:.1%} (benchmark {p.loc['Max drawdown', BENCH]:.1%})."
        if name == "metric_bars":
            return f"Highest Sharpe: {pretty(p.loc['Sharpe', s].idxmax())} ({p.loc['Sharpe', s].max():.2f})."
        if name == "frontier":
            r = res.static_stats
            return f"Highest in-sample Sharpe: {pretty(r['Sharpe'].idxmax())} ({r['Sharpe'].max():.2f}); frontier optimum is sensitive to return estimates."
        if name == "corr_heatmap":
            c = res.corr.where(~np.eye(len(res.corr), dtype=bool)).stack()
            hi, lo = c.idxmax(), c.idxmin()
            return f"Most correlated pair: {hi[0]}/{hi[1]} ({c.max():.2f}); best diversifier pair: {lo[0]}/{lo[1]} ({c.min():.2f})."
        if name == "risk_contrib":
            rows = []
            for k, rc in res.risk_contrib.items():
                if k in ctx.strategies:
                    t = rc["% of portfolio risk"].idxmax()
                    rows.append(f"{pretty(k)}: {t} {rc.loc[t, '% of portfolio risk']:.0%}")
            return "Largest risk contributor per strategy: " + "; ".join(rows) + "."
        if name == "cost_drag":
            ct = res.cost_table
            return f"Highest cost drag: {pretty(ct['Annual cost drag'].idxmax())} ({ct['Annual cost drag'].max():.2%} per year)."
        if name == "sens_heat" and len(res.sens):
            sp = res.sens.groupby("Strategy")["Sharpe"].agg(lambda x: x.max() - x.min())
            return f"Most parameter-sensitive: {pretty(sp.idxmax())} (Sharpe range {sp.max():.2f}); most stable: {pretty(sp.idxmin())} ({sp.min():.2f})."
        if name == "stress_hist":
            d = res.stress_hist[res.stress_hist["Covered by backtest"]]
            if len(d):
                return (
                    f"Covered windows: {len(d)}; worst benchmark window {d['Benchmark'].min():.1%}."
                )
            return "No named stress window is fully covered by the backtest period."
    except KeyError, ValueError:
        return ""
    return ""


def build(
    name: str,
    res,
    wa: WindowAnalytics,
    strategies: list[str] | None = None,
    param: str | None = None,
) -> tuple[go.Figure, str, str]:
    """Build a registered chart -> (figure, caption, interpretation)."""
    if name not in REGISTRY:
        raise KeyError(name)
    spec = REGISTRY[name]
    sel = strategies or list(wa.returns.columns)
    ctx = Ctx(res, wa, sel, param)
    return spec.fn(ctx), spec.caption, interpret(name, ctx)
