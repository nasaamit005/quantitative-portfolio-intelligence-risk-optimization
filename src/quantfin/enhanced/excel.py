"""Excel workbook builder (openpyxl): formula-driven core, tables, conditional formats, native charts."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as L
from openpyxl.worksheet.table import Table, TableStyleInfo

from .analytics import BENCH
from .calendar_utils import period_end_flags
from .charts import REGISTRY, is_available, pretty

logger = logging.getLogger(__name__)

FONT = "Arial"
NAVY, BLUE, LIGHT, GREY = "0F172A", "2563EB", "EFF6FF", "F1F5F9"
HDR = 4  # header row on data sheets
SR, EQ = "Strategy Returns", "Equity & Drawdown"


def _clean(v):
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.to_pydatetime()
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) or np.isinf(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is pd.NaT:
        return None
    return v


_COUNT_NAMES = ("observations", "days", "rebalances", "trading days", "count", "n")
_R1 = (
    "sharpe",
    "sortino",
    "calmar",
    "ratio",
    "skew",
    "kurt",
    "beta",
    "t-stat",
    "jarque",
    "ljung",
    "autocorr",
    "information",
    "turnover",
    "effective n",
    "r-squared",
    "capture",
    "lookback",
)
_R_P = ("p-value",)
_R_PROB = ("prob.",)
_R_PCT = (
    "return",
    "cagr",
    "vol",
    "drawdown",
    "var",
    "cvar",
    "weight",
    "alpha",
    "error",
    "deviation",
    "day",
    "loss",
    "depth",
    "contribution",
    "impact",
    "missing %",
    "zero-return",
    "% of",
    "rate",
    "excess",
    "cost",
    "lost",
    "mean",
    "min",
    "max",
    "std",
    "25%",
    "50%",
    "75%",
    "impact",
)


def fmt_for(col: str, s: pd.Series) -> str | None:
    c = str(col).lower()
    if pd.api.types.is_datetime64_any_dtype(s):
        return "yyyy-mm-dd"
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return None
    if c in _COUNT_NAMES or pd.api.types.is_integer_dtype(s):
        return "0" if c in ("year", "value") else "#,##0"
    if any(k in c for k in _R_P):
        return "0.0000"
    if any(k in c for k in _R_PROB):
        return "0.0%"
    if any(k in c for k in _R1):
        return "0.000"
    if any(k in c for k in _R_PCT):
        return "0.00%"
    return "0.000"


class Book:
    """Thin wrapper around an openpyxl Workbook that tracks a catalog of sheets/tables."""

    def __init__(self) -> None:
        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.catalog: list[dict] = []
        self._tcount = 0

    def sheet(
        self,
        name: str,
        desc: str,
        title: str | None = None,
        note: str | None = None,
        tab: str = BLUE,
    ):
        ws = self.wb.create_sheet(name[:31])
        ws.sheet_properties.tabColor = tab
        ws.sheet_view.showGridLines = False
        ws["A1"] = title or name
        ws["A1"].font = Font(name=FONT, size=16, bold=True, color=NAVY)
        if note:
            ws["A2"] = note
            ws["A2"].font = Font(name=FONT, size=9, italic=True, color="64748B")
        self.catalog.append({"Sheet": ws.title, "Description": desc, "Tables": 0, "Charts": 0})
        return ws

    def _bump(self, ws, key: str) -> None:
        for c in self.catalog:
            if c["Sheet"] == ws.title:
                c[key] += 1

    def table(
        self,
        ws,
        df: pd.DataFrame,
        top: int,
        left: int = 1,
        index: bool = True,
        title: str | None = None,
        fmts: dict[str, str] | None = None,
        as_table: bool = True,
        widths: bool = True,
    ) -> tuple[int, int, int]:
        """Write df with header at row `top`; returns (header_row, first_data_row, last_data_row)."""
        d = df.copy()
        if index:
            d = d.reset_index()
            d.columns = [str(c) if str(c) else "Index" for c in d.columns]
            if d.columns[0] in ("index", "level_0"):
                d = d.rename(columns={d.columns[0]: str(df.index.name or "Series")})
        d.columns = [str(c) for c in d.columns]
        if len(set(d.columns)) != len(d.columns):
            d.columns = [
                f"{c}_{i}" if list(d.columns).count(c) > 1 else c for i, c in enumerate(d.columns)
            ]
        if title:
            ws.cell(row=top - 1, column=left, value=title).font = Font(
                name=FONT, size=12, bold=True, color=BLUE
            )
        for j, col in enumerate(d.columns):
            cell = ws.cell(row=top, column=left + j, value=col)
            cell.font = Font(name=FONT, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor=NAVY)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        fmts = fmts or {}
        for j, col in enumerate(d.columns):
            nf = fmts.get(col, fmt_for(col, d[col]))
            vals = d[col].tolist()
            for i, v in enumerate(vals):
                cell = ws.cell(row=top + 1 + i, column=left + j, value=_clean(v))
                cell.font = Font(name=FONT, size=10)
                if nf:
                    cell.number_format = nf
            if widths:
                w = (
                    max(
                        [len(str(col))]
                        + [len(f"{v:.4f}" if isinstance(v, float) else str(v)) for v in vals[:200]]
                    )
                    + 2
                )
                cur = ws.column_dimensions[L(left + j)].width or 8
                ws.column_dimensions[L(left + j)].width = min(max(cur, w, 11), 60)
        first, last = top + 1, top + len(d)
        if as_table and len(d):
            self._tcount += 1
            ref = f"{L(left)}{top}:{L(left + len(d.columns) - 1)}{last}"
            t = Table(
                displayName=f"tbl_{self._tcount}_{re.sub(r'[^A-Za-z0-9]', '', ws.title)[:12]}",
                ref=ref,
            )
            t.tableStyleInfo = TableStyleInfo(name="TableStyleLight9", showRowStripes=True)
            ws.add_table(t)
            self._bump(ws, "Tables")
        ws.row_dimensions[top].height = 30
        return top, first, last

    def save(self, path: Path) -> None:
        self.wb.calculation.fullCalcOnLoad = True
        path.parent.mkdir(parents=True, exist_ok=True)
        self.wb.save(path)


def _line_chart(
    ws,
    title: str,
    ytitle: str,
    xtitle: str,
    cats: Reference,
    series: list[tuple[Reference, str]],
    anchor: str,
    pct: bool = False,
    w: float = 24,
    h: float = 11,
) -> None:
    ch = LineChart()
    ch.title, ch.width, ch.height = title, w, h
    ch.y_axis.title, ch.x_axis.title = ytitle, xtitle
    ch.y_axis.number_format = "0%" if pct else "0.00"
    ch.x_axis.number_format = "yyyy"
    ch.x_axis.majorTimeUnit = "years"
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    for ref, name in series:
        s = Series(ref, title=name)
        s.smooth = False
        s.marker.symbol = "none"
        s.graphicalProperties.line.width = 15000
        ch.series.append(s)
    ch.set_categories(cats)
    ws.add_chart(ch, anchor)


def _bar_chart(
    ws,
    title: str,
    ytitle: str,
    cats: Reference,
    data: Reference,
    anchor: str,
    pct: bool = False,
    w: float = 16,
    h: float = 9,
    stacked: bool = False,
    horizontal: bool = False,
    titles_from_data: bool = True,
) -> None:
    ch = BarChart()
    ch.type = "bar" if horizontal else "col"
    ch.title, ch.width, ch.height = title, w, h
    ch.y_axis.title = ytitle
    ch.y_axis.number_format = "0%" if pct else "0.00"
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    if stacked:
        ch.grouping, ch.overlap = "stacked", 100
    ch.add_data(data, titles_from_data=titles_from_data)
    ch.set_categories(cats)
    ws.add_chart(ch, anchor)


def build_workbook(res, path: str | Path) -> dict[str, int]:
    """Create the full analytical workbook. Returns summary counts."""
    cfg, wa = res.cfg, res.full
    bk = Book()
    strat_cols = list(res.strategy_returns.columns)
    n_s = len(strat_cols)
    sr = res.strategy_returns
    n = len(sr)
    last = HDR + n

    # ---------------- Executive dashboard placeholder (filled last so it can reference other sheets) -----
    ws_exec = bk.sheet(
        "Executive Dashboard",
        "KPIs, rankings, headline charts (formula-linked)",
        "Executive Dashboard",
        "All KPI cells are Excel formulas linked to 'Strategy Comparison'. Historical backtest; not a forecast.",
        NAVY,
    )

    # ---------------- Overview ----------------
    ov = bk.sheet(
        "Overview",
        "Project description, editable assumptions, methodology and formulas",
        "Project Overview",
    )
    ov["A3"] = (
        "Quantitative Portfolio Intelligence & Risk Optimization Platform: analytics extension"
    )
    ov["A3"].font = Font(name=FONT, bold=True, size=11)
    ov["B5"], ov["C5"] = "Assumption", "Value"
    for c in ("B5", "C5"):
        ov[c].font, ov[c].fill = (
            Font(name=FONT, bold=True, color="FFFFFF"),
            PatternFill("solid", fgColor=NAVY),
        )
    assumptions = [
        ("Periods per year", cfg.periods_per_year, "0"),
        ("Risk-free rate (annual)", cfg.risk_free_rate, "0.00%"),
        ("Transaction cost (bps per unit turnover)", cfg.cost_bps, "0.0"),
        ("Estimation lookback (days)", cfg.lookback, "0"),
        ("Rebalancing", cfg.rebalance, None),
        ("Max weight per asset", cfg.max_weight, "0.0%"),
        ("Min weight per asset", cfg.min_weight, "0.0%"),
        ("VaR confidence", cfg.confidence, "0.0%"),
        ("VaR horizon (days)", cfg.var_horizon, "0"),
        ("Covariance shrinkage", cfg.cov_shrinkage, "0.00"),
        ("Benchmark", res.meta.get("benchmark_label", cfg.benchmark), None),
        ("Backtest start", res.meta["start"], None),
        ("Backtest end", res.meta["end"], None),
        ("Data source", res.meta.get("data_source", "user supplied"), None),
    ]
    for i, (k, v, nf) in enumerate(assumptions):
        r = 6 + i
        ov.cell(row=r, column=2, value=k).font = Font(name=FONT, size=10)
        c = ov.cell(row=r, column=3, value=v)
        c.font = Font(name=FONT, size=10, color="0000FF")  # blue = input
        if nf:
            c.number_format = nf
    PPY, RF = "Overview!$C$6", "Overview!$C$7"
    ov["B21"] = (
        "Blue cells are inputs recorded from the run configuration. Periods per year and risk-free rate drive Excel formulas on 'Strategy Comparison'."
    )
    ov["B21"].font = Font(name=FONT, size=9, italic=True)
    if res.meta.get("synthetic"):
        ov["B22"] = (
            "WARNING: SYNTHETIC DEMO DATA. No number in this workbook describes real markets."
        )
        ov["B22"].font = Font(name=FONT, bold=True, color="DC2626")
    method = [
        (
            "Returns",
            "Simple daily returns; strategies are walk-forward: weights estimated at close t, earn return of t+1.",
        ),
        (
            "Costs",
            "Cost = sum|w_target - w_drifted| x bps / 10,000, charged on the next day's return.",
        ),
        ("CAGR", "(Equity_end)^(PPY/N) - 1"),
        ("Volatility", "STDEV(daily returns) x SQRT(PPY)"),
        ("Sharpe", "(AVERAGE(r) x PPY - Rf) / (STDEV(r) x SQRT(PPY))"),
        ("Sortino", "Mean excess x PPY / (downside deviation vs Rf x SQRT(PPY))"),
        ("Max drawdown", "MIN(Equity / running peak - 1)"),
        ("Calmar", "CAGR / |Max drawdown|"),
        (
            "Beta / Alpha",
            "OLS of excess strategy on excess benchmark returns; alpha annualised x PPY",
        ),
        ("Information ratio", "Mean(active return) x PPY / (STDEV(active) x SQRT(PPY))"),
        (
            "VaR / CVaR",
            "Historical quantile; Gaussian; Cornish-Fisher; Monte Carlo (multivariate normal, latest weights)",
        ),
        (
            "Risk contribution",
            "w_i x (Sigma w)_i / sigma_p; components sum to portfolio volatility",
        ),
        (
            "Optimisers",
            "Min-vol and max-Sharpe (SLSQP), risk parity (Spinu convex form, capped), HRP (Lopez de Prado), equal weight",
        ),
        (
            "Covariance",
            "Sample covariance shrunk toward scaled identity (intensity in assumptions)",
        ),
        (
            "Regimes",
            "Trend: benchmark vs moving average; Volatility: rolling vol vs expanding median; labels lagged one day",
        ),
        (
            "Stress",
            "Historical windows (only if covered); hypothetical = beta x shock, VaR x multiplier, correlations -> 1 (model estimates)",
        ),
        (
            "Walk-forward",
            "Each year the lookback with best trailing Sharpe is chosen using only prior data",
        ),
        (
            "Limitations",
            "Backtests are historical; expected returns estimated from past means; results are not guarantees of future returns.",
        ),
    ]
    bk.table(
        ov,
        pd.DataFrame(method, columns=["Topic", "Definition"]),
        top=26,
        index=False,
        title="Methodology and formulas",
        as_table=False,
    )
    ov.column_dimensions["B"].width, ov.column_dimensions["C"].width = 42, 28
    ov.column_dimensions["A"].width = 18
    for r in range(27, 27 + len(method)):
        ov.cell(row=r, column=2).alignment = Alignment(wrap_text=True)

    # ---------------- Data Quality ----------------
    dq = bk.sheet(
        "Data Quality", "Coverage and integrity checks on the input series", "Data Quality Report"
    )
    bk.table(dq, res.quality, top=HDR, index=False, title=None)
    notes = res.meta.get("data_notes") or ["No data adjustments were required."]
    for i, t in enumerate(notes):
        dq.cell(row=HDR + len(res.quality) + 3 + i, column=1, value=t).font = Font(
            name=FONT, size=10
        )
    dq.freeze_panes = dq.cell(row=HDR + 1, column=2)

    # ---------------- Descriptive stats / correlation ----------------
    ds = bk.sheet(
        "Descriptive Stats",
        "Daily-return descriptive statistics and asset-level annualised metrics",
    )
    bk.table(ds, res.desc, top=HDR, title=None, fmts={"count": "#,##0"})
    bk.table(
        ds,
        res.asset_stats,
        top=HDR + len(res.desc) + 4,
        title="Annualised asset statistics (historical)",
    )
    ds.freeze_panes = ds.cell(row=HDR + 1, column=2)

    cm = bk.sheet(
        "Correlation & Covariance", "Correlation heatmap and annualised covariance matrix"
    )
    nA = res.corr.shape[0]
    h, f, ll = bk.table(
        cm, res.corr, top=HDR, fmts={c: "0.00" for c in res.corr.columns}, as_table=False
    )
    cm.conditional_formatting.add(
        f"B{f}:{L(1 + nA)}{ll}",
        ColorScaleRule(
            start_type="num",
            start_value=-1,
            start_color="2563EB",
            mid_type="num",
            mid_value=0,
            mid_color="FFFFFF",
            end_type="num",
            end_value=1,
            end_color="DC2626",
        ),
    )
    cm.cell(row=HDR - 1, column=1, value="Correlation matrix (daily returns)").font = Font(
        name=FONT, bold=True, color=BLUE, size=12
    )
    h2, f2, l2 = bk.table(
        cm,
        res.cov,
        top=ll + 4,
        title="Annualised covariance matrix",
        fmts={c: "0.0000" for c in res.cov.columns},
        as_table=False,
    )
    cm.conditional_formatting.add(
        f"B{f2}:{L(1 + nA)}{l2}",
        ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="2563EB"),
    )

    # ---------------- Strategy Returns (inputs to formulas) ----------------
    sh = bk.sheet(
        "Strategy Returns",
        "Daily net-of-cost returns per strategy and benchmark (inputs for Excel formulas)",
        note="Daily simple returns. Strategies are net of transaction costs; benchmark aligned to the same dates.",
    )
    out = sr.copy()
    out.columns = [pretty(c) for c in out.columns]
    out.index.name = "Date"
    bk.table(sh, out, top=HDR, as_table=False, fmts={c: "0.0000%" for c in out.columns})
    sh.freeze_panes = sh.cell(row=HDR + 1, column=2)
    rng = {c: f"'{SR}'!${L(2 + j)}${HDR + 1}:${L(2 + j)}${last}" for j, c in enumerate(strat_cols)}

    # ---------------- Equity & Drawdown (formulas) ----------------
    eq = bk.sheet(
        EQ,
        "Formula-driven equity, running peak and drawdown for every series",
        note="Equity_t = Equity_(t-1) x (1 + r_t); Peak = MAX(prior peak, equity, 1); Drawdown = Equity/Peak - 1",
    )
    eq.cell(row=HDR, column=1, value="Date")
    for j, c in enumerate(strat_cols):
        for k, lab in enumerate(("Equity", "Peak", "Drawdown")):
            eq.cell(row=HDR, column=2 + 3 * j + k, value=f"{pretty(c)} {lab}")
    for cell in eq[HDR]:
        if cell.value:
            cell.font, cell.fill = (
                Font(name=FONT, bold=True, color="FFFFFF"),
                PatternFill("solid", fgColor=NAVY),
            )
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for i in range(n):
        r = HDR + 1 + i
        eq.cell(row=r, column=1, value=sr.index[i].to_pydatetime()).number_format = "yyyy-mm-dd"
        for j in range(n_s):
            e, p, d = L(2 + 3 * j), L(3 + 3 * j), L(4 + 3 * j)
            ret = f"'{SR}'!{L(2 + j)}{r}"
            eq[f"{e}{r}"] = f"=1*(1+{ret})" if i == 0 else f"={e}{r - 1}*(1+{ret})"
            eq[f"{p}{r}"] = f"=MAX(1,{e}{r})" if i == 0 else f"=MAX({p}{r - 1},{e}{r})"
            eq[f"{d}{r}"] = f"={e}{r}/{p}{r}-1"
            eq[f"{e}{r}"].number_format = eq[f"{p}{r}"].number_format = "0.0000"
            eq[f"{d}{r}"].number_format = "0.00%"
    for j in range(1, 2 + 3 * n_s):
        eq.column_dimensions[L(j)].width = 14
    eq.freeze_panes = eq.cell(row=HDR + 1, column=2)
    bk._bump(eq, "Tables")
    eq_col = {c: L(2 + 3 * j) for j, c in enumerate(strat_cols)}
    dd_col = {c: L(4 + 3 * j) for j, c in enumerate(strat_cols)}

    # ---------------- Strategy Comparison (formulas + python cross-check) ----------------
    sc = bk.sheet(
        "Strategy Comparison",
        "Headline metrics: Excel formulas, plus Python-computed metrics and an agreement check",
        note="Blue-header block = Excel formulas on 'Strategy Returns'/'Equity & Drawdown'. Python block is cross-check; 'Max abs diff' should be ~0.",
    )
    heads = [
        "Strategy",
        "Observations",
        "Total return",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Max drawdown",
        "Calmar",
    ]
    py_extra = ["Sortino", "Skewness", "Excess kurtosis", "Hit rate", "Best day", "Worst day"]
    check = [
        "Python CAGR",
        "Python Volatility",
        "Python Sharpe",
        "Python Max drawdown",
        "Max abs diff",
    ]
    allh = heads + py_extra + check
    first_row = HDR + 1
    rows = []
    for c in strat_cols:
        rows.append([pretty(c)] + [None] * (len(allh) - 1))
    df_sc = pd.DataFrame(rows, columns=allh)
    for c in allh[1:]:
        df_sc[c] = np.nan
    fm = {h_: fmt_for(h_, pd.Series([0.1])) for h_ in allh[1:]}
    fm["Observations"], fm["Max abs diff"] = "#,##0", "0.0E+00"
    h, f, ll = bk.table(sc, df_sc, top=HDR, index=False, fmts=fm, as_table=False)
    for i, c in enumerate(strat_cols):
        r = f + i
        R, E, D = rng[c], eq_col[c], dd_col[c]
        sc[f"B{r}"] = f"=COUNT({R})"
        sc[f"C{r}"] = f"='{EQ}'!{E}{last}-1"
        sc[f"D{r}"] = f"='{EQ}'!{E}{last}^({PPY}/B{r})-1"
        sc[f"E{r}"] = f"=STDEV({R})*SQRT({PPY})"
        sc[f"F{r}"] = f"=(AVERAGE({R})*{PPY}-{RF})/(STDEV({R})*SQRT({PPY}))"
        sc[f"G{r}"] = f"=MIN('{EQ}'!{D}{HDR + 1}:{D}{last})"
        sc[f"H{r}"] = f'=IF(G{r}<0,D{r}/ABS(G{r}),"n/a")'
        p = wa.perf[c]
        for k, nm in enumerate(py_extra):
            sc.cell(row=r, column=9 + k, value=_clean(float(p[nm])))
        pc = 9 + len(py_extra)
        for k, nm in enumerate(("CAGR", "Volatility", "Sharpe", "Max drawdown")):
            sc.cell(row=r, column=pc + k, value=float(p[nm]))
        sc.cell(
            row=r,
            column=pc + 4,
            value=f"=MAX(ABS(D{r}-{L(pc)}{r}),ABS(E{r}-{L(pc + 1)}{r}),ABS(F{r}-{L(pc + 2)}{r}),ABS(G{r}-{L(pc + 3)}{r}))",
        )
        for j in range(2, len(allh) + 1):
            cell = sc.cell(row=r, column=j)
            cell.font = Font(name=FONT, size=10)
            cell.number_format = fm[allh[j - 1]] or "0.000"
    for j in range(1, 9):
        sc.cell(row=h, column=j).fill = PatternFill("solid", fgColor=BLUE)
    sc.freeze_panes = sc.cell(row=first_row, column=2)
    sc.conditional_formatting.add(
        f"F{f}:F{ll}",
        ColorScaleRule(
            start_type="min",
            start_color="FCA5A5",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFFFFF",
            end_type="max",
            end_color="86EFAC",
        ),
    )
    sc.conditional_formatting.add(
        f"G{f}:G{ll}",
        ColorScaleRule(start_type="min", start_color="FCA5A5", end_type="max", end_color="FFFFFF"),
    )
    sc.conditional_formatting.add(
        f"D{f}:D{ll}", DataBarRule(start_type="min", end_type="max", color="60A5FA")
    )
    SC_F, SC_L = f, ll
    sc_row = {c: f + i for i, c in enumerate(strat_cols)}
    _bar_chart(
        sc,
        "Sharpe ratio by strategy",
        "Sharpe",
        Reference(sc, min_col=1, min_row=f, max_row=ll),
        Reference(sc, min_col=6, min_row=h, max_row=ll),
        f"A{ll + 4}",
    )
    _bar_chart(
        sc,
        "CAGR by strategy",
        "CAGR",
        Reference(sc, min_col=1, min_row=f, max_row=ll),
        Reference(sc, min_col=4, min_row=h, max_row=ll),
        f"J{ll + 4}",
        pct=True,
    )
    bk._bump(sc, "Charts")
    bk._bump(sc, "Charts")

    # ---------------- Benchmark comparison ----------------
    bc = bk.sheet(
        "Benchmark Comparison",
        "Alpha, beta, tracking error, information ratio and capture ratios vs benchmark",
        note="Beta and correlation: Excel formulas (SLOPE/CORREL). Other statistics from Python; diff column compares beta.",
    )
    sc_strats = [c for c in strat_cols if c != BENCH]
    rel = wa.perf.loc[
        [
            "Beta",
            "R-squared",
            "Alpha (ann.)",
            "Tracking error",
            "Information ratio",
            "Excess CAGR vs benchmark",
            "Up capture",
            "Down capture",
        ],
        sc_strats,
    ].T
    rel.columns = [
        "Beta (Python)",
        "R-squared",
        "Alpha (ann.)",
        "Tracking error",
        "Information ratio",
        "Excess CAGR vs benchmark",
        "Up capture",
        "Down capture",
    ]
    rel.insert(0, "Correlation (Excel)", np.nan)
    rel.insert(0, "Beta (Excel SLOPE)", np.nan)
    rel["Beta diff"] = np.nan
    rel.index = [pretty(c) for c in rel.index]
    rel.index.name = "Strategy"
    h, f, ll = bk.table(
        bc,
        rel,
        top=HDR,
        fmts={
            "Beta (Excel SLOPE)": "0.0000",
            "Beta (Python)": "0.0000",
            "Correlation (Excel)": "0.000",
            "Beta diff": "0.0E+00",
            "Up capture": "0.000",
            "Down capture": "0.000",
        },
        as_table=False,
    )
    for i, c in enumerate(sc_strats):
        r = f + i
        bc[f"B{r}"] = f"=SLOPE({rng[c]},{rng[BENCH]})"
        bc[f"C{r}"] = f"=CORREL({rng[c]},{rng[BENCH]})"
        bc[f"L{r}"] = f"=ABS(B{r}-D{r})"
        bc[f"B{r}"].number_format = "0.0000"
        bc[f"C{r}"].number_format = "0.000"
        bc[f"L{r}"].number_format = "0.0E+00"
    bc.freeze_panes = bc.cell(row=f, column=2)

    # ---------------- Risk analytics ----------------
    ra = bk.sheet(
        "Risk Analytics",
        "VaR / CVaR by method, tail statistics and Monte Carlo",
        note=f"{cfg.confidence:.0%} confidence, {cfg.var_horizon}-day horizon. Losses shown as positive fractions.",
    )
    rt = wa.risk_table.copy().join(res.mc_table)
    rt.index = [pretty(c) for c in rt.index]
    rt.index.name = "Strategy"
    h, f, ll = bk.table(ra, rt, top=HDR)
    ra.freeze_panes = ra.cell(row=f, column=2)
    ra.conditional_formatting.add(
        f"B{f}:B{ll}", DataBarRule(start_type="num", start_value=0, end_type="max", color="F87171")
    )
    _bar_chart(
        ra,
        "Historical VaR vs CVaR",
        "Loss",
        Reference(ra, min_col=1, min_row=f, max_row=ll),
        Reference(ra, min_col=2, max_col=3, min_row=h, max_row=ll),
        f"A{ll + 3}",
        pct=True,
    )
    bk._bump(ra, "Charts")

    # ---------------- Drawdown analysis ----------------
    da = bk.sheet(
        "Drawdown Analysis",
        f"Top {cfg.top_drawdowns} drawdown episodes per series (peak, trough, recovery)",
    )
    top = HDR
    for c in strat_cols:
        ep = wa.episodes[c].copy()
        if len(ep):
            ep["Recovered"] = ep["Recovered"].map({True: "Yes", False: "No (ongoing)"})
            _, _, last_r = bk.table(da, ep, top=top, index=False, title=pretty(c))
            top = last_r + 4
    # ---------------- Rolling ----------------
    ro = bk.sheet(
        "Rolling Performance",
        f"Month-end snapshots of rolling {cfg.rolling_window}-day Sharpe, volatility, return and beta",
    )
    flags = period_end_flags(sr.index, "monthly")
    ro_df = pd.concat(
        {k: v.loc[flags].rename(columns=pretty) for k, v in wa.rolling.items() if k != "Beta"},
        axis=1,
    ).dropna(how="all")
    ro_df.columns = [f"{a} | {b}" for a, b in ro_df.columns]
    ro_df.index.name = "Month end"
    h, f, ll = bk.table(
        ro,
        ro_df,
        top=HDR,
        as_table=False,
        fmts={c: ("0.000" if "Sharpe" in c else "0.00%") for c in ro_df.columns},
    )
    ro.freeze_panes = ro.cell(row=f, column=2)
    k = len(strat_cols)
    _line_chart(
        ro,
        "Rolling Sharpe ratio (month-end)",
        "Sharpe",
        "Date",
        Reference(ro, min_col=1, min_row=f, max_row=ll),
        [
            (Reference(ro, min_col=2 + j, min_row=f, max_row=ll), pretty(c))
            for j, c in enumerate(strat_cols)
        ],
        f"A{ll + 3}",
    )
    bk._bump(ro, "Charts")

    # ---------------- Allocation ----------------
    al = bk.sheet(
        "Portfolio Allocation",
        "Latest walk-forward target weights and full-sample (hypothetical) optimal weights",
    )
    latest = pd.DataFrame({s: res.weights[s].iloc[-1] for s in res.weights})
    latest.columns = [pretty(c) for c in latest.columns]
    latest.index.name = "Asset"
    h, f, ll = bk.table(
        al,
        latest,
        top=HDR,
        title="Latest target weights (walk-forward, last rebalance)",
        fmts={c: "0.0%" for c in latest.columns},
    )
    al.conditional_formatting.add(
        f"B{f}:{L(1 + latest.shape[1])}{ll}",
        DataBarRule(start_type="num", start_value=0, end_type="num", end_value=1, color="60A5FA"),
    )
    _bar_chart(
        al,
        "Latest allocation by strategy",
        "Weight",
        Reference(al, min_col=1, min_row=f, max_row=ll),
        Reference(al, min_col=2, max_col=1 + latest.shape[1], min_row=h, max_row=ll),
        f"A{ll + 4}",
        pct=True,
        stacked=True,
        w=22,
    )
    bk._bump(al, "Charts")
    sw = res.static_weights.copy()
    sw.columns = [pretty(c) for c in sw.columns]
    sw.index.name = "Asset"
    bk.table(
        al,
        sw,
        top=ll + 28,
        title="Full-sample optimal weights (HYPOTHETICAL, in-sample)",
        fmts={c: "0.0%" for c in sw.columns},
    )

    # ---------------- Optimization + frontier ----------------
    op = bk.sheet(
        "Optimization Results",
        "Full-sample optimiser outputs (in-sample, hypothetical)",
        note="Expected returns are historical means: treat as illustrative, not forecasts.",
    )
    st = res.static_stats.copy()
    st.index = [pretty(c) for c in st.index]
    st.index.name = "Method"
    bk.table(op, st, top=HDR, fmts={"Effective N": "0.0", "Sharpe": "0.000"})
    wt = res.weights
    summ = pd.DataFrame(
        {
            "Rebalances": {pretty(s): len(w) for s, w in wt.items()},
            "Avg turnover per rebalance": {pretty(s): res.turnover[s].iloc[1:].mean() for s in wt},
        }
    )
    summ.index.name = "Strategy"
    bk.table(
        op,
        summ,
        top=HDR + len(st) + 4,
        title="Walk-forward rebalancing summary",
        fmts={"Rebalances": "#,##0", "Avg turnover per rebalance": "0.000"},
    )

    fr = bk.sheet(
        "Efficient Frontier",
        "Efficient frontier points and random feasible portfolios",
        note="Full-sample, bounded weights. Hypothetical.",
    )
    fdf = res.frontier.copy()
    fdf.index.name = "Point"
    h, f, ll = bk.table(
        fr,
        fdf,
        top=HDR,
        as_table=True,
        fmts={c: "0.0%" for c in fdf.columns if c not in ("Return", "Volatility", "Sharpe")}
        | {"Sharpe": "0.000", "Return": "0.00%", "Volatility": "0.00%"},
    )
    rp = res.random_pf.head(1500).copy()
    rp.index.name = "Draw"
    c0 = fdf.shape[1] + 4
    h2, f2, l2 = bk.table(
        fr,
        rp,
        top=HDR,
        left=c0,
        as_table=False,
        widths=False,
        title="Random portfolios (first 1,500)",
    )
    sch = ScatterChart()
    sch.title, sch.style, sch.width, sch.height = "Efficient frontier", 13, 22, 12
    sch.x_axis.title, sch.y_axis.title = "Volatility", "Expected return"
    sch.x_axis.number_format = sch.y_axis.number_format = "0%"
    sch.x_axis.delete = sch.y_axis.delete = False
    s1 = Series(
        Reference(fr, min_col=c0 + 2, min_row=f2, max_row=l2),
        Reference(fr, min_col=c0 + 1, min_row=f2, max_row=l2),
        title="Random portfolios",
    )
    s1.marker.symbol, s1.marker.size, s1.graphicalProperties.line.noFill = "circle", 3, True
    s2 = Series(
        Reference(fr, min_col=3, min_row=f, max_row=ll),
        Reference(fr, min_col=2, min_row=f, max_row=ll),
        title="Efficient frontier",
    )
    s2.marker.symbol, s2.smooth = "none", False
    sch.series.extend([s1, s2])
    fr.add_chart(sch, f"A{ll + 4}")
    bk._bump(fr, "Charts")

    # ---------------- Risk contributions ----------------
    rc = bk.sheet(
        "Risk Contributions",
        "Marginal and component risk contributions at latest weights (shrunk covariance)",
    )
    top = HDR
    first_tbl = None
    for s, d in res.risk_contrib.items():
        d = d.copy()
        d.index.name = "Asset"
        h, f, ll = bk.table(
            rc,
            d,
            top=top,
            title=pretty(s),
            fmts={
                "Marginal contribution": "0.0000",
                "Component contribution": "0.0000%",
                "Weight": "0.0%",
                "% of portfolio risk": "0.0%",
            },
        )
        if first_tbl is None:
            first_tbl = (h, f, ll)
        top = ll + 4
    h, f, ll = first_tbl
    _bar_chart(
        rc,
        f"% of portfolio risk: {pretty(next(iter(res.risk_contrib)))}",
        "Share of risk",
        Reference(rc, min_col=1, min_row=f, max_row=ll),
        Reference(rc, min_col=5, min_row=h, max_row=ll),
        "H4",
        pct=True,
    )
    bk._bump(rc, "Charts")

    # ---------------- Monthly/annual ----------------
    ma = bk.sheet("Monthly & Annual Returns", "Calendar returns by month and year per series")
    top = HDR
    ann = wa.annual.copy()
    ann.columns = [pretty(c) for c in ann.columns]
    ann.index.name = "Year"
    h, f, ll = bk.table(
        ma,
        ann,
        top=top,
        title="Calendar-year returns",
        fmts={c: "0.00%" for c in ann.columns} | {"Year": "0"},
    )
    top = ll + 4
    for c in strat_cols:
        mt = wa.monthly[c].copy()
        h, f, ll = bk.table(
            ma,
            mt,
            top=top,
            title=f"Monthly returns: {pretty(c)}",
            as_table=False,
            fmts={x: "0.0%" for x in mt.columns} | {"Calendar year": "0"},
        )
        ma.conditional_formatting.add(
            f"B{f}:{L(1 + mt.shape[1])}{ll}",
            ColorScaleRule(
                start_type="num",
                start_value=-0.08,
                start_color="F87171",
                mid_type="num",
                mid_value=0,
                mid_color="FFFFFF",
                end_type="num",
                end_value=0.08,
                end_color="4ADE80",
            ),
        )
        top = ll + 4

    # ---------------- Walk-forward ----------------
    wf = bk.sheet(
        "Walk-Forward Results",
        "Calendar-year out-of-sample metrics and leak-free lookback selection",
    )
    wy = res.wf_year.copy()
    wy["Strategy"] = wy["Strategy"].map(pretty)
    h, f, ll = bk.table(
        wf,
        wy,
        top=HDR,
        index=False,
        title="Out-of-sample performance by calendar year",
        fmts={"Year": "0"},
    )
    wf.freeze_panes = wf.cell(row=f, column=1)
    if len(res.wf_select):
        ws_ = res.wf_select.copy()
        ws_["Strategy"] = ws_["Strategy"].map(pretty)
        ws_["Selection beat average"] = ws_["Selection beat average"].map(
            {True: "Yes", False: "No"}
        )
        bk.table(
            wf,
            ws_,
            top=HDR,
            left=10,
            index=False,
            title="Walk-forward lookback selection (uses only prior data)",
            fmts={"Year": "0", "Selected lookback": "0", "Best lookback ex-post": "0"},
            widths=False,
        )

    # ---------------- Robustness ----------------
    rb_ = bk.sheet(
        "Robustness & Sensitivity",
        "Parameter sweep: lookback, costs, weight cap and rebalancing",
        note="All sweep runs start on the same date for comparability.",
    )
    if len(res.sens):
        sd = res.sens.copy()
        sd["Strategy"] = sd["Strategy"].map(pretty)
        h, f, ll = bk.table(
            rb_, sd, top=HDR, index=False, fmts={"Annual turnover": "0.00", "Value": "@"}
        )
        rb_.freeze_panes = rb_.cell(row=f, column=1)
        rb_.conditional_formatting.add(
            f"E{f}:E{ll}",
            ColorScaleRule(
                start_type="min",
                start_color="FCA5A5",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFFFFF",
                end_type="max",
                end_color="86EFAC",
            ),
        )
        pv = res.sens.assign(Cell=lambda x: x["Parameter"] + " = " + x["Value"]).pivot_table(
            index="Strategy", columns="Cell", values="Sharpe", sort=False
        )
        pv.index = [pretty(c) for c in pv.index]
        pv.index.name = "Sharpe by setting"
        h2, f2, l2 = bk.table(
            rb_,
            pv,
            top=ll + 5,
            title="Sharpe ratio matrix",
            as_table=False,
            fmts={c: "0.00" for c in pv.columns},
            widths=False,
        )
        rb_.conditional_formatting.add(
            f"B{f2}:{L(1 + pv.shape[1])}{l2}",
            ColorScaleRule(
                start_type="min",
                start_color="FCA5A5",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFFFFF",
                end_type="max",
                end_color="86EFAC",
            ),
        )
    else:
        rb_["A4"] = "Parameter sweep was disabled for this run."

    # ---------------- Regimes ----------------
    rg = bk.sheet(
        "Market Regimes",
        "Performance by trend and volatility regimes",
        note="Regime at date t is determined only from data through t-1.",
    )
    rd = res.regimes.copy()
    rd["Strategy"] = rd["Strategy"].map(pretty)
    h, f, ll = bk.table(rg, rd, top=HDR, index=False)
    rg.freeze_panes = rg.cell(row=f, column=1)
    rg.conditional_formatting.add(
        f"G{f}:G{ll}",
        ColorScaleRule(
            start_type="min",
            start_color="FCA5A5",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFFFFF",
            end_type="max",
            end_color="86EFAC",
        ),
    )

    # ---------------- Stress ----------------
    sx = bk.sheet(
        "Stress & Scenarios",
        "Historical crisis windows and hypothetical (model-based) scenarios",
        note="Hypothetical rows are model estimates, not observed outcomes. Uncovered windows are shown blank.",
    )
    sh_ = res.stress_hist.copy()
    sh_ = sh_.rename(columns={c: pretty(c) for c in strat_cols})
    sh_["Covered by backtest"] = sh_["Covered by backtest"].map({True: "Yes", False: "No"})
    h, f, ll = bk.table(
        sx,
        sh_,
        top=HDR,
        index=False,
        title="Historical stress windows (cumulative return)",
        fmts={pretty(c): "0.00%" for c in strat_cols},
    )
    hs = res.stress_hypo.copy()
    hs["Strategy"] = hs["Strategy"].map(pretty)
    bk.table(sx, hs, top=ll + 5, index=False, title="Hypothetical scenarios (model estimates)")

    # ---------------- Costs ----------------
    tc = bk.sheet(
        "Transaction Costs",
        "Gross vs net performance and turnover",
        note=f"Cost model: {cfg.cost_bps:g} bps x one-way turnover at each rebalance.",
    )
    ct = res.cost_table.copy()
    ct.index = [pretty(c) for c in ct.index]
    ct.index.name = "Strategy"
    h, f, ll = bk.table(tc, ct, top=HDR, fmts={"Annual turnover": "0.00", "Rebalances": "#,##0"})
    _bar_chart(
        tc,
        "Gross vs net CAGR",
        "CAGR",
        Reference(tc, min_col=1, min_row=f, max_row=ll),
        Reference(tc, min_col=2, max_col=3, min_row=h, max_row=ll),
        f"A{ll + 4}",
        pct=True,
    )
    bk._bump(tc, "Charts")

    # ---------------- Diagnostics ----------------
    sd_ = bk.sheet(
        "Statistical Diagnostics",
        "Normality, autocorrelation, significance of mean excess return, Sharpe confidence interval, PSR",
    )
    dg = wa.diagnostics.copy()
    dg.index = [pretty(c) for c in dg.index]
    dg.index.name = "Series"
    bk.table(sd_, dg, top=HDR)

    # ---------------- Findings ----------------
    kf = bk.sheet(
        "Key Findings",
        "Auto-generated findings (each tagged historical / model / assumption) and recommendations",
        note="Generated from the numbers in this workbook by rule-based templates.",
    )
    h, f, ll = bk.table(kf, res.findings, top=HDR, index=False, as_table=True)
    (
        kf.column_dimensions["A"].width,
        kf.column_dimensions["B"].width,
        kf.column_dimensions["C"].width,
        kf.column_dimensions["D"].width,
    ) = (
        20,
        100,
        24,
        60,
    )
    for r in range(f, ll + 1):
        for cidx in (2, 4):
            kf.cell(row=r, column=cidx).alignment = Alignment(wrap_text=True, vertical="top")
    h, f, ll = bk.table(
        kf, res.recs, top=ll + 5, index=False, title="Recommendations", as_table=False
    )
    for r in range(f, ll + 1):
        for cidx in (2, 3):
            kf.cell(row=r, column=cidx).alignment = Alignment(wrap_text=True, vertical="top")

    # ---------------- Executive dashboard content ----------------
    e = ws_exec
    names = f"'Strategy Comparison'!$A${SC_F}:$A${SC_L}"

    def col_rng(col: str) -> str:
        return f"'Strategy Comparison'!${col}${SC_F}:${col}${SC_L}"

    kpis = [
        (
            "Best Sharpe",
            f"=INDEX({names},MATCH(MAX({col_rng('F')}),{col_rng('F')},0))",
            f"=MAX({col_rng('F')})",
            "0.00",
        ),
        (
            "Highest CAGR",
            f"=INDEX({names},MATCH(MAX({col_rng('D')}),{col_rng('D')},0))",
            f"=MAX({col_rng('D')})",
            "0.0%",
        ),
        (
            "Shallowest max drawdown",
            f"=INDEX({names},MATCH(MAX({col_rng('G')}),{col_rng('G')},0))",
            f"=MAX({col_rng('G')})",
            "0.0%",
        ),
        (
            "Lowest volatility",
            f"=INDEX({names},MATCH(MIN({col_rng('E')}),{col_rng('E')},0))",
            f"=MIN({col_rng('E')})",
            "0.0%",
        ),
        (
            "Benchmark Sharpe",
            "Benchmark",
            f'=INDEX({col_rng("F")},MATCH("Benchmark",{names},0))',
            "0.00",
        ),
        (
            "Benchmark CAGR",
            "Benchmark",
            f'=INDEX({col_rng("D")},MATCH("Benchmark",{names},0))',
            "0.0%",
        ),
    ]
    for i, (lab, who, val, nf) in enumerate(kpis):
        col = 1 + (i % 3) * 3
        row = 4 + (i // 3) * 4
        e.cell(row=row, column=col, value=lab).font = Font(
            name=FONT, size=9, color="64748B", bold=True
        )
        v = e.cell(row=row + 1, column=col, value=val)
        v.font, v.number_format = Font(name=FONT, size=20, bold=True, color=NAVY), nf
        w_ = e.cell(row=row + 2, column=col, value=who)
        w_.font = Font(name=FONT, size=10, color=BLUE, bold=True)
        for rr in range(row, row + 3):
            for cc in range(col, col + 2):
                e.cell(row=rr, column=cc).fill = PatternFill("solid", fgColor=LIGHT)
    for j in range(1, 10):
        e.column_dimensions[L(j)].width = 16
    rk_df = pd.DataFrame({"Strategy": [pretty(c) for c in strat_cols]})
    rk_df["Sharpe rank"] = np.nan
    rk_df["Sharpe"] = np.nan
    rk_df["CAGR"] = np.nan
    rk_df["Max drawdown"] = np.nan
    h, f, ll = bk.table(
        e,
        rk_df,
        top=14,
        index=False,
        title="Strategy ranking (formulas)",
        as_table=False,
        fmts={"Sharpe": "0.00", "CAGR": "0.0%", "Max drawdown": "0.0%", "Sharpe rank": "0"},
    )
    for i, c in enumerate(strat_cols):
        r, src = f + i, sc_row[c]
        e[f"C{r}"] = f"='Strategy Comparison'!F{src}"
        e[f"B{r}"] = f"=RANK(C{r},$C${f}:$C${ll})"
        e[f"D{r}"] = f"='Strategy Comparison'!D{src}"
        e[f"E{r}"] = f"='Strategy Comparison'!G{src}"
        for col, nf in (("B", "0"), ("C", "0.00"), ("D", "0.0%"), ("E", "0.0%")):
            e[f"{col}{r}"].number_format = nf
    e.conditional_formatting.add(
        f"C{f}:C{ll}",
        ColorScaleRule(
            start_type="min",
            start_color="FCA5A5",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFFFFF",
            end_type="max",
            end_color="86EFAC",
        ),
    )
    cats = Reference(bk.wb[EQ], min_col=1, min_row=HDR + 1, max_row=last)
    _line_chart(
        e,
        "Cumulative performance (growth of 1.00, net of costs)",
        "Portfolio value",
        "Date",
        cats,
        [
            (Reference(bk.wb[EQ], min_col=2 + 3 * j, min_row=HDR + 1, max_row=last), pretty(c))
            for j, c in enumerate(strat_cols)
        ],
        f"A{ll + 3}",
        w=26,
        h=12,
    )
    _line_chart(
        e,
        "Drawdown from running peak",
        "Drawdown",
        "Date",
        cats,
        [
            (Reference(bk.wb[EQ], min_col=4 + 3 * j, min_row=HDR + 1, max_row=last), pretty(c))
            for j, c in enumerate(strat_cols)
        ],
        f"A{ll + 28}",
        pct=True,
        w=26,
        h=11,
    )
    bk._bump(e, "Charts")
    bk._bump(e, "Charts")
    top_f = res.findings.head(6)
    e.cell(row=14, column=7, value="Headline findings").font = Font(
        name=FONT, size=12, bold=True, color=BLUE
    )
    for i, t in enumerate(top_f["Finding"]):
        c = e.cell(row=15 + i, column=7, value=f"• {t}")
        c.font = Font(name=FONT, size=9)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        e.merge_cells(start_row=15 + i, start_column=7, end_row=15 + i, end_column=14)
        e.row_dimensions[15 + i].height = 38
    if res.meta.get("synthetic"):
        e["A3"] = "SYNTHETIC DEMO DATA: figures do not describe real markets."
        e["A3"].font = Font(name=FONT, bold=True, color="DC2626")

    # ---------------- Report catalog (last) ----------------
    cat = bk.sheet(
        "Report Catalog",
        "Index of worksheets, interactive charts and output files",
        "Report Catalog",
    )
    sheets_df = pd.DataFrame(bk.catalog)
    sheets_df.loc[len(sheets_df)] = {
        "Sheet": "Report Catalog",
        "Description": "Index of worksheets",
        "Tables": 0,
        "Charts": 0,
    }
    h, f, ll = bk.table(cat, sheets_df.iloc[:-1], top=HDR, index=False, title="Worksheets")
    charts_df = pd.DataFrame(
        [
            {"Chart": k, "Section": v.section, "Caption": v.caption}
            for k, v in REGISTRY.items()
            if is_available(k, res)
        ]
    )
    bk.table(
        cat,
        charts_df,
        top=ll + 5,
        index=False,
        title="Interactive charts (dashboard and HTML report)",
    )
    cat.column_dimensions["B"].width, cat.column_dimensions["C"].width = 60, 90

    bk.save(Path(path))
    n_tables = sum(c["Tables"] for c in bk.catalog)
    n_charts = sum(c["Charts"] for c in bk.catalog)
    return {"sheets": len(bk.wb.sheetnames), "tables": n_tables, "native_charts": n_charts}
