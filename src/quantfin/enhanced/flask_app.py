"""Flask blueprint exposing the analytics through a JSON API and an interactive Plotly dashboard.

Usage inside your existing Flask app:
    from quantfin.enhanced.flask_app import register
    register(app, output_dir="outputs/qpi")      # dashboard at /qpi/
All numbers come from the pipeline's saved Results; nothing is hardcoded.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, Flask, Response, abort, jsonify, render_template, request, send_file

from . import charts
from . import metrics as mt
from .analytics import BENCH, SHORT_OBS, WindowAnalytics
from .excel import fmt_for
from .glossary import GLOSSARY
from .results import Results

logger = logging.getLogger(__name__)

EXCEL_NAME, REPORT_NAME = "quantitative_portfolio_analytics_enhanced.xlsx", "enhanced_report.html"
WINDOWED = {
    "equity",
    "drawdown",
    "excess",
    "metric_bars",
    "risk_return",
    "annual_bars",
    "monthly_heatmap",
    "rolling_sharpe",
    "rolling_vol",
    "rolling_return",
    "rolling_beta",
    "var_bars",
    "return_dist",
    "allocation_area",
    "turnover",
    "regime_timeline",
}

_FMT_KEY = {
    "0.00%": "pct2",
    "0.0%": "pct1",
    "0.000": "num3",
    "0.0000": "num4",
    "#,##0": "int",
    "0": "int",
    "yyyy-mm-dd": "date",
    None: "text",
}


class Store:
    """Lazy loader for Results with simple window cache and reload-on-change."""

    def __init__(
        self,
        output_dir: str | Path | None,
        results: Results | None = None,
        factory: Callable[[], Results] | None = None,
    ) -> None:
        self.dir = Path(output_dir) if output_dir else None
        self._res = results
        self._factory = factory
        self._mtime = 0.0
        self._cache: dict[tuple, WindowAnalytics] = {}

    def results(self) -> Results:
        if self._res is not None and self.dir is None:
            return self._res
        pkl = self.dir / "results.pkl"  # type: ignore[union-attr]
        if not pkl.exists():
            if self._factory is not None:
                if self._res is None:
                    logger.info("%s not found; building results from project outputs", pkl)
                    self._res = self._factory()
                return self._res
            abort(
                503,
                description=f"No results found at {pkl}. "
                "Run: python scripts/run_enhanced_reporting.py first.",
            )
        mt = pkl.stat().st_mtime
        if self._res is None or mt != self._mtime:
            self._res, self._mtime = Results.load(pkl), mt
            self._cache.clear()
        return self._res

    def window(self, start: str | None, end: str | None) -> WindowAnalytics:
        res = self.results()
        key = (start, end)
        if key not in self._cache:
            if len(self._cache) > 32:
                self._cache.clear()
            try:
                self._cache[key] = res.window(start, end)
            except ValueError as exc:
                abort(400, description=str(exc))
        return self._cache[key]


def _artifact(store: Store, key: str, filename: str) -> Path:
    """Locate a generated file: prefer the served output directory (works for relative paths and moved folders)."""
    if store.dir is not None:
        p = (store.dir / filename).resolve()
        if p.is_file():
            return p
    return Path(store.results().artifacts.get(key, "")).resolve()


def _perf_table(res: Results, wa: WindowAnalytics, sel: list[str]) -> pd.DataFrame:
    t = wa.perf[[c for c in wa.perf.columns if c in sel]].T
    t.index = [charts.pretty(c) for c in t.index]
    t.index.name = "Strategy"
    return t.reset_index()


def _tables() -> dict[
    str, tuple[str, Callable[[Results, WindowAnalytics, list[str]], pd.DataFrame]]
]:
    def pr(df: pd.DataFrame, col: str = "Strategy") -> pd.DataFrame:
        d = df.copy()
        if col in d:
            d[col] = d[col].map(charts.pretty)
        return d

    def bench(res, wa, sel):
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
            [c for c in wa.perf.columns if c in sel and c != BENCH],
        ].T
        rel.index = [charts.pretty(c) for c in rel.index]
        rel.index.name = "Strategy"
        return rel.reset_index()

    def risk(res, wa, sel):
        t = wa.risk_table.loc[[c for c in wa.risk_table.index if c in sel]].join(res.mc_table)
        t.index = [charts.pretty(c) for c in t.index]
        t.index.name = "Strategy"
        return t.reset_index()

    def drawdowns(res, wa, sel):
        rows = []
        for c in wa.episodes:
            if c in sel and len(wa.episodes[c]):
                d = wa.episodes[c].copy()
                d.insert(0, "Strategy", charts.pretty(c))
                rows.append(d)
        return pd.concat(rows) if rows else pd.DataFrame()

    def weights_latest(res, wa, sel):
        t = pd.DataFrame({charts.pretty(s): w.iloc[-1] for s, w in res.weights.items() if s in sel})
        t.index.name = "Asset"
        return t.reset_index()

    def static_opt(res, wa, sel):
        t = res.static_stats.copy()
        t.index = [charts.pretty(c) for c in t.index]
        t.index.name = "Method"
        return t.reset_index()

    def cost(res, wa, sel):
        t = res.cost_table.loc[[c for c in res.cost_table.index if c in sel]].copy()
        t.index = [charts.pretty(c) for c in t.index]
        t.index.name = "Strategy"
        return t.reset_index()

    def diag(res, wa, sel):
        t = wa.diagnostics.loc[[c for c in wa.diagnostics.index if c in sel]].copy()
        t.index = [charts.pretty(c) for c in t.index]
        t.index.name = "Series"
        return t.reset_index()

    def corr(res, wa, sel):
        t = res.corr.copy()
        t.index.name = "Asset"
        return t.reset_index()

    def stress_hist(res, wa, sel):
        d = res.stress_hist.copy()
        return d.rename(columns={c: charts.pretty(c) for c in res.strategy_returns.columns})

    return {
        "perf": ("Performance summary", lambda r, w, s: _perf_table(r, w, s)),
        "benchmark": ("Benchmark-relative statistics", bench),
        "risk": ("VaR / CVaR and tail statistics", risk),
        "drawdowns": ("Largest drawdown episodes", drawdowns),
        "weights_latest": ("Latest target weights", weights_latest),
        "static_opt": ("Full-sample optimisation (hypothetical)", static_opt),
        "cost": ("Transaction-cost analysis", cost),
        "diagnostics": ("Statistical diagnostics", diag),
        "corr": ("Correlation matrix", corr),
        "sensitivity": ("Parameter sensitivity", lambda r, w, s: pr(r.sens)),
        "wf_year": ("Walk-forward: yearly out-of-sample", lambda r, w, s: pr(r.wf_year)),
        "wf_select": ("Walk-forward: lookback selection", lambda r, w, s: pr(r.wf_select)),
        "regimes": ("Regime performance", lambda r, w, s: pr(r.regimes)),
        "stress_hist": ("Historical stress windows", stress_hist),
        "stress_hypo": (
            "Hypothetical scenarios (model estimates)",
            lambda r, w, s: pr(r.stress_hypo),
        ),
        "findings": ("Key findings", lambda r, w, s: r.findings),
        "recs": ("Recommendations", lambda r, w, s: r.recs),
        "quality": ("Data quality", lambda r, w, s: r.quality),
    }


TABLES = _tables()


def _parse_args(store: Store) -> tuple[str | None, str | None, list[str]]:
    res = store.results()
    start, end = request.args.get("start") or None, request.args.get("end") or None
    for v in (start, end):
        if v:
            try:
                pd.Timestamp(v)
            except ValueError:
                abort(400, description=f"Invalid date: {v}")
    cols = list(res.strategy_returns.columns)
    sel = [s for s in (request.args.get("strategies", "").split(",")) if s]
    bad = [s for s in sel if s not in cols]
    if bad:
        abort(400, description=f"Unknown strategies: {bad}")
    return start, end, sel or cols


def _records(df: pd.DataFrame) -> dict:
    d = df.copy()
    fm = {str(c): _FMT_KEY.get(fmt_for(str(c), d[c]), "num3") for c in d.columns}
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    d.columns = [str(c) for c in d.columns]
    return {
        "columns": list(d.columns),
        "formats": fm,
        "rows": json.loads(d.to_json(orient="records")),
    }


SECTION_FINDINGS = {
    "overview": ["Performance", "Benchmark", "Downside"],
    "performance": ["Performance", "Benchmark", "Downside"],
    "rolling": ["Performance", "Diagnostics"],
    "optimization": ["Diversification", "Transaction costs", "Risk contribution"],
    "risk": ["Risk contribution", "Diagnostics", "Downside"],
    "robustness": ["Robustness", "Walk-forward"],
    "stress": ["Regimes", "Stress (historical)", "Stress (hypothetical)"],
    "lab": ["Benchmark", "Performance"],
    "findings": [],
    "about": [],
}
RANKED = {  # metric -> True if higher is better
    "Sharpe": True,
    "CAGR": True,
    "Sortino": True,
    "Calmar": True,
    "Max drawdown": True,
    "Volatility": False,
    "Information ratio": True,
}


def _unpack(obj):
    """Decode Plotly's base64 typed arrays ({'dtype','bdata'[,'shape']}) into plain nested lists."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
            if "shape" in obj:
                arr = arr.reshape([int(x) for x in str(obj["shape"]).split(",")])
            return _plain(arr.tolist())
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v) for v in obj]
    return obj


def _plain(o):
    """Recursively convert numpy/pandas scalars to JSON-safe Python (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if o is pd.NaT:
        return None
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def _parse_mix(raw: str, valid: list[str]) -> dict[str, float]:
    """Parse 'a:0.5,b:0.3' into normalised non-negative weights over valid series."""
    out: dict[str, float] = {}
    for part in filter(None, raw.split(",")):
        name, _, val = part.partition(":")
        if name not in valid:
            abort(400, description=f"Unknown series in mix: {name}")
        try:
            w = float(val)
        except ValueError:
            abort(400, description=f"Bad weight for {name}: {val!r}")
        if w < 0 or w != w:
            abort(400, description="Weights must be non-negative")
        if w > 0:
            out[name] = w
    total = sum(out.values())
    if total <= 0:
        abort(400, description="At least one positive weight is required")
    return {k: v / total for k, v in out.items()}


def create_blueprint(store: Store) -> Blueprint:
    bp = Blueprint("qpi", __name__, template_folder="templates", url_prefix="/qpi")

    @bp.get("/")
    def dashboard():
        return render_template("enhanced/dashboard.html")

    @bp.get("/api/meta")
    def meta():
        res = store.results()
        return jsonify(
            {
                "meta": {k: v for k, v in res.meta.items() if k != "config"},
                "series": [
                    {"key": c, "label": charts.pretty(c), "benchmark": c == BENCH}
                    for c in res.strategy_returns.columns
                ],
                "start": str(res.strategy_returns.index[0].date()),
                "end": str(res.strategy_returns.index[-1].date()),
                "cost_bps": res.cfg.cost_bps,
                "rf": res.cfg.risk_free_rate,
                "confidence": res.cfg.confidence,
                "charts": {
                    k: {"section": v.section, "param": v.needs_param, "windowed": k in WINDOWED}
                    for k, v in charts.REGISTRY.items()
                    if charts.is_available(k, res)
                },
                "tables": {k: v[0] for k, v in TABLES.items()},
                "presets": [
                    {"name": k, "start": a, "end": b}
                    for k, (a, b) in res.cfg.stress_windows.items()
                ],
                "downloads": {
                    "excel": _artifact(store, "excel", EXCEL_NAME).is_file(),
                    "report": _artifact(store, "report", REPORT_NAME).is_file(),
                },
                "warnings": (list(res.backtest_diagnostics) + list(res.sweep_diagnostics))[:20],
                "n_fallbacks": {
                    "backtest": len(res.backtest_diagnostics),
                    "sweep": len(res.sweep_diagnostics),
                },
            }
        )

    @bp.get("/api/kpis")
    def kpis():
        res = store.results()
        start, end, sel = _parse_args(store)
        wa = store.window(start, end)
        p = wa.perf
        strat = [c for c in sel if c != BENCH and c in p.columns] or [
            c for c in p.columns if c != BENCH
        ]
        b = p[BENCH]
        out = []

        def add(label, who, val, kind, sub=""):
            out.append(
                {"label": label, "who": charts.pretty(who), "value": val, "kind": kind, "sub": sub}
            )

        s = p.loc["Sharpe", strat].idxmax()
        add(
            "Best Sharpe ratio",
            s,
            float(p.loc["Sharpe", s]),
            "num2",
            f"Benchmark {b['Sharpe']:.2f}",
        )
        s = p.loc["CAGR", strat].idxmax()
        add("Highest CAGR", s, float(p.loc["CAGR", s]), "pct1", f"Benchmark {b['CAGR']:.1%}")
        s = p.loc["Max drawdown", strat].idxmax()
        add(
            "Shallowest max drawdown",
            s,
            float(p.loc["Max drawdown", s]),
            "pct1",
            f"Benchmark {b['Max drawdown']:.1%}",
        )
        s = p.loc["Volatility", strat].idxmin()
        add(
            "Lowest volatility",
            s,
            float(p.loc["Volatility", s]),
            "pct1",
            f"Benchmark {b['Volatility']:.1%}",
        )
        s = p.loc["Information ratio", strat].idxmax()
        add(
            "Best information ratio",
            s,
            float(p.loc["Information ratio", s]),
            "num2",
            "vs benchmark",
        )
        rt = wa.risk_table.loc[strat, "Historical CVaR"]
        s = rt.idxmin()
        add(
            f"Lowest {res.cfg.confidence:.0%} CVaR",
            s,
            float(rt.min()),
            "pct2",
            f"{res.cfg.var_horizon}-day historical",
        )
        return jsonify(
            {
                "window": [str(wa.start.date()), str(wa.end.date())],
                "n": int(len(wa.returns)),
                "short": bool(len(wa.returns) < SHORT_OBS),
                "kpis": out,
            }
        )

    @bp.get("/api/figure/<name>")
    def figure(name: str):
        if name not in charts.REGISTRY:
            abort(404, description=f"Unknown chart {name}")
        res = store.results()
        start, end, sel = _parse_args(store)
        wa = store.window(start, end) if name in WINDOWED else store.window(None, None)
        try:
            fig, cap, ins = charts.build(name, res, wa, sel, request.args.get("param"))
        except (KeyError, ValueError, IndexError) as exc:
            logger.exception("chart %s failed", name)
            abort(422, description=f"Cannot build {name} for this selection: {exc}")
        return jsonify(
            {
                "figure": _unpack(json.loads(fig.to_json())),
                "caption": cap,
                "insight": ins,
                "how_to": charts.HOW_TO_READ.get(name, ""),
                "windowed": name in WINDOWED,
            }
        )

    @bp.get("/api/table/<name>")
    def table(name: str):
        if name not in TABLES:
            abort(404, description=f"Unknown table {name}")
        res = store.results()
        start, end, sel = _parse_args(store)
        wa = store.window(start, end)
        df = TABLES[name][1](res, wa, sel)
        return jsonify({"title": TABLES[name][0], **_records(df)})

    @bp.get("/api/glossary")
    def glossary():
        return jsonify(GLOSSARY)

    @bp.get("/api/insights")
    def insights():
        res = store.results()
        section = request.args.get("section", "overview")
        if section not in SECTION_FINDINGS:
            abort(404, description=f"Unknown section {section}")
        f = res.findings
        keep = (
            list(f["Section"].isin(SECTION_FINDINGS[section]))
            if SECTION_FINDINGS[section]
            else [True] * len(f)
        )
        f = f[keep] if section != "findings" else f
        rows = [
            {"section": r.Section, "finding": r.Finding, "kind": r._3, "detail": r.Detail}
            for r in f.itertuples()
        ]
        caveats = (
            [r for r in rows if r["section"] == "Caveat"]
            if section in ("overview", "findings", "about")
            else []
        )
        rows = [r for r in rows if r["section"] != "Caveat"] if section != "findings" else rows
        return jsonify({"insights": rows, "caveats": caveats, "scope": "full sample"})

    @bp.get("/api/strategy/<name>")
    def strategy(name: str):
        res = store.results()
        start, end, _ = _parse_args(store)
        wa = store.window(start, end)
        if name not in wa.returns.columns:
            abort(404, description=f"Unknown series {name}")
        perf = wa.perf[name]
        names = [c for c in wa.perf.columns if c != BENCH]
        ranks = {}
        if name != BENCH:
            for k, higher in RANKED.items():
                ranks[k] = int(wa.perf.loc[k, names].rank(ascending=not higher)[name])
        ann = wa.annual[name]
        bench_ann = wa.annual[BENCH]
        yrs_ahead = int((ann > bench_ann).sum()) if name != BENCH else None
        bullets = []
        if name != BENCH:
            best, worst = ann.idxmax(), ann.idxmin()
            bullets += [
                f"Best calendar year {best}: {ann[best]:+.1%}; worst {worst}: {ann[worst]:+.1%}.",
                f"Beat SPY in {yrs_ahead} of {len(ann)} calendar years (first/last may be partial).",
                f"Beta {perf['Beta']:.2f}: tends to move {perf['Beta']:.2f}% for each 1% SPY move. "
                f"Captures {perf['Up capture']:.0%} of up days and {perf['Down capture']:.0%} of down days.",
            ]
            if name in res.risk_contrib:
                rc = res.risk_contrib[name]
                top = rc["% of portfolio risk"].idxmax()
                bullets.append(
                    f"Latest holdings: {top} supplies {rc.loc[top, 'Component contribution'] / rc['Component contribution'].sum():.0%} "
                    f"of risk on {rc.loc[top, 'Weight']:.0%} of capital; effective holdings "
                    f"{1 / float((rc['Weight'] ** 2).sum()):.1f} of {len(rc)}."
                )
        risk = wa.risk_table.loc[name].to_dict()
        if name in res.mc_table.index:
            risk.update(res.mc_table.loc[name].to_dict())
        return jsonify(
            _plain(
                {
                    "name": name,
                    "label": charts.pretty(name),
                    "window": [str(wa.start.date()), str(wa.end.date())],
                    "perf": perf.to_dict(),
                    "ranks": ranks,
                    "n_ranked": len(names),
                    "risk": risk,
                    "episodes": wa.episodes[name].to_dict(orient="records"),
                    "weights": res.risk_contrib[name]
                    .reset_index()
                    .rename(columns={"index": "Asset"})
                    .to_dict(orient="records")
                    if name in res.risk_contrib
                    else [],
                    "cost": res.cost_table.loc[name].to_dict()
                    if name in res.cost_table.index
                    else {},
                    "annual": ann.to_dict(),
                    "bullets": bullets,
                }
            )
        )

    @bp.get("/api/mix")
    def mix():
        res = store.results()
        start, end, _ = _parse_args(store)
        wa = store.window(start, end)
        w = _parse_mix(request.args.get("w", ""), list(wa.returns.columns))
        r = sum(
            wa.returns[k] * v for k, v in w.items()
        )  # constant-mix, rebalanced daily (assumption)
        r.name = "Blend"
        bench = wa.returns[BENCH]
        perf = mt.perf_summary(r, bench, res.cfg.risk_free_rate, res.cfg.periods_per_year)
        keep = [
            "CAGR",
            "Volatility",
            "Sharpe",
            "Sortino",
            "Max drawdown",
            "Calmar",
            "Beta",
            "Information ratio",
            "Total return",
        ]
        eq, dd = mt.equity_curve(r), mt.drawdown_series(r)
        beq = mt.equity_curve(bench)
        return jsonify(
            _plain(
                {
                    "weights": w,
                    "window": [str(wa.start.date()), str(wa.end.date())],
                    "dates": [d.strftime("%Y-%m-%d") for d in r.index],
                    "equity": eq.round(6).tolist(),
                    "benchmark": beq.round(6).tolist(),
                    "drawdown": dd.round(6).tolist(),
                    "metrics": {k: perf[k] for k in keep},
                    "note": "Constant-mix portfolio of the selected series' net returns, rebalanced daily. Illustrative; "
                    "real implementation would add trading costs for the daily rebalance.",
                }
            )
        )

    @bp.get("/download/excel")
    def dl_excel():
        store.results()
        p = _artifact(store, "excel", EXCEL_NAME)
        if not p.is_file():
            abort(404, description="Workbook not found; re-run the pipeline.")
        return send_file(p, as_attachment=True, download_name=p.name)

    @bp.get("/download/report")
    def dl_report():
        store.results()
        p = _artifact(store, "report", REPORT_NAME)
        if not p.is_file():
            abort(404, description="HTML report not found; re-run the pipeline.")
        return send_file(p, as_attachment=True, download_name=p.name)

    @bp.get("/download/csv/<name>")
    def dl_csv(name: str):
        if name not in TABLES:
            abort(404)
        res = store.results()
        start, end, sel = _parse_args(store)
        df = TABLES[name][1](res, store.window(start, end), sel)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=qpi_{name}.csv"},
        )

    @bp.errorhandler(400)
    @bp.errorhandler(404)
    @bp.errorhandler(422)
    @bp.errorhandler(503)
    def _err(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    return bp


def register(
    app: Flask,
    output_dir: str | Path | None = "outputs/qpi",
    results: Results | None = None,
    url_prefix: str = "/qpi",
    results_factory: Callable[[], Results] | None = None,
) -> Store:
    """Mount the dashboard on an existing Flask app. Pass `results` to serve in-memory results instead of a saved run."""
    store = Store(None if results is not None else output_dir, results, results_factory)
    bp = create_blueprint(store)
    app.register_blueprint(bp, url_prefix=url_prefix)
    return store
