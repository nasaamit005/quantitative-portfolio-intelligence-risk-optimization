"""Static outputs: standalone interactive HTML report, CSV tables and the full export routine."""

from __future__ import annotations

import html
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from . import charts
from .excel import build_workbook

logger = logging.getLogger(__name__)

SECTIONS = [
    ("performance", "Performance and benchmark comparison"),
    ("rolling", "Rolling analytics"),
    ("optimization", "Optimisation, allocation and costs"),
    ("risk", "Risk analytics"),
    ("robustness", "Robustness, sensitivity and walk-forward"),
    ("stress", "Regimes and stress testing"),
]

CSS = """body{font-family:Inter,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:0}
header{background:#fff;border-bottom:1px solid #e2e8f0;padding:28px 40px}h1{margin:0 0 6px;font-size:26px}
main{max-width:1200px;margin:0 auto;padding:24px 40px}h2{margin-top:44px;border-bottom:2px solid #2563eb;padding-bottom:6px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:18px 0;box-shadow:0 1px 3px rgba(15,23,42,.05)}
.cap{color:#475569;font-size:13px;margin:4px 8px}.ins{color:#1d4ed8;font-size:13px;margin:2px 8px 6px;font-weight:600}
table{border-collapse:collapse;font-size:13px;width:100%}th{background:#0f172a;color:#fff;padding:6px 8px;text-align:left}
td{padding:5px 8px;border-bottom:1px solid #e2e8f0}.warn{background:#fef2f2;color:#b91c1c;padding:10px 14px;border-radius:8px;font-weight:600}
.badge{font-size:11px;background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:99px}nav a{margin-right:14px;color:#2563eb;text-decoration:none;font-size:14px}"""


def _tbl(df: pd.DataFrame, pct_cols: set[str] | None = None) -> str:
    return df.to_html(escape=True, float_format=lambda x: f"{x:,.4f}", border=0)


def build_html_report(res, path: str | Path) -> int:
    """Write a self-contained (CDN Plotly) interactive report. Returns number of charts."""
    wa = res.full
    parts: list[str] = []
    first = True
    n = 0
    nav = (
        " ".join(f'<a href="#{k}">{t}</a>' for k, t in SECTIONS)
        + '<a href="#findings">Findings</a>'
    )
    for key, title in SECTIONS:
        parts.append(f'<h2 id="{key}">{html.escape(title)}</h2>')
        for name, spec in charts.REGISTRY.items():
            if spec.section != key or not charts.is_available(name, res):
                continue
            fig, cap, ins = charts.build(name, res, wa)
            div = fig.to_html(
                full_html=False,
                include_plotlyjs="cdn" if first else False,
                config={"displaylogo": False, "responsive": True},
            )
            first = False
            n += 1
            parts.append(
                f'<div class="card">{div}<div class="cap">{html.escape(cap)}</div>'
                f"{f'<div class=ins>{html.escape(ins)}</div>' if ins else ''}</div>"
            )
        if key == "performance":
            perf = wa.perf.loc[
                [
                    "Total return",
                    "CAGR",
                    "Volatility",
                    "Sharpe",
                    "Sortino",
                    "Max drawdown",
                    "Calmar",
                    "Beta",
                    "Alpha (ann.)",
                    "Information ratio",
                ]
                if "Beta" in wa.perf.index
                else None
            ].T
            parts.append(
                f'<div class="card"><b>Performance summary (net of costs)</b>{_tbl(perf.rename(index=charts.pretty))}</div>'
            )
        if key == "stress":
            parts.append(
                f'<div class="card"><b>Historical stress windows</b>{_tbl(res.stress_hist)}</div>'
            )
    parts.append(
        '<h2 id="findings">Key findings</h2><div class="card">'
        + res.findings.to_html(index=False, border=0, escape=True)
        + "</div>"
    )
    parts.append(
        '<h2>Recommendations</h2><div class="card">'
        + res.recs.to_html(index=False, border=0, escape=True)
        + "</div>"
    )
    warn = (
        '<p class="warn">SYNTHETIC DEMO DATA: no figure describes real markets.</p>'
        if res.meta.get("synthetic")
        else ""
    )
    head = (
        f"<header><h1>Quantitative Portfolio Intelligence: analytical report</h1>"
        f"<div>Backtest {res.meta['start']} to {res.meta['end']} | {res.meta['n_obs']:,} days | {len(res.meta['assets'])} assets | "
        f"cost {res.cfg.cost_bps:g} bps | rf {res.cfg.risk_free_rate:.1%} <span class=badge>historical, not a forecast</span></div>{warn}<nav>{nav}</nav></header>"
    )
    Path(path).write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>QPI Report</title><style>{CSS}</style></head><body>{head}<main>{''.join(parts)}</main></body></html>",
        encoding="utf-8",
    )
    return n


def export_csvs(res, folder: str | Path) -> list[str]:
    """Write every analytical table to CSV for downstream use."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {
        "performance": res.full.perf.T,
        "risk": res.full.risk_table.join(res.mc_table),
        "diagnostics": res.full.diagnostics,
        "strategy_returns": res.strategy_returns,
        "annual_returns": res.full.annual,
        "correlation": res.corr,
        "covariance": res.cov,
        "static_weights": res.static_weights,
        "static_stats": res.static_stats,
        "frontier": res.frontier,
        "asset_stats": res.asset_stats,
        "sensitivity": res.sens,
        "walk_forward_yearly": res.wf_year,
        "walk_forward_selection": res.wf_select,
        "regimes": res.regimes,
        "stress_historical": res.stress_hist,
        "stress_hypothetical": res.stress_hypo,
        "transaction_costs": res.cost_table,
        "data_quality": res.quality,
        "findings": res.findings,
        "recommendations": res.recs,
    }
    for s, w in res.weights.items():
        tables[f"weights_{s}"] = w
    for s, d in res.risk_contrib.items():
        tables[f"risk_contrib_{s}"] = d
    written = []
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p)
        written.append(str(p))
    return written


def export_all(res, out_dir: str | Path | None = None) -> dict[str, object]:
    """Excel + HTML + CSV + pickle. Returns a manifest (also written to manifest.json)."""
    out = Path(out_dir or res.cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = out / "quantitative_portfolio_analytics_enhanced.xlsx"
    logger.info("Writing Excel workbook")
    xl = build_workbook(res, xlsx)
    logger.info("Writing HTML report")
    n_charts = build_html_report(res, out / "enhanced_report.html")
    shutil.copyfile(out / "enhanced_report.html", out / "index.html")  # GitHub Pages entry point
    csvs = export_csvs(res, out / "tables")
    res.artifacts = {
        "excel": str(xlsx),
        "report": str(out / "enhanced_report.html"),
        "tables_dir": str(out / "tables"),
        "pickle": str(out / "results.pkl"),
    }
    res.save(out / "results.pkl")
    manifest = {
        "excel": str(xlsx),
        **xl,
        "html_charts": n_charts,
        "csv_tables": len(csvs),
        "backtest": f"{res.meta['start']}..{res.meta['end']}",
        "synthetic": bool(res.meta.get("synthetic")),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
