"""Integration tests: reporting layer built from the project's own saved outputs (reports/ and data/)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flask import Flask
from openpyxl import load_workbook

from quantfin.enhanced import charts
from quantfin.enhanced.adapter import build_results
from quantfin.enhanced.flask_app import TABLES, register
from quantfin.enhanced.report import export_all

ROOT = Path(__file__).resolve().parents[2]
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")
needs_outputs = pytest.mark.skipif(
    not (ROOT / "reports/walk_forward_comparison/strategy_daily_results.csv").exists(),
    reason="project reports not generated",
)
pytestmark = needs_outputs


@pytest.fixture(scope="module")
def res():
    return build_results(ROOT, mc_sims=4000)


@pytest.fixture(scope="module")
def exported(res, tmp_path_factory):
    out = tmp_path_factory.mktemp("enh")
    return out, export_all(res, out)


def test_strategy_returns_equal_source_files(res):
    src = pd.read_csv(
        ROOT / "reports/walk_forward_comparison/strategy_daily_results.csv", parse_dates=["date"]
    )
    for s in res.strategies:
        ref = src[src["strategy"] == s].set_index("date")["net_daily_return"]
        assert np.allclose(res.strategy_returns[s].to_numpy(), ref.to_numpy(), atol=1e-15)


def test_benchmark_aligned_and_complete(res):
    sr = res.strategy_returns
    assert not sr.isna().any().any()
    spy = pd.read_csv(
        ROOT / "data/processed/us_benchmarks_simple_returns.csv", index_col=0, parse_dates=True
    )["SPY"]
    assert np.allclose(sr["Benchmark"].to_numpy(), spy.reindex(sr.index).to_numpy())


def test_metrics_agree_with_project_performance_summary(res):
    theirs = pd.read_csv(ROOT / "reports/performance_analytics/performance_summary.csv").set_index(
        "strategy"
    )
    mine = res.full.perf
    pairs = {
        "CAGR": "annualized_return",
        "Volatility": "annualized_volatility",
        "Max drawdown": "max_drawdown",
        "Beta": "beta_vs_spy",
        "Tracking error": "tracking_error",
        "Information ratio": "information_ratio",
    }
    for s in res.strategies:
        for a, b in pairs.items():
            assert mine.loc[a, s] == pytest.approx(theirs.loc[s, b], rel=1e-5, abs=1e-7), (s, a)
        # Sharpe/Sortino differ only through the risk-free convention (rf/252 vs compounded daily rf)
        assert mine.loc["Sharpe", s] == pytest.approx(theirs.loc[s, "sharpe_ratio"], rel=0.01)
        assert mine.loc["Sortino", s] == pytest.approx(theirs.loc[s, "sortino_ratio"], rel=0.01)


def test_costs_consistent_with_trading_cost_column(res):
    src = pd.read_csv(ROOT / "reports/walk_forward_comparison/strategy_daily_results.csv")
    ct = res.cost_table
    assert (ct["Gross CAGR"] >= ct["Net CAGR"] - 1e-12).all()
    for s in res.strategies:
        d = src[src["strategy"] == s]
        assert ct.loc[s, "Annual turnover"] == pytest.approx(d["turnover"].sum() / (len(d) / 252))
        assert ct.loc[s, "Rebalances"] == (d["turnover"] > 0).sum()


def test_weights_sum_to_one_and_respect_cap(res):
    for s, w in res.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6), s
        assert (w.min(axis=1) >= -1e-9).all()
        # first row is the purchase day and already carries one day of drift; every later rebalance is exact
        assert (w.iloc[1:].max(axis=1) <= res.cfg.max_weight + 1e-6).all(), s
        assert w.iloc[0].max() <= res.cfg.max_weight + 2e-3, s


def test_risk_contributions_and_var_ordering(res):
    for rc in res.risk_contrib.values():
        assert rc["% of portfolio risk"].sum() == pytest.approx(1.0)
    rt = res.full.risk_table
    assert (rt["Historical CVaR"] >= rt["Historical VaR"]).all()


def test_stress_hist_covered_and_values_match_source(res):
    src = pd.read_csv(ROOT / "reports/stress_testing/historical_stress_results.csv")
    for row in res.stress_hist.itertuples():
        ref = src[src["historical_period"] == row.Scenario].set_index("strategy")[
            "cumulative_return"
        ]
        for s in res.strategies:
            assert getattr(row, s) == pytest.approx(ref[s])


def test_hypothetical_findings_do_not_claim_beta_for_uniform_shocks(res):
    text = " ".join(res.findings["Finding"])
    assert "beta-implied" not in text and "uniform" in text


def test_every_chart_builds_and_json_serialises(res):
    available = [n for n in charts.REGISTRY if charts.is_available(n, res)]
    assert "wf_select" not in available and len(available) >= 30
    for name in available:
        fig, cap, _ = charts.build(name, res, res.full)
        assert len(fig.data) > 0 and cap, name
        json.loads(fig.to_json())


def test_export_files(exported):
    out, man = exported
    assert (out / "quantitative_portfolio_analytics_enhanced.xlsx").stat().st_size > 50_000
    assert man["sheets"] >= 24 and man["synthetic"] is False
    assert man["html_charts"] == sum(
        charts.is_available(n, build_results(ROOT, mc_sims=500)) for n in charts.REGISTRY
    )


@pytest.mark.skipif(not SOFFICE, reason="LibreOffice not installed")
def test_excel_recalculates_without_errors_and_matches_python(exported, res, tmp_path):
    src = exported[0] / "quantitative_portfolio_analytics_enhanced.xlsx"
    subprocess.run(
        [SOFFICE, "--headless", "--convert-to", "xlsx", "--outdir", str(tmp_path), str(src)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    wb = load_workbook(tmp_path / src.name, data_only=True)
    errors = [
        (w.title, c.coordinate)
        for w in wb.worksheets
        for row in w.iter_rows()
        for c in row
        if isinstance(c.value, str) and c.value.startswith("#") and c.value[1:4].isupper()
    ]
    assert not errors, errors[:5]
    ws = wb["Strategy Comparison"]
    hdr = [c.value for c in ws[4]]
    for row in ws.iter_rows(
        min_row=5, max_row=4 + len(res.strategy_returns.columns), values_only=True
    ):
        assert dict(zip(hdr, row, strict=False))["Max abs diff"] < 1e-9


@pytest.fixture(scope="module")
def client(res, exported):
    out = exported[0]
    res.artifacts = {
        "excel": str(out / "quantitative_portfolio_analytics_enhanced.xlsx"),
        "report": str(out / "enhanced_report.html"),
    }
    app = Flask(__name__)
    register(app, results=res)
    return app.test_client()


def test_flask_charts_tables_filters_and_errors(client, res):
    assert client.get("/qpi/").status_code == 200
    shown = client.get("/qpi/api/meta").json["charts"]
    assert "wf_select" not in shown and len(shown) >= 30
    for name in shown:
        r = client.get(f"/qpi/api/figure/{name}")
        assert r.status_code == 200 and r.json["figure"]["data"], name
    for name in TABLES:
        assert client.get(f"/qpi/api/table/{name}").status_code == 200, name
    mid = str(res.strategy_returns.index[len(res.strategy_returns) // 2].date())
    assert (
        client.get("/qpi/api/kpis").json["n"] > client.get(f"/qpi/api/kpis?start={mid}").json["n"]
    )
    assert client.get("/qpi/api/kpis?start=2099-01-01").status_code == 400
    assert client.get("/qpi/api/kpis?strategies=bogus").status_code == 400
    assert client.get("/qpi/download/excel").data[:2] == b"PK"


def test_project_app_serves_both_old_and_new_pages():
    import sys

    sys.path.insert(0, str(ROOT))
    import app as project_app

    c = project_app.app.test_client()
    for url in [
        "/",
        "/optimization",
        "/performance",
        "/risk",
        "/robustness",
        "/methodology",
        "/health",
        "/qpi/",
    ]:
        assert c.get(url).status_code == 200, url


def test_dashboard_builds_results_on_demand_when_no_pickle(res, tmp_path):
    """A fresh clone has no results.pkl: the blueprint must fall back to the factory instead of failing."""
    app = Flask(__name__)
    register(app, output_dir=tmp_path / "missing", results_factory=lambda: res)
    c = app.test_client()
    assert c.get("/qpi/api/meta").status_code == 200
    assert c.get("/qpi/api/kpis").json["n"] == len(res.strategy_returns)
    plain = Flask(__name__)
    register(plain, output_dir=tmp_path / "missing")
    r = plain.test_client().get("/qpi/api/meta")
    assert r.status_code == 503 and "run_enhanced_reporting" in r.json["error"]


def test_new_interactive_endpoints(client, res):
    """Drill-down, blender, insights and glossary endpoints, including a short crisis window with open drawdowns."""
    assert len(client.get("/qpi/api/glossary").json) >= 30
    ins = client.get("/qpi/api/insights?section=overview").json
    assert ins["insights"] and all({"finding", "kind"} <= set(r) for r in ins["insights"])
    assert client.get("/qpi/api/insights?section=nope").status_code == 404
    s = res.strategies[0]
    j = client.get(f"/qpi/api/strategy/{s}").json
    assert set(j["ranks"]) >= {"Sharpe", "CAGR"} and j["perf"]["CAGR"] == pytest.approx(
        res.full.perf.loc["CAGR", s]
    )
    assert client.get("/qpi/api/strategy/nope").status_code == 404
    short = "?start=2020-02-19&end=2020-03-23"
    assert client.get(f"/qpi/api/kpis{short}").json["short"] is True
    assert (
        client.get(f"/qpi/api/strategy/{s}{short}").status_code == 200
    )  # NaT recovery dates must serialise
    for name in client.get("/qpi/api/meta").json["charts"]:
        assert client.get(f"/qpi/api/figure/{name}{short}").status_code == 200, name
    assert client.get("/qpi/api/kpis?start=2020-03-01&end=2020-03-10").status_code == 400


def test_mix_matches_direct_computation_and_validates(client, res):
    a, b = res.strategies[0], res.strategies[1]
    j = client.get(f"/qpi/api/mix?w={a}:30,{b}:70").json
    r = 0.3 * res.strategy_returns[a] + 0.7 * res.strategy_returns[b]
    assert j["weights"][a] == pytest.approx(0.3) and j["weights"][b] == pytest.approx(0.7)
    assert j["equity"][-1] == pytest.approx((1 + r).prod())
    assert j["metrics"]["Max drawdown"] == pytest.approx(
        ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(), abs=1e-6
    )
    for bad in ("", "zzz:1", f"{a}:-1", f"{a}:x", f"{a}:0"):
        assert client.get(f"/qpi/api/mix?w={bad}").status_code == 400, bad


def test_figures_contain_no_binary_encoded_arrays(client):
    for name in client.get("/qpi/api/meta").json["charts"]:
        assert "bdata" not in client.get(f"/qpi/api/figure/{name}").get_data(as_text=True), name


def test_every_chart_has_how_to_read():
    assert not [n for n in charts.REGISTRY if n not in charts.HOW_TO_READ]
