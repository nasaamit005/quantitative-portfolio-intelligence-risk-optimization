# Enhanced Reporting Layer

`src/quantfin/enhanced/` adds reporting on top of the existing pipeline. It never re-optimises or re-backtests. It reads the files the existing scripts already wrote, so its numbers cannot contradict the project's own.

## Quick start

```powershell
python scripts/run_enhanced_reporting.py   # build workbook, HTML report, tables
python app.py                              # start the Flask dashboard
# open http://127.0.0.1:5000/qpi/
```

The address `127.0.0.1` is local to the computer running the app. The public version of the dashboard is the static GitHub Pages site linked from the main [README](../README.md).

## Data flow

| Input (existing) | Used for |
|---|---|
| `reports/walk_forward_comparison/strategy_daily_results.csv` | Net and gross returns, turnover, trading costs, weights per strategy |
| `data/processed/us_benchmarks_simple_returns.csv` (SPY) | Benchmark |
| `reports/tables/*` (Ledoit-Wolf covariance, correlation, expected returns, frontier, optimised weights) | Risk contributions, Monte Carlo, frontier chart |
| `reports/robustness_analysis/*` | Sensitivity grid, regime classification and performance |
| `reports/stress_testing/*` | Historical and hypothetical stress |

`python scripts/run_enhanced_reporting.py` writes `reports/enhanced/`: the workbook, the HTML report, 31 CSV tables and `results.pkl`. If `results.pkl` is absent, the dashboard builds its inputs on the first request.

## Dashboard API

All endpoints return JSON and are windowed by the `start`, `end` and `strategies` query parameters.

| Endpoint | Purpose |
|---|---|
| `/qpi/api/meta` | Series, charts available, tables, crisis presets, run notes |
| `/qpi/api/kpis` | Headline KPIs for the window, with a short-window flag |
| `/qpi/api/figure/<chart>` | Plotly figure, caption, data-driven takeaway, how-to-read text |
| `/qpi/api/table/<table>` | Table rows with column formats |
| `/qpi/api/strategy/<key>` | Drill-down: metrics, ranks, tail risk, holdings, drawdowns, costs, narrative |
| `/qpi/api/mix?w=a:50,b:50` | Blended portfolio: metrics and curves (weights normalised; invalid input returns 400) |
| `/qpi/api/insights?section=` | Auto-generated findings per section (full sample) |
| `/qpi/api/glossary` | Metric definitions used for tooltips and the About page |
| `/qpi/download/{excel,report,csv/<table>}` | File downloads |

## What is new

- **Excel:** 25 sheets. Strategy returns, equity, running peak and drawdown are Excel formulas. CAGR, volatility, Sharpe, max drawdown, Calmar and beta are formulas on those. A Python cross-check column must read about 0 (tested).
- **Risk:** Historical, Gaussian, Cornish-Fisher and Monte Carlo VaR and CVaR, plus component risk contributions for all strategies.
- **Diagnostics:** Jarque-Bera, Ljung-Box, t-statistic, Sharpe confidence interval and probabilistic Sharpe ratio.
- **Dashboard (`/qpi/`):** Filters recompute return-based analytics on the selected window. Other charts are marked *Full sample*.

## Documented discrepancies with the existing outputs

| Item | Existing project | Enhanced layer | Cause |
|---|---|---|---|
| Sharpe, Sortino | e.g. Maximum Sharpe 1.127 | 1.124 (about 0.4% lower) | The project compounds the daily risk-free rate, `(1+rf)^(1/252)-1`. This layer uses `rf/252`. |
| CAGR, volatility, max drawdown, beta, tracking error, information ratio | Reference values | Identical (tested to 1e-5 relative) | None |
| Hypothetical market shocks | Uniform loss for every strategy | Shown as-is; excluded from charts and rankings | Not beta-scaled in the source |
| Walk-forward lookback selection | Not computed | Chart hidden | No source data |
| Windows under 60 days | Not applicable | Allowed down to 15 days and flagged as noisy | Crisis presets such as the 24-day COVID crash |
| Portfolio blender | Not applicable | Constant mix rebalanced daily, no extra costs | Illustrative only. A real implementation would pay costs on the daily rebalance. |
| Costs | Not applicable | 7 bps derived from `gross - net` per unit turnover (5 cost + 2 slippage) | Derived, not an input |

### Which Sharpe do I see where?

| Where | Maximum Sharpe | Convention |
|---|---:|---|
| Main README results table | 1.13 | Project (compounded daily risk-free rate) |
| Interactive dashboard | 1.12 | Enhanced layer (`rf/252`) |

Both are correct for their stated convention. The ranking of strategies is identical under either.

## Limits

Regime labels, stress windows and robustness grids are those of the existing scripts. This layer only re-presents them.
