# Enhanced reporting layer

`src/quantfin/enhanced/` adds reporting on top of the existing pipeline. It never re-optimises or re-backtests: it reads
the files the existing scripts already wrote, so its numbers cannot contradict the project's own.

## Data flow

| Input (existing) | Used for |
|---|---|
| `reports/walk_forward_comparison/strategy_daily_results.csv` | net/gross returns, turnover, trading costs, weights per strategy |
| `data/processed/us_benchmarks_simple_returns.csv` (SPY) | benchmark |
| `reports/tables/*` (Ledoit-Wolf covariance, correlation, expected returns, frontier, optimised weights) | risk contributions, Monte Carlo, frontier chart |
| `reports/robustness_analysis/*` | sensitivity grid, regime classification and performance |
| `reports/stress_testing/*` | historical and hypothetical stress |

`python scripts/run_enhanced_reporting.py` writes `reports/enhanced/`: workbook, HTML report, 31 CSV tables, `results.pkl`.
If `results.pkl` is absent the dashboard builds its inputs on first request.

## Dashboard API (all JSON, windowed by `start`, `end`, `strategies`)
| Endpoint | Purpose |
|---|---|
| `/qpi/api/meta` | series, charts available, tables, crisis presets, run notes |
| `/qpi/api/kpis` | headline KPIs for the window, with a short-window flag |
| `/qpi/api/figure/<chart>` | Plotly figure, caption, data-driven takeaway, how-to-read text |
| `/qpi/api/table/<table>` | table rows with column formats |
| `/qpi/api/strategy/<key>` | drill-down: metrics, ranks, tail risk, holdings, drawdowns, costs, narrative |
| `/qpi/api/mix?w=a:50,b:50` | blended portfolio: metrics and curves (weights normalised; invalid input returns 400) |
| `/qpi/api/insights?section=` | auto-generated findings per section (full sample) |
| `/qpi/api/glossary` | metric definitions used for tooltips and the About page |
| `/qpi/download/{excel,report,csv/<table>}` | files |

## What is new
- **Excel**: 25 sheets. Strategy returns, equity, running peak and drawdown are Excel formulas; CAGR, volatility, Sharpe,
  max drawdown, Calmar and beta are formulas on those; a Python cross-check column must read ~0 (tested).
- **Risk**: historical, Gaussian, Cornish-Fisher and Monte Carlo VaR/CVaR; component risk contributions for all strategies.
- **Diagnostics**: Jarque-Bera, Ljung-Box, t-stat, Sharpe confidence interval, probabilistic Sharpe ratio.
- **Dashboard** (`/qpi/`): filters recompute return-based analytics on the selected window; other charts are marked *Full sample*.

## Documented discrepancies with the existing outputs
| Item | Existing project | Enhanced layer | Cause |
|---|---|---|---|
| Sharpe, Sortino | e.g. Max Sharpe 1.127 | 1.124 (about 0.4% lower) | project compounds the daily risk-free rate, `(1+rf)^(1/252)-1`; this layer uses `rf/252` |
| CAGR, volatility, max drawdown, beta, tracking error, information ratio | | identical (tested to 1e-5 relative) | |
| Hypothetical market shocks | uniform loss for every strategy | shown as-is; excluded from charts and rankings | not beta-scaled in the source |
| Walk-forward lookback selection | not computed | chart hidden | no source data |
| Windows under 60 days | | allowed down to 15 days, flagged as noisy | crisis presets such as the 24-day COVID crash |
| Portfolio blender | | constant mix rebalanced daily, no extra costs | illustrative; real implementation would pay costs on the daily rebalance |
| Costs | | 7 bps derived from `gross - net` per unit turnover (5 cost + 2 slippage) | |

## Limits
Regime labels, stress windows and robustness grids are those of the existing scripts; this layer only re-presents them.
