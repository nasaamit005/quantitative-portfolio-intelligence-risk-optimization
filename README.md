# Quantitative Portfolio Intelligence & Risk Optimization Platform

[![CI](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg)](pyproject.toml)
[![Live report](https://img.shields.io/badge/live-interactive%20report-c0392b.svg)](https://nasaamit005.github.io/quantitative-portfolio-intelligence-risk-optimization/)
[![Deploy](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/deploy-dashboard.yml/badge.svg)](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/deploy-dashboard.yml)

**Walk-forward portfolio construction, risk analytics and automated reporting on real US equity data.**

[**Live interactive report**](https://nasaamit005.github.io/quantitative-portfolio-intelligence-risk-optimization/)  |  [Enhanced reporting design](docs/ENHANCED_REPORTING.md)  |  [Results](reports/)

**Author:** Amit Singh  ·  **GitHub:** [@nasaamit005](https://github.com/nasaamit005)

> **Read this first.** The universe is 12 large US stocks (AAPL, MSFT, NVDA, META and others) that were chosen with hindsight. Every strategy and the SPY comparison therefore carry **survivorship and selection bias**, and absolute returns overstate what an investor could have known in 2016. Use the results to compare *methods*, not to forecast returns. Nothing here is investment advice.

* * *

## 1. Overview

The project asks one question: **which portfolio construction method holds up out of sample, after costs, across parameter choices and market regimes, and how much risk does each carry?**

It covers the full chain on real daily data (yfinance, 2016 to 2026):

- **Data engineering:** download, cleaning, return construction, rolling statistics, correlation and covariance (sample and Ledoit-Wolf).
- **Optimisers:** minimum volatility, maximum Sharpe, efficient frontier, Hierarchical Risk Parity, risk parity, Black-Litterman.
- **Walk-forward backtest:** 252-day lookback, monthly rebalancing, transaction cost plus slippage (7 bps per unit turnover), five strategies against SPY.
- **Analytics:** performance, drawdown episodes, rolling metrics, benchmark-relative statistics.
- **Robustness and stress:** 80 sensitivity experiments (lookback, costs, rebalancing, weight cap), market regimes, historical crises and hypothetical scenarios.
- **Reporting:** CSV/Parquet tables, a 25-sheet formula-driven Excel workbook, an interactive HTML report and a Flask dashboard.

## 2. Key results (real data, net of costs)

2016-01-05 to 2026-11-02 (2,723 trading days), $100,000 start. Sharpe uses the project's own convention (`reports/performance_analytics`).

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Info. ratio | Turnover | COVID crash |
|---|---|---|---|---|---|---|---|
| Maximum Sharpe | 30.9% | 22.7% | 1.13 | -29.0% | 0.89 | 30.1x | -24.8% |
| Equal weight | 26.0% | 19.1% | 1.10 | -30.2% | 1.56 | 4.2x | -29.5% |
| Risk parity | 22.3% | 16.5% | 1.07 | -29.8% | 0.95 | 4.6x | -29.3% |
| HRP | 20.5% | 15.4% | 1.03 | -29.2% | 0.55 | 11.7x | -28.8% |
| Minimum volatility | 13.9% | 14.2% | 0.71 | -29.7% | -0.12 | 9.4x | -29.4% |
| SPY (benchmark) | 14.8% | 17.7% | 0.65 | -33.7% | n/a | n/a | -33.4% |

- **Maximum Sharpe** earned the highest return, but it is the most concentrated and the most parameter-sensitive: its Sharpe ranges 0.25 across the 15 robustness settings versus 0.09 for equal weight. At the latest rebalance XOM supplies 32% of its risk on 26% of its capital, and turnover is 30x versus 4.2x.
- **Equal weight is hard to beat.** It has the best information ratio (1.56) and the second-highest Sharpe, with the most stable results.
- **Minimum volatility** delivers the lowest volatility (14.2% vs SPY 17.7%) but a CAGR below SPY (13.9% vs 14.8%).
- **Drawdowns were similar for every strategy** (about -29% to -30%, SPY -33.7%): in this concentrated, equity-only universe, diversification methods did not avoid the 2020 crash (-29% to -25% vs SPY -33.4%).
- **Costs are small** at 7 bps: the largest drag is 0.20% per year (Maximum Sharpe). SPY is shown without trading costs, which slightly favours it.

## 3. Figures

| | |
|---|---|
| ![Cumulative performance](reports/performance_analytics/figures/cumulative_performance.png) | ![Drawdowns](reports/performance_analytics/figures/drawdown_comparison.png) |
| ![Risk-return](reports/performance_analytics/figures/risk_return_comparison.png) | ![Robustness scorecard](reports/robustness_analysis/figures/strategy_robustness_scorecard.png) |
| ![Historical crises](reports/stress_testing/figures/historical_crisis_returns.png) | ![Lookback sensitivity](reports/robustness_analysis/figures/lookback_sensitivity.png) |

## 4. Interactive analytics dashboard (`/qpi/`)

A thin layer, `quantfin.enhanced`, reads the outputs above (it does **not** re-run or duplicate any strategy) and serves an interactive dashboard, a formula-driven Excel workbook and a standalone HTML report. Screenshots below are from the real data.

| | |
|---|---|
| ![Overview](docs/figures/dashboard_overview.png) | ![Strategy drill-down](docs/figures/dashboard_drilldown.png) |
| **Overview:** key insights, KPIs, strategy scorecards with sparklines | **Drill-down drawer:** ranks, tail risk (4 VaR methods), holdings and risk contribution |
| ![Portfolio lab](docs/figures/dashboard_portfolio_lab.png) | ![Stress](docs/figures/dashboard_stress.png) |
| **Portfolio lab:** blend strategies with sliders | **Stress:** regimes, crises and hypothetical scenarios |

**What you can do**

- **Filter anything:** toggle strategies, pick any date range or a preset (1Y, 3Y, 5Y, YTD, and your crisis windows such as COVID-19 Crash). Windows as short as 15 days work, with a visible warning below 60 days.
- **Drill down:** click a KPI or strategy card for a side panel with rank per metric, narrative bullets, VaR/CVaR by four methods, holdings versus risk contribution, drawdown episodes, monthly heatmap and return distribution.
- **Blend portfolios:** the Portfolio lab mixes strategies and SPY with sliders and recomputes CAGR, volatility, Sharpe, drawdown and beta on the server from the real net returns (constant mix, rebalanced daily, costs not re-applied).
- **Simulate a market shock:** drag SPY from -50% to +30% and see each strategy's beta-implied move (a labelled model estimate).
- **Explore any two metrics:** the metric explorer plots any pair of measures with a third as bubble size.
- **Replay history:** animate the growth of every strategy over time.
- **Learn as you go:** each chart has a plain-English "How to read" note, 34 glossary terms appear as hover tooltips on every metric and column header, and an About page documents data lineage and limitations.
- **Work with the data:** sortable, filterable tables with a heatmap toggle and CSV export; expand any chart full-screen; download any chart as PNG; download the Excel workbook or HTML report.

**Also included**

- a **formula-driven Excel workbook** (Excel formulas for CAGR, volatility, Sharpe, drawdown, Calmar, beta; tables, conditional formatting, native charts; an Excel-versus-Python agreement column),
- four **VaR/CVaR** methods (historical, Gaussian, Cornish-Fisher, Monte Carlo), component **risk contributions** for every strategy, probabilistic Sharpe and normality/autocorrelation diagnostics,
- auto-generated **findings** tagged *historical observation*, *model estimate* or *forward-looking assumption*.

Design, data mapping and documented discrepancies: [`docs/ENHANCED_REPORTING.md`](docs/ENHANCED_REPORTING.md).

## 5. Validation

595 automated tests (`pytest`), including:

| Claim | Check |
|---|---|
| Existing pipeline | optimisers, constraints, estimators, walk-forward, returns, risk, strategies (original suite) |
| Metric definitions | known-answer tests for CAGR, drawdown, Sharpe, alpha/beta, VaR methods, risk-contribution identity |
| Agreement with the project | the reporting layer's CAGR, volatility, drawdown, beta, tracking error and information ratio equal `performance_summary.csv` (Sharpe/Sortino within 1%, see docs) |
| Inputs unchanged | strategy returns equal `strategy_daily_results.csv`; benchmark dates aligned; weights sum to 1 and respect the cap |
| Excel integrity | LibreOffice recalculation: zero formula errors; Excel equals Python to 1e-9 |
| Dashboard | every chart/table route, date filters (including a 24-day crisis window), drill-down, blender checked against direct computation, error codes, downloads, original pages still served |

## 6. Installation

```bash
git clone https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization.git
cd quantitative-portfolio-intelligence-risk-optimization
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

## 7. Usage

```bash
python scripts/run_real_walk_forward_comparison.py   # strategies, walk-forward results
python scripts/run_performance_analytics.py
python scripts/run_robustness_analysis.py
python scripts/run_portfolio_stress_testing.py
python scripts/run_automated_reporting.py            # 28-sheet workbook of all tables
python scripts/run_enhanced_reporting.py             # formula-driven workbook + HTML report + dashboard inputs
python app.py                                        # the URL printed by Flask, plus /qpi/
```

## 8. Repository structure

```
app.py                      Flask app (original pages + /qpi/)
src/quantfin/               data, estimators, optimization, backtesting, strategies, analytics, services
src/quantfin/enhanced/      reporting layer: adapter, metrics, risk, charts, excel, insights, flask blueprint
scripts/                    pipeline and validation scripts
config/                     settings.yaml, assets.yaml, logging.yaml
data/                       raw (git-ignored), processed
reports/                    tables, figures, workbooks; reports/enhanced/ = enhanced outputs, published to
                            GitHub Pages by .github/workflows/deploy-dashboard.yml (index.html)
docs/                       ENHANCED_REPORTING.md, figures/ (dashboard screenshots)
tests/unit/                 595 tests
```

## 9. Limitations

- **Hindsight universe** (see top of this file) biases every result upward and favours growth names.
- **In-sample strategy design.** The strategy set, lookbacks and weight cap were chosen after seeing data. The sensitivity grid quantifies but does not remove this.
- **Expected returns** are historical means with large estimation error; Maximum Sharpe is exposed to them, the other methods mostly are not.
- **Hypothetical stress:** uniform market shocks apply the same loss to every strategy, so they do not rank strategies; asset/sector scenarios are linear model estimates.
- **Regimes** use the project's own trailing-window classification. Days labelled "bear" show high subsequent returns because the labels lag rebounds (for example mid-2020), so read them as "after a decline", not as forecasts.
- **Costs** are flat basis points on turnover. No market impact, taxes, financing or shorting. SPY carries no cost.
- **Not covered here:** the lookback walk-forward selection analysis (hidden from the dashboard because the project does not compute it).

## 10. Data and licensing

Market data come from Yahoo Finance via `yfinance`; check Yahoo's terms before redistributing the raw series. Raw data is git-ignored; processed returns are committed for reproducibility.

## 11. Roadmap

- [ ] Point-in-time or broader index universe to remove hindsight bias
- [ ] Out-of-sample holdout and walk-forward parameter selection
- [ ] Factor and regime-conditional stress scenarios
- [ ] Docker image and hosted dashboard

## 12. How to cite

See [`CITATION.cff`](CITATION.cff); GitHub shows a "Cite this repository" button.

## 13. License and disclaimer

MIT License ([LICENSE](LICENSE)). Educational and quantitative-research software, not investment, risk-management or regulatory advice. Historical performance does not guarantee future returns.

## 🖥️ Local Interactive Analytics Dashboard (Flask)

This project includes a Flask-powered interactive analytics dashboard for exploring portfolio optimization, performance, risk, and robustness results.

### Launch locally

From the project root, run:

```powershell
.\.venv\Scripts\python.exe app.py
```

### Open the dashboard

**[Open Interactive Analytics Dashboard](http://127.0.0.1:5000/qpi/)**

The Flask server must be running on your computer for this link to work.

### Other application routes

- Main Dashboard: http://127.0.0.1:5000/
- Optimization: http://127.0.0.1:5000/optimization
- Performance: http://127.0.0.1:5000/performance
- Risk: http://127.0.0.1:5000/risk
- Robustness: http://127.0.0.1:5000/robustness
- Methodology: http://127.0.0.1:5000/methodology
- Health: http://127.0.0.1:5000/health

### Public dashboard

**[Visit the publicly hosted GitHub Pages Dashboard](https://nasaamit005.github.io/quantitative-portfolio-intelligence-risk-optimization/)**

GitHub Pages provides public access to the static analytics report, while Flask provides the locally running interactive application.