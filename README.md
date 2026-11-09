# Quantitative Portfolio Intelligence & Risk Optimization Platform

[![CI](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg)](pyproject.toml)
[![Live Report](https://img.shields.io/badge/live-interactive%20report-c0392b.svg)](https://nasaamit005.github.io/quantitative-portfolio-intelligence-risk-optimization/)
[![Deploy](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/deploy-dashboard.yml/badge.svg)](https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization/actions/workflows/deploy-dashboard.yml)

**Walk-forward portfolio construction, risk analytics and automated reporting on real US equity data.**

## Explore the Dashboard

**[Open Quantitative Portfolio Intelligence Dashboard](https://nasaamit005.github.io/quantitative-portfolio-intelligence-risk-optimization/)**

Explore portfolio performance, optimization strategies, risk analytics, robustness experiments and investment insights through one public dashboard entry point.

**Author:** Amit Singh · **GitHub:** [@nasaamit005](https://github.com/nasaamit005)

> **Important research limitation:** The universe consists of 12 large US stocks (including AAPL, MSFT, NVDA and META) selected with hindsight. Every strategy and the SPY comparison therefore carries survivorship and selection bias. Absolute historical returns overstate what an investor could have known in 2016. Use these results to compare methods, not to forecast returns. This project is not investment advice.

---

## 1. Overview

The project investigates one central question:

**Which portfolio construction method holds up out of sample, after costs, across parameter choices and market regimes, and how much risk does each strategy carry?**

It covers the complete quantitative portfolio research workflow using real daily US market data from Yahoo Finance.

The platform includes:

- **Data engineering:** Data acquisition, cleaning, return construction, rolling statistics, correlation and covariance estimation using sample covariance and Ledoit-Wolf shrinkage.
- **Portfolio optimization:** Minimum volatility, maximum Sharpe ratio, efficient frontier, Hierarchical Risk Parity, risk parity and Black-Litterman.
- **Walk-forward backtesting:** 252-day estimation lookback, monthly rebalancing, transaction costs and slippage, and five portfolio strategies compared against SPY.
- **Performance analytics:** CAGR, volatility, Sharpe ratio, drawdown episodes, rolling performance and benchmark-relative statistics.
- **Robustness and stress testing:** Sensitivity experiments covering lookback windows, transaction costs, rebalancing frequencies and portfolio weight constraints, along with market regimes, historical crises and hypothetical scenarios.
- **Automated reporting:** CSV and Parquet datasets, formula-driven Excel workbooks, interactive HTML reporting and a Flask analytics application.

## 2. Key Results

**Real historical data, net of portfolio transaction costs**

Evaluation period: 2016-01-05 to 2026-11-02  
Trading observations: 2,723  
Initial capital: $100,000

Sharpe ratios follow the project's documented performance analytics convention.

| Strategy | CAGR | Volatility | Sharpe | Max Drawdown | Information Ratio | Turnover | COVID Crash |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maximum Sharpe | 30.9% | 22.7% | 1.13 | -29.0% | 0.89 | 30.1x | -24.8% |
| Equal Weight | 26.0% | 19.1% | 1.10 | -30.2% | 1.56 | 4.2x | -29.5% |
| Risk Parity | 22.3% | 16.5% | 1.07 | -29.8% | 0.95 | 4.6x | -29.3% |
| HRP | 20.5% | 15.4% | 1.03 | -29.2% | 0.55 | 11.7x | -28.8% |
| Minimum Volatility | 13.9% | 14.2% | 0.71 | -29.7% | -0.12 | 9.4x | -29.4% |
| SPY Benchmark | 14.8% | 17.7% | 0.65 | -33.7% | N/A | N/A | -33.4% |

### Key Findings

- **Maximum Sharpe achieved the highest historical return.** However, it was also the most concentrated and parameter-sensitive strategy. Its Sharpe ratio varied by 0.25 across the 15 robustness settings, compared with 0.09 for equal weight. At the latest rebalance, XOM represented 32% of portfolio risk with 26% of invested capital. Turnover reached 30.1x.
- **Equal weight remained highly competitive.** It achieved the best information ratio (1.56) and second-highest Sharpe ratio, with relatively stable results.
- **Minimum volatility reduced portfolio volatility.** Its annualized volatility was 14.2%, compared with 17.7% for SPY, although its CAGR was below the benchmark.
- **Drawdowns remained substantial across strategies.** Maximum drawdowns were approximately -29% to -30%, compared with -33.7% for SPY. Diversification methods did not eliminate the 2020 market crash.
- **Transaction costs had a measurable but relatively small effect.** At the assumed 7 basis points, the largest reported annual drag was approximately 0.20% for Maximum Sharpe. SPY is shown without trading costs, which slightly favours the benchmark.

## 3. Analytical Figures

The following figures are generated by the project's performance analytics, robustness analysis and stress-testing pipelines.

| Cumulative Performance | Drawdown Comparison |
|---|---|
| ![Cumulative performance](reports/performance_analytics/figures/cumulative_performance.png) | ![Drawdown comparison](reports/performance_analytics/figures/drawdown_comparison.png) |

| Risk-Return Comparison | Robustness Scorecard |
|---|---|
| ![Risk return comparison](reports/performance_analytics/figures/risk_return_comparison.png) | ![Strategy robustness scorecard](reports/robustness_analysis/figures/strategy_robustness_scorecard.png) |

| Historical Crisis Returns | Lookback Sensitivity |
|---|---|
| ![Historical crisis returns](reports/stress_testing/figures/historical_crisis_returns.png) | ![Lookback sensitivity](reports/robustness_analysis/figures/lookback_sensitivity.png) |

## 4. Enhanced Interactive Analytics

The enhanced reporting layer, `quantfin.enhanced`, consumes the existing project outputs. It does not independently rerun or duplicate the underlying portfolio strategies.

It provides an interactive analytics experience, a formula-driven Excel workbook and a standalone HTML report.

### Dashboard Screenshots

| Dashboard Overview | Strategy Drill-Down |
|---|---|
| ![Dashboard overview](docs/figures/dashboard_overview.png) | ![Strategy drill-down](docs/figures/dashboard_drilldown.png) |
| **Overview:** Key insights, KPIs, strategy scorecards and performance sparklines. | **Drill-down:** Strategy rankings, tail risk, holdings and risk contribution. |

| Portfolio Lab | Stress Analysis |
|---|---|
| ![Portfolio lab](docs/figures/dashboard_portfolio_lab.png) | ![Dashboard stress analysis](docs/figures/dashboard_stress.png) |
| **Portfolio Lab:** Blend strategies using interactive allocation sliders. | **Stress Analysis:** Explore market regimes, historical crises and hypothetical scenarios. |

### Interactive Features

- **Flexible filtering:** Select strategies, custom date ranges and presets such as 1Y, 3Y, 5Y, YTD and historical crisis windows. Windows as short as 15 days are supported, with warnings for periods shorter than 60 days.
- **Strategy drill-down:** Select a KPI or strategy card to inspect rankings, narrative insights, VaR and CVaR, holdings, risk contribution, drawdown episodes, monthly returns and return distributions.
- **Portfolio blending:** Combine strategies and SPY through interactive allocation sliders. The server recalculates CAGR, volatility, Sharpe ratio, drawdown and beta from the historical net returns. This is a constant-mix, daily-rebalanced illustration; transaction costs are not reapplied.
- **Market shock simulation:** Adjust the assumed SPY market shock from -50% to +30% and inspect beta-implied strategy movements. These are labelled model estimates.
- **Metric explorer:** Visualize any two performance metrics with an additional measure represented by bubble size.
- **Historical replay:** Animate the cumulative growth of strategies over time.
- **Interactive explanations:** Charts include plain-English interpretation notes. Thirty-four glossary terms provide contextual explanations through tooltips.
- **Data exploration:** Sortable and filterable tables, heatmap toggles, CSV exports, full-screen charts and PNG downloads.
- **Report downloads:** Download the generated Excel workbook and standalone HTML report.

### Enhanced Risk and Reporting Features

The enhanced reporting layer also includes:

- A formula-driven Excel workbook with CAGR, volatility, Sharpe ratio, drawdown, Calmar ratio and beta formulas.
- Excel tables, conditional formatting and native charts.
- Excel-versus-Python metric agreement checks.
- Four VaR and CVaR approaches: historical, Gaussian, Cornish-Fisher and Monte Carlo.
- Component risk contributions for every strategy.
- Probabilistic Sharpe ratio and normality/autocorrelation diagnostics.
- Automated findings labelled as historical observations, model estimates or forward-looking assumptions.

For the reporting design, data mapping and documented discrepancies, see [Enhanced Reporting Documentation](docs/ENHANCED_REPORTING.md).

## 5. Validation and Testing

The project includes **595 automated tests**, covering the core quantitative pipeline and enhanced reporting functionality.

| Validation Area | Checks |
|---|---|
| Core pipeline | Optimizers, constraints, estimators, walk-forward backtesting, returns, risk and portfolio strategies |
| Metric definitions | Known-answer tests for CAGR, drawdown, Sharpe, alpha/beta, VaR methods and risk-contribution identities |
| Reporting consistency | Reporting-layer CAGR, volatility, drawdown, beta, tracking error and information ratio compared with `performance_summary.csv` |
| Input integrity | Strategy returns compared with `strategy_daily_results.csv`; benchmark date alignment; portfolio weights sum to one and respect constraints |
| Excel integrity | LibreOffice recalculation, formula-error checks and Excel-versus-Python agreement to 1e-9 |
| Dashboard | Chart and table routes, date filters, drill-down, portfolio blending, error handling and downloads |
| Regression protection | Existing application pages and original quantitative pipeline functionality |

Run the test suite with:

```powershell
pytest
```

Run lint checks with:

```powershell
ruff check .
```

## 6. Installation

### Prerequisites

- Python 3.12 or 3.14
- Git
- Windows, Linux or macOS
- Internet access for downloading market data

### Clone the Repository

```powershell
git clone https://github.com/nasaamit005/quantitative-portfolio-intelligence-risk-optimization.git

cd quantitative-portfolio-intelligence-risk-optimization
```

### Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m pip install -e .
```

### Run Tests

```powershell
pytest
```

## 7. Usage

The following commands run the project's individual analytical workflows.

### Portfolio Backtesting

```powershell
python scripts/run_real_walk_forward_comparison.py
```

### Performance Analytics

```powershell
python scripts/run_performance_analytics.py
```

### Robustness Analysis

```powershell
python scripts/run_robustness_analysis.py
```

### Portfolio Stress Testing

```powershell
python scripts/run_portfolio_stress_testing.py
```

### Automated Excel Reporting

```powershell
python scripts/run_automated_reporting.py
```

This generates the automated portfolio analytics workbook and supporting reporting outputs.

### Enhanced Reporting

```powershell
python scripts/run_enhanced_reporting.py
```

This generates the enhanced formula-driven workbook, HTML report and dashboard inputs.

### Launch the Flask Application

```powershell
python app.py
```

The Flask terminal displays the local server address when the application starts.

## 8. Repository Structure

```text
quantitative-portfolio-intelligence-risk-optimization/
│
├── app.py
│
├── src/
│   └── quantfin/
│       ├── data/
│       ├── features/
│       ├── estimators/
│       ├── optimization/
│       ├── strategies/
│       ├── backtesting/
│       ├── risk/
│       ├── analytics/
│       ├── visualization/
│       ├── services/
│       ├── utils/
│       └── enhanced/
│
├── scripts/
│   ├── run_real_walk_forward_comparison.py
│   ├── run_performance_analytics.py
│   ├── run_robustness_analysis.py
│   ├── run_portfolio_stress_testing.py
│   ├── run_automated_reporting.py
│   └── run_enhanced_reporting.py
│
├── config/
│   ├── settings.yaml
│   ├── assets.yaml
│   └── logging.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   ├── automated_reporting/
│   ├── backtests/
│   ├── excel/
│   ├── enhanced/
│   ├── figures/
│   ├── performance_analytics/
│   ├── robustness_analysis/
│   ├── stress_testing/
│   ├── tables/
│   └── walk_forward_comparison/
│
├── docs/
│   ├── ENHANCED_REPORTING.md
│   └── figures/
│
├── tests/
│   └── unit/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy-dashboard.yml
│
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```

The raw market data directory is excluded from Git. The processed datasets and generated reports provide the reproducible analytical inputs and outputs included in the repository.

## 9. Research Limitations

The results should be interpreted in light of the following limitations:

- **Hindsight universe:** The 12-stock universe was selected with knowledge of the historical period. This introduces survivorship and selection bias and favours successful growth companies.
- **In-sample strategy design:** Strategy choices, lookback windows and weight constraints were developed after observing the dataset. Sensitivity analysis measures parameter dependence but does not eliminate selection bias.
- **Expected-return estimation:** Historical average returns have substantial estimation error. Maximum Sharpe optimization is particularly exposed to this uncertainty.
- **Hypothetical stress scenarios:** Uniform market shocks apply the same assumed loss across strategies and therefore do not independently rank their resilience. Asset and sector scenarios are linear model estimates.
- **Market regime classification:** Regimes use trailing-window classifications. Days labelled bear can include strong subsequent returns because the classification can lag market rebounds. Interpret these as historical conditions following declines, not forecasts.
- **Transaction costs:** Costs use fixed basis points on turnover. Market impact, taxes, financing costs and short-selling costs are not modelled. SPY is shown without trading costs.
- **Parameter selection:** Robustness experiments evaluate selected parameter ranges; they do not establish that those ranges are optimal.
- **Not covered:** A dedicated out-of-sample lookback-selection analysis is not currently computed by the project.

## 10. Data Sources and Licensing

Market data are obtained from Yahoo Finance using `yfinance`.

The raw market data are excluded from Git. Processed returns and analytical outputs are committed where appropriate to support reproducibility.

Check Yahoo Finance's terms and applicable data licensing conditions before redistributing raw market data.

## 11. Roadmap

- [ ] Introduce a point-in-time or broader historical index universe to reduce hindsight and survivorship bias.
- [ ] Add a dedicated out-of-sample holdout and walk-forward parameter-selection framework.
- [ ] Extend factor-based and regime-conditional stress scenarios.
- [ ] Add Docker-based deployment.
- [ ] Deploy the full Flask application to a public hosting platform.
- [ ] Expand portfolio constraints and institutional risk reporting.

## 12. How to Cite

See [CITATION.cff](CITATION.cff). GitHub provides a **Cite this repository** option when the citation file is available.

## 13. License and Disclaimer

This project is licensed under the [MIT License](LICENSE).

It is educational and quantitative-research software. It does not constitute investment advice, professional risk-management advice or regulatory guidance.

Historical performance does not guarantee future results.

---

## 14. Run the Full Flask Dashboard Locally

The repository also includes a Flask-powered application that serves the interactive analytics interface and the project's additional application pages.

### Start the Application

Open PowerShell in the project root and execute:

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait until the terminal confirms that the server is running:

```text
* Running on http://127.0.0.1:5000
```

Keep the terminal open while using the application.

### Open the Local Interactive Dashboard

**[Launch Local Interactive Analytics Dashboard](http://127.0.0.1:5000/qpi/)**

> **Local access only:** The localhost address works on the computer running Flask. It is not a public website URL and will not work for recruiters or other visitors on their own devices.

### Public Dashboard vs. Local Flask Application

- **Public dashboard:** The GitHub Pages URL at the top of this README publishes the static analytics report and is accessible to visitors.
- **Local Flask application:** The Flask server provides the locally hosted interactive application and requires Python to be running on your computer.

The public GitHub Pages report and the local Flask application are separate deployments. A public deployment of the full Flask application is planned as a future enhancement.

---

**Thank you for exploring the Quantitative Portfolio Intelligence & Risk Optimization Platform.**