# Monte Carlo Portfolio Risk Simulator

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-43%20passed-brightgreen)](#testing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python-based portfolio risk analysis project that simulates possible future portfolio outcomes using Monte Carlo methods and compares a multivariate-normal return model with a GARCH(1,1) volatility model combined with joint residual bootstrap sampling.

**GitHub repository:** https://github.com/nasaamit005/monte-carlo-portfolio-risk-simulator

> **Important:** Simulation results are scenario estimates, not predictions or guarantees. They depend on the selected assets, historical sample, assumptions, model specification, and random seed. This project is educational and is not investment advice.

---

## Contents

- [Project Overview](#project-overview)
- [Objectives and Business Questions](#objectives-and-business-questions)
- [Assets and Data](#assets-and-data)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Mathematical Questions, Equations, and Symbol Meanings](#mathematical-questions-equations-and-symbol-meanings)
- [Model Comparison Results](#model-comparison-results)
- [Reports and Interactive Charts](#reports-and-interactive-charts)
- [Installation and Execution](#installation-and-execution)
- [Testing](#testing)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

## Project Overview

The Monte Carlo Portfolio Risk Simulator explores a range of plausible portfolio outcomes by generating simulated return paths from historical market data. It is designed to demonstrate financial data preparation, portfolio statistics, risk measurement, volatility modelling, simulation, testing, and reporting in a reproducible Python workflow.

The project includes two simulation approaches:

1. **Multivariate Normal Monte Carlo:** estimates a joint return distribution from historical returns and simulates correlated asset returns.
2. **GARCH(1,1) + Joint Residual Bootstrap:** models time-varying volatility for each asset and resamples aligned standardized residual vectors to retain observed cross-asset residual dependence.

The GARCH forecasting validation workflow separately evaluates one-step-ahead variance forecasts on a chronological holdout sample against an expanding historical-variance baseline using QLIKE loss.

## Objectives and Business Questions

The project is intended to help investigate questions such as:

- What range of terminal portfolio values could arise under the model assumptions?
- How frequently do simulated terminal values fall below the starting portfolio value?
- What do the lower-tail outcomes suggest about downside exposure?
- How do Value at Risk (VaR) and Expected Shortfall (ES) differ between simulation approaches?
- How does allowing volatility to change over time affect the simulated distribution of outcomes?
- Does GARCH produce better out-of-sample variance forecasts than a historical-variance baseline under QLIKE?
- How can results be presented through interactive charts and reproducible reports?

These are model-based analyses. A simulation cannot establish the probability of future market outcomes without assumptions about the relationship between historical data and the future.

## Assets and Data

The analysis has been configured for the following tickers:

| Ticker | Asset |
|---|---|
| `AAPL` | Apple |
| `AMZN` | Amazon |
| `GOOGL` | Alphabet |
| `MSFT` | Microsoft |
| `NVDA` | NVIDIA |
| `^GSPC` | S&P 500 benchmark |

The five individual equities are the portfolio assets; `^GSPC` is used as a benchmark in the broader analysis. Check each script's configuration before changing the asset universe.

The project contains a combined daily market-data file at `data/raw/all_tickers_daily.csv` and processed datasets under `data/processed/`. Data coverage and availability depend on the downloaded source data. Inspect the generated data and report metadata for the exact date range used by a particular run.

### Data handling principles

- Parse dates and sort observations chronologically.
- Check for missing, non-finite, duplicated, or invalid price observations.
- Align asset returns by date before estimating cross-asset relationships.
- Use only training observations to fit forecasting-model parameters.
- Keep the holdout period separate when evaluating out-of-sample forecasts.

## Technology Stack

- **Python** for data processing, modelling, and automation.
- **pandas / NumPy** for tabular data and numerical calculations.
- **Plotly** for interactive charts.
- **`arch`** for GARCH modelling.
- **pytest** for automated tests.
- **HTML, CSV, and Markdown** for shareable reports and exported results.

The exact dependencies are listed in `requirements.txt`.

## Project Structure

```text
monte-carlo-portfolio-risk-simulator/
├── data/
│   ├── raw/
│   │   └── all_tickers_daily.csv
│   └── processed/
│       ├── daily_returns.csv
│       ├── performance_metrics.csv
│       ├── portfolio_daily_returns.csv
│       ├── prices_wide.csv
│       └── stock_correlation.csv
├── docs/
│   ├── MONTE_CARLO_METHODOLOGY.md
│   ├── MONTE_CARLO_METHODOLOGY.html
│   ├── REFERENCES.md
│   └── REFERENCES.html
├── reports/
│   ├── garch_comparison/
│   │   ├── charts/
│   │   └── ...
│   └── garch_validation/
│       ├── charts/
│       └── ...
├── src/
│   ├── monte_carlo_simulation.py
│   ├── portfolio_analysis.py
│   ├── garch_portfolio_simulation.py
│   └── validate_garch_forecasts.py
├── tests/
│   ├── test_monte_carlo.py
│   ├── test_garch_comparison.py
│   └── test_garch_validation.py
├── requirements.txt
└── README.md
```

The tree highlights core project components. Your local repository may contain additional scripts and generated artifacts.

## Methodology

### 1. Data preparation

Load daily price observations, validate the date and price fields, sort them chronologically, and align the asset series. Returns are calculated from successive observations.

### 2. Historical analysis

Estimate asset return averages, volatilities, covariance, and correlation from historical returns. These statistics describe the sample; they are not guaranteed to remain stable.

### 3. Multivariate Normal Monte Carlo

The baseline model estimates a vector of mean returns and a covariance matrix from historical data. It then generates correlated multivariate-normal log-return observations and simulates portfolio paths.

Default settings documented for the baseline workflow include:

- Simulations: **10,000**
- Horizon: **252 trading days**
- Initial portfolio value: **$10,000**
- Portfolio allocation: **equal weights**
- Random seed: **42**

Confirm the active configuration in the script or generated configuration metadata when reproducing a run.

### 4. GARCH(1,1) + joint residual bootstrap

A GARCH(1,1) model estimates conditional volatility for each asset. Standardized residual vectors are resampled jointly by date, rather than independently asset by asset. This helps retain the cross-asset dependence present in the aligned residual observations.

### 5. Out-of-sample volatility validation

The validation workflow uses a chronological split: the first 80% of observations for training and the final 20% for holdout evaluation, subject to minimum sample-size checks. It compares GARCH variance forecasts with an expanding historical-variance baseline using QLIKE.

The validation procedure fits model parameters on training data and forecasts one step ahead through the holdout. Realized squared returns are used as a proxy for latent variance. This is a noisy proxy, not a directly observed true volatility value.

### 6. Risk reporting

The reports compare terminal portfolio values and downside risk statistics. Risk measures are sensitive to the loss definition, confidence level, simulation horizon, and quantile convention. Always read the corresponding report before interpreting a number.

---

## Mathematical Questions, Equations, and Symbol Meanings

This section presents common questions relevant to portfolio analysis, Monte Carlo simulation, and GARCH validation. Each equation is followed by definitions of its symbols and an interpretation. Some equations are general financial reference formulas; whether a particular metric is calculated by a script depends on that script's implementation.

### Question 1. How is daily log return calculated?

**Equation**

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(r_t\) | Log return at time \(t\), dimensionless |
| \(P_t\) | Asset price at time \(t\) |
| \(P_{t-1}\) | Asset price at the preceding observation |
| \(\ln(\cdot)\) | Natural logarithm |
| \(t\) | Current observation or time index |

**Interpretation:** Log return measures the continuously compounded change between two positive prices. It is useful for statistical modelling and is the return representation used by the simulation workflow.

### Question 2. How is simple return calculated, and how is it related to log return?

**Equations**

\[
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}
\]

\[
r_t = \ln(1+R_t), \qquad R_t=e^{r_t}-1
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(R_t\) | Simple return over the period |
| \(r_t\) | Log return over the period |
| \(P_t\) | Current price |
| \(P_{t-1}\) | Previous price |
| \(e\) | Euler's number, the base of natural logarithms |

**Interpretation:** A simple return of \(0.05\) represents a 5% increase. The log and simple returns are related but are not interchangeable in portfolio aggregation.

### Question 3. How is the historical mean return calculated?

**Equation**

\[
\bar r = \frac{1}{T}\sum_{t=1}^{T} r_t
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\bar r\) | Sample mean return |
| \(r_t\) | Return observed at time \(t\) |
| \(T\) | Number of return observations |
| \(\sum\) | Summation over the specified observations |
| \(t\) | Observation index |

**Interpretation:** The mean summarizes the average return per observation in the selected historical sample. It is not necessarily the expected return that will occur in the future.

### Question 4. How are sample variance and standard deviation calculated?

**Equations**

\[
s^2 = \frac{1}{T-1}\sum_{t=1}^{T}(r_t-\bar r)^2
\]

\[
s = \sqrt{s^2}
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(s^2\) | Sample variance |
| \(s\) | Sample standard deviation |
| \(r_t\) | Return at observation \(t\) |
| \(\bar r\) | Sample mean return |
| \(T\) | Number of observations |
| \(T-1\) | Degrees-of-freedom adjustment for sample variance |
| \(\sqrt{\cdot}\) | Square-root operator |

**Interpretation:** Variance measures dispersion in squared-return units; standard deviation expresses dispersion in return units. Daily volatility should not be confused with annualized volatility.

### Question 5. How are covariance and correlation between two assets calculated?

**Equations**

\[
\operatorname{Cov}(r_i,r_j)
=
\frac{1}{T-1}\sum_{t=1}^{T}
(r_{i,t}-\bar r_i)(r_{j,t}-\bar r_j)
\]

\[
\rho_{ij}
=
\frac{\operatorname{Cov}(r_i,r_j)}{s_i s_j}
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\operatorname{Cov}(r_i,r_j)\) | Sample covariance between assets \(i\) and \(j\) |
| \(\rho_{ij}\) | Correlation coefficient between assets \(i\) and \(j\) |
| \(r_{i,t}\) | Return of asset \(i\) at time \(t\) |
| \(r_{j,t}\) | Return of asset \(j\) at time \(t\) |
| \(\bar r_i,\bar r_j\) | Sample mean returns for assets \(i\) and \(j\) |
| \(s_i,s_j\) | Sample standard deviations for assets \(i\) and \(j\) |
| \(i,j\) | Asset indices |
| \(T\) | Number of aligned observations |

**Interpretation:** Covariance captures how two return series move together in their original units. Correlation standardizes that relationship and lies between \(-1\) and \(+1\), when both series have nonzero variance.

### Question 6. How is a portfolio's simple return calculated from asset simple returns?

**Equation**

\[
R_{p,t} = \sum_{i=1}^{N} w_i R_{i,t}
\]

with

\[
\sum_{i=1}^{N}w_i=1
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(R_{p,t}\) | Portfolio simple return at time \(t\) |
| \(R_{i,t}\) | Simple return of asset \(i\) at time \(t\) |
| \(w_i\) | Portfolio weight assigned to asset \(i\) |
| \(N\) | Number of assets in the portfolio |
| \(\sum w_i=1\) | Fully invested portfolio-weight constraint |

**Interpretation:** For a fully invested portfolio with weights maintained over the return period, the portfolio simple return is the weighted sum of asset simple returns. The relationship must be applied carefully when the simulated inputs are log returns.

### Question 7. How is portfolio variance calculated from a covariance matrix?

**Equations**

\[
\sigma_p^2 = \mathbf{w}^{\mathsf T}\boldsymbol{\Sigma}\mathbf{w}
\]

\[
\sigma_p = \sqrt{\mathbf{w}^{\mathsf T}\boldsymbol{\Sigma}\mathbf{w}}
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\sigma_p^2\) | Portfolio variance |
| \(\sigma_p\) | Portfolio standard deviation |
| \(\mathbf{w}\) | Column vector of portfolio weights |
| \(\mathbf{w}^{\mathsf T}\) | Transpose of the weight vector |
| \(\boldsymbol{\Sigma}\) | Asset-return covariance matrix |
| \(N\) | Number of assets represented in the vector and matrix |

**Interpretation:** Portfolio risk depends on asset weights and the full covariance matrix, not only on each asset's individual volatility. Covariance is the mathematical basis for modelling cross-asset dependence in the multivariate-normal simulation.

### Question 8. What distribution does the multivariate-normal model assume for asset log returns?

**Equation**

\[
\mathbf{r}_t \sim \mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\mathbf{r}_t\) | Vector of asset log returns at time \(t\) |
| \(\sim\) | “Is distributed according to” |
| \(\mathcal{N}\) | Multivariate normal distribution |
| \(\boldsymbol{\mu}\) | Vector of estimated mean log returns |
| \(\boldsymbol{\Sigma}\) | Covariance matrix of log returns |
| \(t\) | Simulated time index |

**Interpretation:** This model represents dependence through a covariance matrix. Its standard form does not capture all features that can occur in financial returns, such as heavy tails, changing correlations, or volatility clustering.

### Question 9. How is portfolio value updated when the simulated inputs are log returns and the portfolio is rebalanced to target weights?

For a portfolio rebalanced to weights \(w_i\) at each period, the simple portfolio return implied by asset log returns is

\[
R_{p,t+1} = \sum_{i=1}^{N}w_i\left(e^{r_{i,t+1}}-1\right)
\]

and the corresponding value update is

\[
V_{t+1}
=
V_t\left[1+\sum_{i=1}^{N}w_i\left(e^{r_{i,t+1}}-1\right)\right].
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(V_t\) | Portfolio value at time \(t\) |
| \(V_{t+1}\) | Portfolio value at the next period |
| \(r_{i,t+1}\) | Simulated log return for asset \(i\) over the next period |
| \(e^{r_{i,t+1}}-1\) | Simple return corresponding to the simulated log return |
| \(w_i\) | Target portfolio weight for asset \(i\) |
| \(N\) | Number of portfolio assets |

**Interpretation:** This equation is appropriate for a daily-rebalanced portfolio with fixed target weights and no fees, taxes, cash flows, or market frictions. Verify the value-update expression in the specific simulation script before treating this as an exact description of its implementation.

### Question 10. How is compound annual growth rate (CAGR) calculated?

**Equation**

\[
\operatorname{CAGR}
=
\left(\frac{V_{\mathrm{end}}}{V_{\mathrm{start}}}\right)^{1/Y}-1
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(V_{\mathrm{end}}\) | Portfolio value at the end of the period |
| \(V_{\mathrm{start}}\) | Portfolio value at the start of the period |
| \(Y\) | Length of the period in years |
| \(\operatorname{CAGR}\) | Compound annual growth rate |

**Interpretation:** CAGR expresses the constant annual growth rate that connects the starting and ending values over the specified period. This is a reference formula; do not assume the Monte Carlo script reports CAGR unless the relevant output confirms it.

### Question 11. How is annualized volatility estimated from daily volatility?

**Equation**

\[
\sigma_{\mathrm{annual}}=\sigma_{\mathrm{daily}}\sqrt{D}
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\sigma_{\mathrm{annual}}\) | Annualized standard deviation |
| \(\sigma_{\mathrm{daily}}\) | Daily standard deviation |
| \(D\) | Assumed number of trading days per year; commonly 252 |
| \(\sqrt{D}\) | Square root of the annualization factor |

**Interpretation:** This is a conventional annualization under assumptions about the return process. It is not a forecast that volatility will remain constant throughout the year.

### Question 12. How is the Sharpe ratio calculated?

**Equation**

\[
\operatorname{Sharpe}
=
\frac{\bar R_p-R_f}{s_p}
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\bar R_p\) | Average portfolio return per period |
| \(R_f\) | Risk-free return for the same period and in compatible units |
| \(s_p\) | Standard deviation of portfolio returns for the same period |
| \(\operatorname{Sharpe}\) | Return in excess of the risk-free rate per unit of return standard deviation |

**Interpretation:** The Sharpe ratio compares excess return with variability. If the numerator uses daily returns and the denominator uses daily standard deviation, the result is a daily-period ratio; annualization requires consistent treatment of both quantities. This is a reference formula and should only be reported as a project result if it is calculated by the relevant script.

### Question 13. What is Value at Risk (VaR)?

Let \(L\) denote portfolio loss under a clearly specified loss convention. The loss quantile at confidence level \(\alpha\) is

\[
\operatorname{VaR}_{\alpha}(L)=F_L^{-1}(\alpha)
\]

where \(F_L\) is the cumulative distribution function of loss.

**Symbols**

| Symbol | Meaning |
|---|---|
| \(L\) | Portfolio loss, defined according to the report's convention |
| \(\operatorname{VaR}_{\alpha}(L)\) | Loss quantile at confidence level \(\alpha\) |
| \(\alpha\) | Quantile level, such as \(0.95\) or \(0.99\) |
| \(F_L\) | Cumulative distribution function of loss |
| \(F_L^{-1}\) | Quantile function, or inverse CDF under the chosen convention |

**Interpretation:** At a 95% loss quantile, approximately 95% of modelled losses are at or below the VaR threshold, with approximately 5% above it, subject to ties and the quantile convention. VaR does not describe the average size of losses beyond the threshold. The report must state whether loss is defined relative to starting value, a benchmark, or another reference.

### Question 14. What is Expected Shortfall (ES)?

For a continuous loss distribution, Expected Shortfall at level \(\alpha\) can be written as

\[
\operatorname{ES}_{\alpha}(L)
=
\mathbb{E}\left[L\mid L\geq \operatorname{VaR}_{\alpha}(L)\right].
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\operatorname{ES}_{\alpha}(L)\) | Expected Shortfall at level \(\alpha\) |
| \(\mathbb{E}[\cdot]\) | Expected value or mean |
| \(L\) | Portfolio loss |
| \(\operatorname{VaR}_{\alpha}(L)\) | VaR threshold |
| \(\alpha\) | Confidence level |
| \(\mid\) | “Conditional on” |
| \(\geq\) | Greater than or equal to |

**Interpretation:** ES estimates the average loss in the tail beyond the VaR threshold. For finite Monte Carlo samples or discrete distributions, the exact empirical calculation depends on how tail observations and quantiles are selected; consult the implementation for its convention.

### Question 15. What is a GARCH(1,1) model?

**Equations**

\[
r_t=\mu+\varepsilon_t
\]

\[
\varepsilon_t=\sigma_t z_t
\]

\[
\sigma_t^2
=
\omega+\alpha_1\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2
\]

A common sufficient parameter condition for a finite unconditional variance in the standard model is

\[
\omega>0,\qquad \alpha_1\geq 0,\qquad \beta_1\geq 0,\qquad
\alpha_1+\beta_1<1.
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(r_t\) | Return at time \(t\) |
| \(\mu\) | Conditional mean or mean parameter, according to the specification |
| \(\varepsilon_t\) | Return innovation or residual at time \(t\) |
| \(\sigma_t\) | Conditional standard deviation at time \(t\) |
| \(\sigma_t^2\) | Conditional variance at time \(t\) |
| \(z_t\) | Standardized innovation, commonly with mean zero and variance one |
| \(\omega\) | Variance-equation constant |
| \(\alpha_1\) | Coefficient on the previous squared innovation (ARCH term) |
| \(\beta_1\) | Coefficient on the previous conditional variance (GARCH term) |
| \(t-1\) | Previous observation |
| \(1,1\) | One lag of squared innovations and one lag of conditional variance |

**Interpretation:** GARCH models volatility as changing over time. Large recent shocks can raise conditional variance, and high variance can persist. The parameter condition shown is a standard stationarity condition for the usual GARCH(1,1) specification, not a guarantee that the model fits every return series well.

### Question 16. What is standardized-residual bootstrap sampling, and why resample residual vectors jointly?

For asset \(i\), a standardized residual is

\[
z_{i,t}=\frac{\varepsilon_{i,t}}{\sigma_{i,t}}.
\]

If residuals from all \(N\) assets are aligned by date, a sampled residual vector can be represented as

\[
\mathbf{z}^{*}_{t}
=
\begin{bmatrix}
z_{1,k}\\
z_{2,k}\\
\vdots\\
z_{N,k}
\end{bmatrix},
\]

where \(k\) is a sampled historical index shared by all assets in that vector.

**Symbols**

| Symbol | Meaning |
|---|---|
| \(z_{i,t}\) | Standardized residual of asset \(i\) at time \(t\) |
| \(\varepsilon_{i,t}\) | Estimated innovation for asset \(i\) |
| \(\sigma_{i,t}\) | Estimated conditional standard deviation for asset \(i\) |
| \(\mathbf{z}^{*}_{t}\) | Resampled vector of standardized residuals for a simulated period |
| \(N\) | Number of assets |
| \(k\) | Historical row selected for resampling |
| \(*\) | Indicates a bootstrap-sampled quantity |

**Interpretation:** Sampling the same historical row across assets preserves the observed cross-sectional pattern of standardized residuals at that date. It does not guarantee that every dependence feature of future markets is reproduced.

### Question 17. How is QLIKE loss calculated for variance forecasts?

**Equation**

\[
\operatorname{QLIKE}
=
\frac{1}{T}\sum_{t=1}^{T}
\left[
\ln(h_t)+\frac{y_t}{h_t}
\right]
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\operatorname{QLIKE}\) | Average quasi-likelihood loss |
| \(T\) | Number of evaluated holdout observations |
| \(h_t\) | Positive forecast variance for observation \(t\) |
| \(y_t\) | Nonnegative realized-variance proxy for observation \(t\), such as squared return |
| \(\ln(h_t)\) | Natural logarithm of the forecast variance |
| \(\sum\) | Sum across the evaluation observations |

**Interpretation:** Lower QLIKE is generally preferred when models are evaluated on the same observations with the same realized proxy and compatible conventions. In this project, squared returns are used as a proxy for realized variance, so the score should not be interpreted as comparison against perfectly observed latent volatility.

**Implementation checks:** Forecast variances must be strictly positive; realized proxies must be nonnegative; and the forecast and realized arrays must be aligned and of equal length. The project tests invalid inputs and the chronological split.

### Question 18. How can the GARCH and historical-variance forecasts be compared fairly?

A useful comparison is the average QLIKE score for each model over the **same holdout dates and the same realized-variance proxy**:

\[
\Delta_{\mathrm{QLIKE}}
=
\operatorname{QLIKE}_{\mathrm{historical}}
-
\operatorname{QLIKE}_{\mathrm{GARCH}}.
\]

**Symbols**

| Symbol | Meaning |
|---|---|
| \(\Delta_{\mathrm{QLIKE}}\) | Difference between the two models' QLIKE scores |
| \(\operatorname{QLIKE}_{\mathrm{historical}}\) | QLIKE of the expanding historical-variance baseline |
| \(\operatorname{QLIKE}_{\mathrm{GARCH}}\) | QLIKE of the GARCH model |
| Positive \(\Delta_{\mathrm{QLIKE}}\) | GARCH has the lower QLIKE score in this defined comparison |
| Negative \(\Delta_{\mathrm{QLIKE}}\) | Historical-variance baseline has the lower QLIKE score |

**Interpretation:** This sign convention makes a positive difference favor GARCH. The actual output files must be inspected to determine which model performed better; do not infer model superiority from the model names alone.

### General mathematical notation glossary

| Symbol or notation | Meaning |
|---|---|
| \(t\) | Time or observation index |
| \(i,j\) | Asset indices |
| \(N\) | Number of assets |
| \(T\) | Number of observations |
| \(\bar{x}\) | Sample mean of \(x\) |
| \(s^2\) | Sample variance |
| \(s\), \(\sigma\) | Standard deviation; context determines whether it is sample or conditional |
| \(\Sigma\) | Covariance matrix |
| \(\rho\) | Correlation coefficient |
| \(\mu\) | Mean parameter or vector |
| \(\omega,\alpha_1,\beta_1\) | GARCH variance parameters |
| \(\sum\) | Summation |
| \(\ln\) | Natural logarithm |
| \(\exp(x)\) or \(e^x\) | Exponential function |
| \(\sqrt{x}\) | Nonnegative square root |
| \(x^2\) | \(x\) multiplied by itself |
| \(\mathbb{E}[X]\) | Expected value of random variable \(X\) |
| \(\sim\) | Follows a stated probability distribution |
| \(\geq,\leq\) | Greater than or equal to; less than or equal to |
| \(\mathsf{T}\) | Matrix or vector transpose |
| \(\mathbf{x}\) | A vector |
| \(\boldsymbol{\Sigma}\) | A matrix |
| \(1-\alpha\) | Tail probability associated with quantile level \(\alpha\) |
| \(*\) | Bootstrap-sampled value when used as a superscript in this section |

---

## Model Comparison Results

One recorded run produced the following terminal-value and risk comparison. These are outputs from a particular configuration and historical sample, not universal or guaranteed results.

| Metric | Multivariate Normal | GARCH(1,1) + Joint Residual Bootstrap |
|---|---:|---:|
| Simulations | 10,000 | 10,000 |
| Horizon | 252 trading days | 252 trading days |
| Initial value | $10,000 | $10,000 |
| Mean terminal value | $14,210.08 | $14,571.12 |
| Median terminal value | $13,628.99 | $13,871.93 |
| 5th-percentile terminal value | $8,544.50 | $8,854.19 |
| 95th-percentile terminal value | $21,777.04 | $21,254.57 |
| Simulated probability of loss (%) | 13.97% | 11.05% |
| Reported 95% VaR | $1,455.50 | $1,145.81 |
| Reported 95% ES | $2,345.24 | $2,226.71 |
| Reported 99% VaR | $2,984.64 | $2,924.60 |
| Reported 99% ES | $3,517.86 | $3,746.47 |

**How to read this comparison**

- The GARCH/bootstrap run has a higher mean and median terminal value in this particular simulation.
- Its reported 95% ES is lower, while its reported 99% ES is higher.
- This mixed result does not establish that one model is safer or superior.
- The precise VaR/ES meaning depends on the implementation's loss definition and empirical quantile/tail convention. Check the generated report and code before comparing these values with another system.

**GARCH out-of-sample forecast validation:** Use `reports/garch_validation/qlike_scores.csv`, `reports/garch_validation/qlike_improvement.csv`, and `reports/garch_validation/garch_validation_report.md` to determine which model has the lower QLIKE score. This README does not claim a winning model without those measured results.

## Reports and Interactive Charts

The scripts generate HTML charts that can be opened in a web browser. Open the relevant `.html` file from File Explorer, or use PowerShell from the repository root, for example:

```powershell
Start-Process .\reports\garch_comparison\charts\terminal_value_comparison.html
```

### GARCH simulation comparison

- `reports/garch_comparison/charts/terminal_value_comparison.html` — terminal value comparison.
- `reports/garch_comparison/charts/risk_measure_comparison.html` — downside risk statistics.
- `reports/garch_comparison/charts/terminal_distribution_comparison.html` — terminal-value distributions.

### GARCH forecast validation

- `reports/garch_validation/charts/qlike_model_comparison.html` — QLIKE score comparison.
- `reports/garch_validation/charts/holdout_volatility_forecasts.html` — holdout volatility forecasts.
- `reports/garch_validation/charts/nvda_variance_forecast_vs_realized.html` — NVIDIA variance forecast versus the realized proxy.
- `reports/garch_validation/charts/qlike_improvement.html` — QLIKE difference under the convention described above.

These are local report paths. They become accessible to other people only if the generated HTML files are committed to the repository or otherwise hosted and shared.

## Installation and Execution

### 1. Clone the repository

```powershell
git clone https://github.com/nasaamit005/monte-carlo-portfolio-risk-simulator.git
cd monte-carlo-portfolio-risk-simulator
```

### 2. Create and activate a virtual environment

For Windows PowerShell with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked in the current PowerShell session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the GARCH simulation comparison

```powershell
python .\src\garch_portfolio_simulation.py
```

### 5. Run out-of-sample GARCH validation

```powershell
python .\src\validate_garch_forecasts.py
```

### 6. Run the automated tests

```powershell
python -m pytest .\tests -v
```

**Note:** The commands above assume the script names and data paths match your checkout. Run commands from the repository root. Some outputs may require running the relevant data preparation scripts first.

## Testing

The latest recorded test run completed with **43 tests passing**:

```text
43 passed
```

The test suite covers the Monte Carlo workflow, GARCH comparison, and GARCH validation. Relevant test files include:

- `tests/test_monte_carlo.py`
- `tests/test_garch_comparison.py`
- `tests/test_garch_validation.py`

Test counts can change as the project evolves. Re-run pytest to confirm the status of your current checkout.

## Assumptions and Limitations

- Historical estimates may not represent future market behavior.
- The multivariate-normal model may underrepresent heavy tails and extreme market events.
- GARCH(1,1) models conditional variance but does not capture every possible market regime or dependence pattern.
- Joint residual bootstrap depends on the representativeness of historical standardized residuals.
- A squared return is a noisy proxy for latent daily variance.
- Forecast validation depends on the selected holdout window, benchmark model, and loss function.
- Transaction costs, taxes, slippage, liquidity constraints, corporate-action details, and changing market regimes may not be fully modelled.
- Equal weighting is a simplifying assumption and is not necessarily an optimal allocation.
- Monte Carlo probabilities are conditional on the fitted model and its assumptions.
- The reported numerical results depend on data, configuration, and implementation details.

## Future Improvements

- Add rolling-window and regime-sensitive parameter estimation.
- Compare alternative innovation distributions and volatility models.
- Evaluate portfolio tail risk across multiple holdout windows.
- Add transaction-cost and rebalancing assumptions.
- Add confidence intervals and sensitivity analysis for risk measures.
- Add a configurable web dashboard for assets, weights, horizon, and simulation count.
- Add continuous integration to run tests on every push.
- Document data provenance, retrieval date, and licensing for the underlying market data.

## References

- Engle, R. F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation*. Econometrica.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
- McNeil, A. J., Frey, R., and Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton University Press.
- `docs/MONTE_CARLO_METHODOLOGY.md` — project methodology notes.
- `docs/REFERENCES.md` — additional project references.

---

## Author / Portfolio Use

This project demonstrates applied Python, financial return analysis, Monte Carlo simulation, volatility modelling, model validation, automated testing, and interactive reporting. In interviews, describe the methods you implemented, the assumptions you tested, the actual outputs you measured, and the limitations you identified.

**Disclaimer:** For educational and portfolio-demonstration purposes only. Nothing in this repository constitutes investment, trading, tax, or legal advice.
