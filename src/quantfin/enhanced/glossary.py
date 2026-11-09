"""Plain-language definitions shown as tooltips and on the About page. Formulas match metrics.py / risk.py."""

from __future__ import annotations

GLOSSARY: dict[str, dict[str, str]] = {
    "CAGR": {
        "def": "Compound annual growth rate: the constant yearly return that would give the same final value.",
        "formula": "(final value / start value)^(252 / days) - 1",
        "read": "Higher is better, but compare it with volatility and drawdown, not alone.",
    },
    "Volatility": {
        "def": "Annualised standard deviation of daily returns: how much returns fluctuate.",
        "formula": "std(daily return) x sqrt(252)",
        "read": "Lower means a smoother ride. It treats gains and losses alike.",
    },
    "Sharpe": {
        "def": "Return earned per unit of total risk, after the risk-free rate.",
        "formula": "(mean daily return x 252 - risk-free) / (std x sqrt(252))",
        "read": "Above 1 is strong, below 0.5 is weak. Backtested Sharpe ratios are optimistic.",
    },
    "Sortino": {
        "def": "Like Sharpe but only penalises downside volatility.",
        "formula": "excess return / downside deviation",
        "read": "Higher is better; it is more forgiving of upside swings than Sharpe.",
    },
    "Calmar": {
        "def": "CAGR divided by the size of the worst drawdown.",
        "formula": "CAGR / |max drawdown|",
        "read": "How much growth you earned per unit of worst-case pain.",
    },
    "Max drawdown": {
        "def": "Largest peak-to-trough fall of the portfolio value.",
        "formula": "min(value / running peak - 1)",
        "read": "-30% means at the worst point you were 30% below your previous high.",
    },
    "Beta": {
        "def": "Sensitivity to the benchmark: the share of a benchmark move the strategy tends to follow.",
        "formula": "cov(strategy, benchmark) / var(benchmark)",
        "read": "1.0 moves with SPY; 0.6 tends to move 0.6% for each 1% SPY move.",
    },
    "Alpha (ann.)": {
        "def": "Annualised return not explained by beta to the benchmark.",
        "formula": "intercept of excess-return regression x 252",
        "read": "Positive alpha means return above what market exposure alone predicts.",
    },
    "Information ratio": {
        "def": "Active return per unit of tracking error versus the benchmark.",
        "formula": "mean(active return) x 252 / tracking error",
        "read": "Above 0.5 is good, above 1 is exceptional; negative means lagging the benchmark.",
    },
    "Tracking error": {
        "def": "Annualised volatility of the difference between strategy and benchmark returns.",
        "formula": "std(strategy - benchmark) x sqrt(252)",
        "read": "How far the strategy strays from the benchmark.",
    },
    "Up capture": {
        "def": "Share of the benchmark's gains captured on days the benchmark rose.",
        "formula": "mean(strategy | benchmark up) / mean(benchmark | up)",
        "read": "Above 1 captures more than the benchmark in rallies.",
    },
    "Down capture": {
        "def": "Share of the benchmark's losses suffered on days the benchmark fell.",
        "formula": "mean(strategy | benchmark down) / mean(benchmark | down)",
        "read": "Below 1 loses less than the benchmark in sell-offs.",
    },
    "Historical VaR": {
        "def": "Loss not exceeded on 95% of days, taken directly from past returns.",
        "formula": "-quantile(returns, 5%)",
        "read": "2% means on 1 day in 20 the loss was at least 2%.",
    },
    "Historical CVaR": {
        "def": "Average loss on the worst 5% of days (expected shortfall).",
        "formula": "-mean(returns | returns <= VaR quantile)",
        "read": "Always at least as large as VaR; it measures how bad the tail is.",
    },
    "Parametric VaR": {
        "def": "VaR assuming returns are normally distributed.",
        "formula": "-(mean + z x std)",
        "read": "Understates risk when returns have fat tails, as these do.",
    },
    "Cornish-Fisher VaR": {
        "def": "Normal VaR adjusted for skewness and fat tails.",
        "formula": "z adjusted by skewness and excess kurtosis",
        "read": "Closer to historical VaR than the plain normal estimate.",
    },
    "Monte Carlo VaR": {
        "def": "VaR from thousands of simulated days at the latest weights, using the project's expected returns and covariance.",
        "formula": "quantile of simulated portfolio returns",
        "read": "Forward-looking model estimate; depends on a normality assumption.",
    },
    "Risk contribution": {
        "def": "Share of total portfolio volatility coming from each holding.",
        "formula": "w_i x (Sigma w)_i / sigma_p",
        "read": "Compare it with the capital weight: a 26% weight giving 32% of risk is a risk concentration.",
    },
    "Turnover": {
        "def": "Total one-way trading as a multiple of portfolio value.",
        "formula": "sum |target weight - drifted weight|",
        "read": "30x over 10 years is about 3x per year; costs scale with it.",
    },
    "Cost drag": {
        "def": "Return lost to transaction cost and slippage.",
        "formula": "gross CAGR - net CAGR",
        "read": "Small here (7 bps per unit traded) but higher for high-turnover strategies.",
    },
    "Effective N": {
        "def": "Number of equally weighted holdings that would give the same concentration.",
        "formula": "1 / sum(w^2)",
        "read": "12 holdings with an effective N of 4 behave like a 4-stock portfolio.",
    },
    "PSR": {
        "def": "Probabilistic Sharpe ratio: probability the true Sharpe exceeds zero given sample length, skew and kurtosis.",
        "formula": "Bailey & Lopez de Prado (2012)",
        "read": "Above 95% suggests the Sharpe is unlikely to be luck within this sample; selection bias is not covered.",
    },
    "Jarque-Bera": {
        "def": "Test of whether returns are normally distributed.",
        "formula": "n/6 (S^2 + (K-3)^2 / 4)",
        "read": "A tiny p-value means returns are not normal, so Gaussian VaR is unreliable.",
    },
    "Ljung-Box": {
        "def": "Test for autocorrelation in returns over 10 lags.",
        "formula": "n(n+2) sum rho_k^2 / (n-k)",
        "read": "A small p-value indicates predictable structure in daily returns.",
    },
    "Efficient frontier": {
        "def": "Portfolios with the highest expected return for each level of risk.",
        "formula": "min w'Sigma w subject to w'mu = target",
        "read": "Portfolios below the curve are dominated; it is built from estimated, uncertain inputs.",
    },
    "Capital allocation line": {
        "def": "Line from the risk-free rate through the maximum-Sharpe portfolio.",
        "formula": "rf + Sharpe x volatility",
        "read": "Mixes the tangency portfolio with cash to reach any risk level.",
    },
    "Minimum volatility": {"def": "Weights that minimise portfolio variance under the weight cap.", "formula": "min w'Sigma w", "read": "Ignores expected returns, so it is robust but can lag in bull markets."},
    "Maximum Sharpe": {"def": "Weights that maximise expected Sharpe from estimated returns and covariance.", "formula": "max (w'mu - rf) / sqrt(w'Sigma w)", "read": "Uses return estimates, so it is the most sensitive and concentrated method."},
    "Risk parity": {"def": "Weights chosen so each holding contributes equal risk.", "formula": "w_i (Sigma w)_i equal across i", "read": "Diversifies risk, not capital; no return forecast needed."},
    "HRP": {"def": "Hierarchical Risk Parity: clusters correlated assets and splits risk down the tree.", "formula": "Lopez de Prado (2016)", "read": "Avoids matrix inversion, so it is stable with correlated assets."},
    "Equal weight": {"def": "The same weight in every asset, rebalanced monthly.", "formula": "1 / n", "read": "A strong naive benchmark that optimisers must beat after costs."},
    "Regime": {"def": "Market state (bull or bear trend, high or normal volatility) from the project's own classification.", "formula": "trailing return and volatility rules", "read": "Labels use past data, so days tagged bear often precede rebounds."},
    "Walk-forward": {"def": "Weights are estimated using only data available at each rebalance, then applied to the following days.", "formula": "estimate at t, earn t+1", "read": "Avoids look-ahead bias in the backtest."},
    "Survivorship bias": {"def": "Using stocks known to have succeeded by today.", "formula": "n/a", "read": "Inflates returns; the 12-stock universe here was chosen with hindsight."},
}
