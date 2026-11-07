
"""Flask dashboard for the Quantitative Portfolio Intelligence platform."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, jsonify, render_template, request

from quantfin.services.dashboard_optimization import (
    DashboardOptimizationError,
    optimize_dashboard_portfolio,
)

ROOT_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


REPORT_FILES = {
    "strategy_comparison": (
        "reports/walk_forward_comparison/strategy_comparison_summary.csv"
    ),
    "performance": "reports/performance_analytics/performance_summary.csv",
    "benchmark": "reports/performance_analytics/benchmark_summary.csv",
    "stress": "reports/stress_testing/historical_stress_results.csv",
    "robustness": (
        "reports/robustness_analysis/strategy_robustness_scorecard.parquet"
    ),
    "sensitivity": (
        "reports/robustness_analysis/sensitivity_experiment_results.parquet"
    ),
    "regimes": (
        "reports/robustness_analysis/market_regime_classification.csv"
    ),
    "regime_performance": (
        "reports/robustness_analysis/market_regime_performance.csv"
    ),
    "rolling_robustness": (
        "reports/robustness_analysis/rolling_robustness_metrics.csv"
    ),
}


PAGES = {
    "overview": {
        "label": "Overview",
        "endpoint": "overview",
        "icon": "layout-dashboard",
    },
    "optimization": {
        "label": "Portfolio Optimizer",
        "endpoint": "optimization",
        "icon": "sliders-horizontal",
    },
    "performance": {
        "label": "Performance",
        "endpoint": "performance",
        "icon": "chart-no-axes-combined",
    },
    "risk": {
        "label": "Risk Analysis",
        "endpoint": "risk",
        "icon": "shield-check",
    },
    "robustness": {
        "label": "Robustness",
        "endpoint": "robustness",
        "icon": "activity",
    },
    "methodology": {
        "label": "Methodology",
        "endpoint": "methodology",
        "icon": "book-open",
    },
}


def load_report(relative_path: str) -> pd.DataFrame:
    """Load a CSV or Parquet report, returning an empty frame if unavailable."""
    path = ROOT_DIR / relative_path

    if not path.exists():
        logger.warning("Report not found: %s", path)
        return pd.DataFrame()

    try:
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            return pd.read_parquet(path)

        if suffix == ".csv":
            return pd.read_csv(path)

        logger.warning("Unsupported report format: %s", path)
        return pd.DataFrame()

    except (
        OSError,
        ValueError,
        ImportError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        logger.exception("Could not load report: %s", path)
        return pd.DataFrame()


def safe_number(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if np.isfinite(result) else default


def find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Find a column using case-insensitive candidate names."""
    if frame.empty:
        return None

    lookup = {str(column).strip().lower(): column for column in frame.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return str(lookup[candidate.lower()])

    return None


def first_valid_value(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Read the first non-null value from the first matching column."""
    column = find_column(frame, candidates)

    if column is None or frame.empty:
        return default

    values = frame[column].dropna()

    return values.iloc[0] if not values.empty else default


def frame_records(
    frame: pd.DataFrame,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Convert report data into JSON-safe records."""
    if frame.empty:
        return []

    output = frame.head(limit).copy()
    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.astype(object).where(pd.notna(output), None)

    return output.to_dict(orient="records")


def make_chart(
    figure: go.Figure,
    height: int = 360,
) -> str:
    """Render a responsive Plotly chart without embedding another JS copy."""
    figure.update_layout(
        height=height,
        autosize=True,
        margin={"l": 45, "r": 25, "t": 45, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#344054"},
        hoverlabel={
            "bgcolor": "#ffffff",
            "font": {"color": "#101828"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor="#edf1f7",
        zeroline=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#edf1f7",
        zeroline=False,
    )

    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def build_strategy_charts(
    strategy: pd.DataFrame,
) -> dict[str, str]:
    """Build strategy comparison charts when the source columns exist."""
    charts: dict[str, str] = {}

    if strategy.empty:
        return charts

    name_col = find_column(
        strategy,
        ("strategy", "strategy_name", "portfolio", "name"),
    )
    return_col = find_column(
        strategy,
        ("annual_return", "annualized_return", "cagr"),
    )
    volatility_col = find_column(
        strategy,
        ("annual_volatility", "annualized_volatility", "volatility"),
    )
    sharpe_col = find_column(
        strategy,
        ("sharpe_ratio", "sharpe"),
    )
    drawdown_col = find_column(
        strategy,
        ("max_drawdown", "maximum_drawdown", "mdd"),
    )

    if name_col and return_col and volatility_col:
        frame = strategy.copy()
        frame[return_col] = pd.to_numeric(
            frame[return_col], errors="coerce"
        )
        frame[volatility_col] = pd.to_numeric(
            frame[volatility_col], errors="coerce"
        )
        frame = frame.dropna(subset=[return_col, volatility_col])

        if not frame.empty:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=frame[volatility_col] * 100,
                    y=frame[return_col] * 100,
                    mode="markers+text",
                    text=frame[name_col].astype(str),
                    textposition="top center",
                    marker={
                        "size": 13,
                        "line": {"width": 1, "color": "#ffffff"},
                    },
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Annualized volatility: %{x:.2f}%<br>"
                        "Annualized return: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                    name="Strategies",
                )
            )
            figure.update_layout(
                title="Annualized Return vs. Volatility",
                xaxis_title="Annualized volatility (%)",
                yaxis_title="Annualized return (%)",
            )
            charts["return_risk"] = make_chart(figure)

    if name_col and sharpe_col:
        frame = strategy.copy()
        frame[sharpe_col] = pd.to_numeric(
            frame[sharpe_col], errors="coerce"
        )
        frame = frame.dropna(subset=[sharpe_col]).sort_values(
            sharpe_col,
            ascending=True,
        )

        if not frame.empty:
            figure = go.Figure(
                go.Bar(
                    x=frame[sharpe_col],
                    y=frame[name_col].astype(str),
                    orientation="h",
                    text=frame[sharpe_col].round(3),
                    textposition="outside",
                    marker={"color": "#5267e8"},
                )
            )
            figure.update_layout(
                title="Risk-Adjusted Performance",
                xaxis_title="Sharpe ratio",
                yaxis_title="",
            )
            charts["sharpe"] = make_chart(figure)

    if name_col and drawdown_col:
        frame = strategy.copy()
        frame[drawdown_col] = pd.to_numeric(
            frame[drawdown_col], errors="coerce"
        )
        frame = frame.dropna(subset=[drawdown_col])

        if not frame.empty:
            figure = go.Figure(
                go.Bar(
                    x=frame[name_col].astype(str),
                    y=frame[drawdown_col] * 100,
                    marker={"color": "#ef8354"},
                    hovertemplate=(
                        "%{x}<br>Maximum drawdown: %{y:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )
            figure.update_layout(
                title="Maximum Drawdown Comparison",
                xaxis_title="Strategy",
                yaxis_title="Maximum drawdown (%)",
            )
            charts["drawdown"] = make_chart(figure)

    return charts


def build_stress_chart(stress: pd.DataFrame) -> str | None:
    """Build a stress chart when scenario and loss columns are available."""
    if stress.empty:
        return None

    scenario_col = find_column(
        stress,
        ("scenario", "scenario_name", "stress_scenario", "name"),
    )
    loss_col = find_column(
        stress,
        ("portfolio_return", "portfolio_loss", "return", "loss"),
    )

    if not scenario_col or not loss_col:
        return None

    frame = stress.copy()
    frame[loss_col] = pd.to_numeric(frame[loss_col], errors="coerce")
    frame = frame.dropna(subset=[loss_col])

    if frame.empty:
        return None

    figure = go.Figure(
        go.Bar(
            x=frame[scenario_col].astype(str),
            y=frame[loss_col] * 100,
            marker={"color": "#e76f51"},
            hovertemplate="%{x}<br>Scenario result: %{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        title="Historical Stress Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Return / loss (%)",
    )

    return make_chart(figure, height=400)


def build_robustness_chart(robustness: pd.DataFrame) -> str | None:
    """Build a robustness scorecard chart if compatible columns exist."""
    if robustness.empty:
        return None

    name_col = find_column(
        robustness,
        ("strategy", "strategy_name", "name"),
    )
    score_col = find_column(
        robustness,
        ("mean_sharpe", "mean_sharpe_ratio", "sharpe_ratio"),
    )

    if not name_col or not score_col:
        return None

    frame = robustness.copy()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=[score_col]).sort_values(score_col)

    if frame.empty:
        return None

    figure = go.Figure(
        go.Bar(
            x=frame[score_col],
            y=frame[name_col].astype(str),
            orientation="h",
            marker={"color": "#16a6a1"},
        )
    )
    figure.update_layout(
        title="Robustness: Mean Sharpe Across Experiments",
        xaxis_title="Mean Sharpe ratio",
        yaxis_title="",
    )

    return make_chart(figure)


def build_dashboard_context(active_page: str) -> dict[str, Any]:
    """Load reports and prepare shared page context."""
    reports = {
        key: load_report(path)
        for key, path in REPORT_FILES.items()
    }

    strategy = reports["strategy_comparison"]
    performance = reports["performance"]
    benchmark = reports["benchmark"]

    charts = build_strategy_charts(strategy)
    stress_chart = build_stress_chart(reports["stress"])
    robustness_chart = build_robustness_chart(reports["robustness"])

    if stress_chart:
        charts["stress"] = stress_chart

    if robustness_chart:
        charts["robustness"] = robustness_chart

    strategy_name_col = find_column(
        strategy,
        ("strategy", "strategy_name", "portfolio", "name"),
    )
    sharpe_col = find_column(strategy, ("sharpe_ratio", "sharpe"))

    best_strategy = "Not available"

    if strategy_name_col and sharpe_col and not strategy.empty:
        candidates = strategy.copy()
        candidates[sharpe_col] = pd.to_numeric(
            candidates[sharpe_col], errors="coerce"
        )
        candidates = candidates.dropna(subset=[sharpe_col])

        if not candidates.empty:
            best_row = candidates.loc[candidates[sharpe_col].idxmax()]
            best_strategy = str(best_row[strategy_name_col])

    performance_return = first_valid_value(
        performance,
        ("annual_return", "annualized_return", "cagr"),
    )
    performance_volatility = first_valid_value(
        performance,
        ("annual_volatility", "annualized_volatility", "volatility"),
    )
    performance_sharpe = first_valid_value(
        performance,
        ("sharpe_ratio", "sharpe"),
    )
    performance_drawdown = first_valid_value(
        performance,
        ("max_drawdown", "maximum_drawdown", "mdd"),
    )

    return {
        "pages": PAGES,
        "active_page": active_page,
        "reports": reports,
        "charts": charts,
        "best_strategy": best_strategy,
        "strategy_count": int(len(strategy)),
        "performance_metrics": {
            "annual_return": safe_number(performance_return, float("nan")),
            "annual_volatility": safe_number(
                performance_volatility, float("nan")
            ),
            "sharpe_ratio": safe_number(performance_sharpe, float("nan")),
            "max_drawdown": safe_number(performance_drawdown, float("nan")),
        },
        "strategy_records": frame_records(strategy),
        "performance_records": frame_records(performance),
        "benchmark_records": frame_records(benchmark),
        "stress_records": frame_records(reports["stress"]),
        "robustness_records": frame_records(reports["robustness"]),
        "sensitivity_records": frame_records(reports["sensitivity"]),
        "regime_records": frame_records(reports["regimes"]),
        "regime_performance_records": frame_records(
            reports["regime_performance"]
        ),
        "rolling_robustness_records": frame_records(
            reports["rolling_robustness"]
        ),
        "report_status": {
            key: not frame.empty
            for key, frame in reports.items()
        },
    }


@app.get("/")
def overview():
    return render_template(
        "index.html",
        **build_dashboard_context("overview"),
    )


@app.get("/optimization")
def optimization():
    return render_template(
        "index.html",
        **build_dashboard_context("optimization"),
    )


@app.get("/performance")
def performance():
    return render_template(
        "index.html",
        **build_dashboard_context("performance"),
    )


@app.get("/risk")
def risk():
    return render_template(
        "index.html",
        **build_dashboard_context("risk"),
    )


@app.get("/robustness")
def robustness():
    return render_template(
        "index.html",
        **build_dashboard_context("robustness"),
    )


@app.get("/methodology")
def methodology():
    return render_template(
        "index.html",
        **build_dashboard_context("methodology"),
    )


@app.post("/api/optimization")
def optimization_api():
    """Run a dashboard optimization request and return JSON results."""
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(
            {
                "status": "error",
                "message": "A valid JSON request body is required.",
            }
        ), 400

    try:
        objective = str(
            payload.get("objective", "maximum_sharpe")
        ).strip().lower()

        result = optimize_dashboard_portfolio(
            objective=objective,
            lookback=int(payload.get("lookback", 252)),
            risk_free_rate=float(payload.get("risk_free_rate", 0.04)),
            min_weight=float(payload.get("min_weight", 0.0)),
            max_weight=float(payload.get("max_weight", 0.50)),
            frontier_points=int(payload.get("frontier_points", 25)),
        )

        return jsonify(result)

    except (DashboardOptimizationError, ValueError, TypeError) as exc:
        return jsonify(
            {
                "status": "error",
                "message": str(exc),
            }
        ), 400

    except Exception:
        logger.exception("Unexpected dashboard optimization error.")
        return jsonify(
            {
                "status": "error",
                "message": (
                    "An unexpected optimization error occurred. "
                    "Check the terminal log for details."
                ),
            }
        ), 500


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "application": "Quantitative Portfolio Intelligence Dashboard",
            "optimization_api": "/api/optimization",
        }
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
    )
