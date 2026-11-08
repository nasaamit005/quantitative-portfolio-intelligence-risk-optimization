"""Results container consumed by the reporting layer (Excel, HTML, Plotly charts, Flask dashboard)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analytics import BENCH, WindowAnalytics, analyze_window
from .config import Config


@dataclass
class Results:
    cfg: Config
    meta: dict[str, Any]
    assets: pd.DataFrame
    bench_full: pd.Series
    strategy_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: dict[str, pd.Series]
    backtest_diagnostics: list[str]
    full: WindowAnalytics
    static_weights: pd.DataFrame
    static_stats: pd.DataFrame
    frontier: pd.DataFrame
    random_pf: pd.DataFrame
    asset_stats: pd.DataFrame
    corr: pd.DataFrame
    cov: pd.DataFrame
    desc: pd.DataFrame
    quality: pd.DataFrame
    sens: pd.DataFrame
    wf_year: pd.DataFrame
    wf_select: pd.DataFrame
    regimes: pd.DataFrame
    regime_labels: pd.DataFrame
    stress_hist: pd.DataFrame
    stress_hypo: pd.DataFrame
    cost_table: pd.DataFrame
    risk_contrib: dict[str, pd.DataFrame]
    mc_table: pd.DataFrame
    mc_sample: dict[str, np.ndarray]
    findings: pd.DataFrame
    recs: pd.DataFrame
    artifacts: dict[str, str] = field(default_factory=dict)
    sweep_diagnostics: list[str] = field(default_factory=list)

    @property
    def strategies(self) -> list[str]:
        return [c for c in self.strategy_returns.columns if c != BENCH]

    def window(self, start: str | None = None, end: str | None = None) -> WindowAnalytics:
        """Re-compute return-based analytics on a date sub-window (used by dashboard filters)."""
        sr = self.strategy_returns.loc[start:end]
        return analyze_window(sr, self.cfg)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> Results:
        """Load a Results pickle written by `save` (trusted, self-generated artifact only)."""
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301
