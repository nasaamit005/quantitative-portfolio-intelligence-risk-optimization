"""Configuration for the analytics extension (dataclass, loadable from JSON/YAML/TOML)."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

REBALANCE_CHOICES = ("weekly", "monthly", "quarterly", "annual")
DEFAULT_STRATEGIES = ("equal_weight", "min_vol", "max_sharpe", "risk_parity", "hrp")
DEFAULT_STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC 2008-09": ("2008-09-02", "2009-03-09"),
    "US downgrade 2011": ("2011-07-22", "2011-10-03"),
    "China deval. 2015": ("2015-08-17", "2015-08-25"),
    "Q4 2018 sell-off": ("2018-09-20", "2018-12-24"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "Rate shock 2022": ("2022-01-03", "2022-10-12"),
}


@dataclass(frozen=True)
class Config:
    """All tunable parameters. Every analytical result is a function of (data, Config)."""

    benchmark: str = "SPY"
    benchmark_in_universe: bool = False
    risk_free_rate: float = 0.02
    periods_per_year: int = 252
    lookback: int = 252
    rebalance: str = "monthly"
    min_weight: float = 0.0
    max_weight: float = 0.35
    cost_bps: float = 10.0
    cov_shrinkage: float = 0.10
    confidence: float = 0.95
    var_horizon: int = 1
    mc_sims: int = 20000
    seed: int = 42
    frontier_points: int = 40
    random_portfolios: int = 3000
    rolling_window: int = 126
    top_drawdowns: int = 5
    ma_window: int = 200
    vol_window: int = 63
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    lookback_grid: tuple[int, ...] = (126, 252, 378, 504)
    cost_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0)
    max_weight_grid: tuple[float, ...] = (0.25, 0.35, 0.50, 1.0)
    rebalance_grid: tuple[str, ...] = REBALANCE_CHOICES
    equity_shocks: tuple[float, ...] = (-0.10, -0.20, -0.30)
    vol_multipliers: tuple[float, ...] = (1.5, 2.0, 3.0)
    stress_windows: dict[str, tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_STRESS_WINDOWS)
    )
    output_dir: str = "outputs/qpi"

    def validate(self) -> None:
        """Raise ValueError for inconsistent settings."""
        if self.rebalance not in REBALANCE_CHOICES:
            raise ValueError(
                f"rebalance must be one of {REBALANCE_CHOICES}, got {self.rebalance!r}"
            )
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if not 0 <= self.min_weight < self.max_weight <= 1:
            raise ValueError("require 0 <= min_weight < max_weight <= 1")
        if self.lookback < 30:
            raise ValueError("lookback must be at least 30 observations")
        if not 0 <= self.cov_shrinkage <= 1:
            raise ValueError("cov_shrinkage must be in [0, 1]")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.var_horizon < 1:
            raise ValueError("var_horizon must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        clean: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "stress_windows":
                clean[key] = {k: tuple(v) for k, v in value.items()}
            elif isinstance(value, list):
                clean[key] = tuple(value)
            else:
                clean[key] = value
        cfg = cls(**clean)
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load from .json, .yaml/.yml or .toml."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            raw = json.loads(text)
        elif p.suffix in (".yaml", ".yml"):
            import yaml  # optional dependency, imported lazily

            raw = yaml.safe_load(text) or {}
        elif p.suffix == ".toml":
            raw = tomllib.loads(text)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")
        return cls.from_dict(raw)
