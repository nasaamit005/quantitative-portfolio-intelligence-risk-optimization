
"""Portfolio constraint definitions and independent weight validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    """Investment constraints applied to portfolio weights."""

    min_weight: float = 0.0
    max_weight: float = 0.50
    allow_short_selling: bool = False
    fully_invested: bool = True
    weight_sum: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_weight):
            raise ValueError("min_weight must be finite.")

        if not np.isfinite(self.max_weight):
            raise ValueError("max_weight must be finite.")

        if not np.isfinite(self.weight_sum):
            raise ValueError("weight_sum must be finite.")

        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot exceed max_weight.")

        if self.weight_sum <= 0:
            raise ValueError("weight_sum must be positive.")

        if not self.allow_short_selling and self.min_weight < 0:
            raise ValueError(
                "Negative minimum weights require allow_short_selling=True."
            )

        if self.max_weight < 0:
            raise ValueError("max_weight cannot be negative.")

    def bounds(self, n_assets: int) -> list[tuple[float, float]]:
        """Return per-asset lower and upper weight bounds."""
        if n_assets < 1:
            raise ValueError("At least one asset is required.")

        lower = self.min_weight

        if self.fully_invested:
            if n_assets * lower > self.weight_sum + 1e-12:
                raise ValueError("Minimum weight constraints are infeasible.")

            if n_assets * self.max_weight < self.weight_sum - 1e-12:
                raise ValueError("Maximum weight constraints are infeasible.")

        return [(lower, self.max_weight) for _ in range(n_assets)]


def validate_portfolio_weights(
    weights: pd.Series,
    constraints: PortfolioConstraints | None = None,
    tolerance: float = 1e-6,
) -> dict[str, bool | float | str]:
    """Independently validate portfolio weights against constraints."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series.")

    if weights.empty:
        raise ValueError("weights cannot be empty.")

    if weights.index.has_duplicates:
        raise ValueError("Weight asset labels must be unique.")

    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio weights contain NaN or infinite values.")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    rules = constraints or PortfolioConstraints()
    values = weights.to_numpy(dtype=float)

    weight_sum = float(values.sum())
    minimum_weight = float(values.min())
    maximum_weight = float(values.max())

    lower_bound = rules.min_weight
    if rules.allow_short_selling:
        lower_bound = min(lower_bound, -1.0)

    sum_valid = (
        not rules.fully_invested
        or abs(weight_sum - rules.weight_sum) <= tolerance
    )
    lower_valid = minimum_weight >= lower_bound - tolerance
    upper_valid = maximum_weight <= rules.max_weight + tolerance

    return {
        "weight_sum": weight_sum,
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
        "sum_valid": sum_valid,
        "lower_bound_valid": lower_valid,
        "upper_bound_valid": upper_valid,
        "all_valid": sum_valid and lower_valid and upper_valid,
        "message": (
            "All portfolio weight constraints satisfied."
            if sum_valid and lower_valid and upper_valid
            else "One or more portfolio weight constraints failed."
        ),
    }
