
"""Unit tests for portfolio constraints and weight validation."""

import numpy as np
import pandas as pd
import pytest

from quantfin.optimization.constraints import (
    PortfolioConstraints,
    validate_portfolio_weights,
)


class TestPortfolioConstraints:
    """Tests for PortfolioConstraints."""

    def test_default_constraints(self):
        constraints = PortfolioConstraints()

        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 0.50
        assert constraints.allow_short_selling is False
        assert constraints.fully_invested is True
        assert constraints.weight_sum == 1.0

    def test_valid_custom_constraints(self):
        constraints = PortfolioConstraints(
            min_weight=0.05,
            max_weight=0.40,
            fully_invested=True,
        )

        assert constraints.min_weight == 0.05
        assert constraints.max_weight == 0.40

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"min_weight": np.nan}, "min_weight must be finite"),
            ({"min_weight": np.inf}, "min_weight must be finite"),
            ({"max_weight": np.nan}, "max_weight must be finite"),
            ({"max_weight": np.inf}, "max_weight must be finite"),
            ({"weight_sum": np.nan}, "weight_sum must be finite"),
            ({"weight_sum": np.inf}, "weight_sum must be finite"),
            (
                {"min_weight": 0.60, "max_weight": 0.50},
                "min_weight cannot exceed max_weight",
            ),
            ({"weight_sum": 0.0}, "weight_sum must be positive"),
            ({"weight_sum": -1.0}, "weight_sum must be positive"),
            (
                {"min_weight": -0.10},
                "Negative minimum weights require allow_short_selling=True",
            ),
            (
                {
                    "min_weight": -0.20,
                    "max_weight": -0.10,
                    "allow_short_selling": True,
                },
                "max_weight cannot be negative",
            ),
        ],
    )
    def test_invalid_constraint_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PortfolioConstraints(**kwargs)

    def test_short_selling_allows_negative_minimum_weight(self):
        constraints = PortfolioConstraints(
            min_weight=-0.20,
            max_weight=1.0,
            allow_short_selling=True,
        )

        assert constraints.min_weight == -0.20
        assert constraints.allow_short_selling is True

    def test_valid_bounds(self):
        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.50,
        )

        bounds = constraints.bounds(4)

        assert bounds == [
            (0.0, 0.50),
            (0.0, 0.50),
            (0.0, 0.50),
            (0.0, 0.50),
        ]

    def test_bounds_reject_zero_assets(self):
        constraints = PortfolioConstraints()

        with pytest.raises(ValueError, match="At least one asset"):
            constraints.bounds(0)

    def test_bounds_reject_negative_asset_count(self):
        constraints = PortfolioConstraints()

        with pytest.raises(ValueError, match="At least one asset"):
            constraints.bounds(-1)

    def test_infeasible_minimum_weight_constraints(self):
        constraints = PortfolioConstraints(
            min_weight=0.30,
            max_weight=0.60,
        )

        with pytest.raises(
            ValueError,
            match="Minimum weight constraints are infeasible",
        ):
            constraints.bounds(4)

    def test_infeasible_maximum_weight_constraints(self):
        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.20,
        )

        with pytest.raises(
            ValueError,
            match="Maximum weight constraints are infeasible",
        ):
            constraints.bounds(4)

    def test_feasible_boundary_constraints(self):
        constraints = PortfolioConstraints(
            min_weight=0.25,
            max_weight=0.25,
        )

        bounds = constraints.bounds(4)

        assert bounds == [(0.25, 0.25)] * 4

    def test_non_fully_invested_bounds(self):
        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.20,
            fully_invested=False,
        )

        assert constraints.bounds(3) == [(0.0, 0.20)] * 3

    def test_bounds_with_short_selling(self):
        constraints = PortfolioConstraints(
            min_weight=-0.20,
            max_weight=0.60,
            allow_short_selling=True,
        )

        assert constraints.bounds(3) == [(-0.20, 0.60)] * 3


class TestValidatePortfolioWeights:
    """Tests for independent portfolio weight validation."""

    @pytest.fixture
    def valid_weights(self):
        return pd.Series(
            [0.30, 0.25, 0.25, 0.20],
            index=["AAPL", "MSFT", "NVDA", "JPM"],
        )

    def test_valid_portfolio_weights(self, valid_weights):
        result = validate_portfolio_weights(valid_weights)

        assert result["all_valid"] is True
        assert result["sum_valid"] is True
        assert result["lower_bound_valid"] is True
        assert result["upper_bound_valid"] is True
        assert result["weight_sum"] == pytest.approx(1.0)

    def test_weight_sum_calculation(self, valid_weights):
        result = validate_portfolio_weights(valid_weights)

        assert result["weight_sum"] == pytest.approx(1.0)

    def test_minimum_weight_calculation(self, valid_weights):
        result = validate_portfolio_weights(valid_weights)

        assert result["minimum_weight"] == pytest.approx(0.20)

    def test_maximum_weight_calculation(self, valid_weights):
        result = validate_portfolio_weights(valid_weights)

        assert result["maximum_weight"] == pytest.approx(0.30)

    def test_weight_sum_violation(self):
        weights = pd.Series([0.30, 0.30, 0.30])

        result = validate_portfolio_weights(weights)

        assert result["sum_valid"] is False
        assert result["all_valid"] is False

    def test_negative_weight_violation(self):
        weights = pd.Series([0.50, 0.30, 0.20, -0.01])

        result = validate_portfolio_weights(weights)

        assert result["lower_bound_valid"] is False
        assert result["all_valid"] is False

    def test_maximum_weight_violation(self):
        weights = pd.Series([0.60, 0.20, 0.10, 0.10])

        result = validate_portfolio_weights(weights)

        assert result["upper_bound_valid"] is False
        assert result["all_valid"] is False

    def test_custom_maximum_weight(self):
        weights = pd.Series([0.40, 0.30, 0.20, 0.10])

        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.40,
        )

        result = validate_portfolio_weights(weights, constraints)

        assert result["all_valid"] is True

    def test_custom_minimum_weight(self):
        weights = pd.Series([0.40, 0.30, 0.20, 0.10])

        constraints = PortfolioConstraints(
            min_weight=0.15,
            max_weight=0.50,
        )

        result = validate_portfolio_weights(weights, constraints)

        assert result["lower_bound_valid"] is False
        assert result["all_valid"] is False

    def test_non_fully_invested_portfolio(self):
        weights = pd.Series([0.20, 0.20, 0.20])

        constraints = PortfolioConstraints(
            max_weight=0.50,
            fully_invested=False,
        )

        result = validate_portfolio_weights(weights, constraints)

        assert result["sum_valid"] is True
        assert result["all_valid"] is True

    def test_custom_weight_sum(self):
        weights = pd.Series([0.40, 0.40])

        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.50,
            weight_sum=0.80,
        )

        result = validate_portfolio_weights(weights, constraints)

        assert result["sum_valid"] is True
        assert result["all_valid"] is True

    def test_short_selling_weights(self):
        weights = pd.Series([0.80, 0.30, -0.10])

        constraints = PortfolioConstraints(
            min_weight=-0.20,
            max_weight=1.0,
            allow_short_selling=True,
        )

        result = validate_portfolio_weights(weights, constraints)

        assert result["lower_bound_valid"] is True
        assert result["all_valid"] is True

    def test_duplicate_asset_labels_rejected(self):
        weights = pd.Series(
            [0.50, 0.50],
            index=["AAPL", "AAPL"],
        )

        with pytest.raises(ValueError, match="labels must be unique"):
            validate_portfolio_weights(weights)

    def test_empty_weights_rejected(self):
        weights = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_portfolio_weights(weights)

    @pytest.mark.parametrize(
        "weights",
        [
            [0.50, np.nan, 0.50],
            [0.50, np.inf, 0.50],
            [0.50, -np.inf, 0.50],
        ],
    )
    def test_non_finite_weights_rejected(self, weights):
        series = pd.Series(weights)

        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_portfolio_weights(series)

    def test_non_series_input_rejected(self):
        with pytest.raises(TypeError, match="pandas Series"):
            validate_portfolio_weights([0.50, 0.50])

    def test_zero_tolerance_rejected(self, valid_weights):
        with pytest.raises(ValueError, match="tolerance must be positive"):
            validate_portfolio_weights(valid_weights, tolerance=0.0)

    def test_negative_tolerance_rejected(self, valid_weights):
        with pytest.raises(ValueError, match="tolerance must be positive"):
            validate_portfolio_weights(valid_weights, tolerance=-1e-6)

    def test_tolerance_allows_small_weight_sum_difference(self):
        weights = pd.Series([0.50, 0.4999995])

        result = validate_portfolio_weights(
            weights,
            tolerance=1e-6,
        )

        assert result["sum_valid"] is True

    def test_tolerance_detects_large_weight_sum_difference(self):
        weights = pd.Series([0.50, 0.499])

        result = validate_portfolio_weights(
            weights,
            tolerance=1e-6,
        )

        assert result["sum_valid"] is False

    def test_validation_success_message(self, valid_weights):
        result = validate_portfolio_weights(valid_weights)

        assert result["message"] == (
            "All portfolio weight constraints satisfied."
        )

    def test_validation_failure_message(self):
        weights = pd.Series([0.60, 0.30, 0.10])

        result = validate_portfolio_weights(weights)

        assert result["message"] == (
            "One or more portfolio weight constraints failed."
        )
