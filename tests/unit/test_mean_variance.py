
"""Unit tests for mean-variance portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from quantfin.optimization.constraints import PortfolioConstraints
from quantfin.optimization.mean_variance import (
    efficient_frontier,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    portfolio_performance,
)


@pytest.fixture
def expected_returns() -> pd.Series:
    """Provide annualized expected returns for four assets."""

    return pd.Series(
        [0.08, 0.12, 0.16, 0.20],
        index=["A", "B", "C", "D"],
        dtype=float,
    )


@pytest.fixture
def covariance() -> pd.DataFrame:
    """Provide a positive-definite annualized covariance matrix."""

    values = np.array(
        [
            [0.0400, 0.0060, 0.0080, 0.0040],
            [0.0060, 0.0625, 0.0100, 0.0075],
            [0.0080, 0.0100, 0.0900, 0.0120],
            [0.0040, 0.0075, 0.0120, 0.1225],
        ],
        dtype=float,
    )

    assets = ["A", "B", "C", "D"]

    return pd.DataFrame(values, index=assets, columns=assets)


@pytest.fixture
def constraints() -> PortfolioConstraints:
    """Provide standard long-only portfolio constraints."""

    return PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.50,
        allow_short_selling=False,
        fully_invested=True,
        weight_sum=1.0,
    )


def assert_valid_weights(
    weights: pd.Series,
    constraints: PortfolioConstraints,
) -> None:
    """Check portfolio weights against the supplied constraints."""

    assert isinstance(weights, pd.Series)
    assert not weights.empty
    assert weights.index.is_unique
    assert np.isfinite(weights.to_numpy()).all()

    assert weights.sum() == pytest.approx(
        constraints.weight_sum,
        abs=1e-6,
    )

    assert weights.min() >= constraints.min_weight - 1e-6
    assert weights.max() <= constraints.max_weight + 1e-6


# ---------------------------------------------------------------------
# Minimum-volatility optimization
# ---------------------------------------------------------------------


def test_minimum_volatility_returns_valid_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> None:
    weights = minimum_volatility_weights(
        expected_returns,
        covariance,
        constraints,
    )

    assert_valid_weights(weights, constraints)
    assert list(weights.index) == list(expected_returns.index)


def test_minimum_volatility_weights_have_correct_name(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = minimum_volatility_weights(
        expected_returns,
        covariance,
    )

    assert weights.name == "weight"


def test_minimum_volatility_weights_are_deterministic(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    first = minimum_volatility_weights(
        expected_returns,
        covariance,
    )

    second = minimum_volatility_weights(
        expected_returns,
        covariance,
    )

    np.testing.assert_allclose(
        first.to_numpy(),
        second.to_numpy(),
        atol=1e-6,
    )


def test_minimum_volatility_accepts_reordered_covariance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    reordered = covariance.loc[
        ["D", "C", "B", "A"],
        ["D", "C", "B", "A"],
    ]

    weights = minimum_volatility_weights(
        expected_returns,
        reordered,
    )

    assert list(weights.index) == list(expected_returns.index)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_minimum_volatility_respects_custom_constraints(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    rules = PortfolioConstraints(
        min_weight=0.05,
        max_weight=0.40,
    )

    weights = minimum_volatility_weights(
        expected_returns,
        covariance,
        rules,
    )

    assert_valid_weights(weights, rules)


# ---------------------------------------------------------------------
# Maximum-Sharpe optimization
# ---------------------------------------------------------------------


def test_maximum_sharpe_returns_valid_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> None:
    weights = maximum_sharpe_weights(
        expected_returns,
        covariance,
        risk_free_rate=0.02,
        constraints=constraints,
    )

    assert_valid_weights(weights, constraints)
    assert list(weights.index) == list(expected_returns.index)


def test_maximum_sharpe_weights_have_correct_name(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = maximum_sharpe_weights(
        expected_returns,
        covariance,
        risk_free_rate=0.02,
    )

    assert weights.name == "weight"


def test_maximum_sharpe_weights_are_deterministic(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    first = maximum_sharpe_weights(
        expected_returns,
        covariance,
        risk_free_rate=0.02,
    )

    second = maximum_sharpe_weights(
        expected_returns,
        covariance,
        risk_free_rate=0.02,
    )

    np.testing.assert_allclose(
        first.to_numpy(),
        second.to_numpy(),
        atol=1e-6,
    )


def test_maximum_sharpe_accepts_custom_constraints(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    rules = PortfolioConstraints(
        min_weight=0.02,
        max_weight=0.45,
    )

    weights = maximum_sharpe_weights(
        expected_returns,
        covariance,
        risk_free_rate=0.02,
        constraints=rules,
    )

    assert_valid_weights(weights, rules)


def test_maximum_sharpe_accepts_reordered_covariance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    reordered = covariance.loc[
        ["C", "A", "D", "B"],
        ["C", "A", "D", "B"],
    ]

    weights = maximum_sharpe_weights(
        expected_returns,
        reordered,
        risk_free_rate=0.02,
    )

    assert list(weights.index) == list(expected_returns.index)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------


def test_efficient_frontier_returns_requested_number_of_points(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        points=10,
    )

    assert len(frontier) == 10


def test_efficient_frontier_contains_required_columns(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        points=5,
    )

    required_columns = {
        "target_return",
        "expected_annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "weight_A",
        "weight_B",
        "weight_C",
        "weight_D",
    }

    assert required_columns.issubset(frontier.columns)


def test_efficient_frontier_has_finite_results(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        points=8,
    )

    numeric_values = frontier.select_dtypes(
        include=[np.number]
    ).to_numpy()

    assert np.isfinite(numeric_values).all()


def test_efficient_frontier_target_returns_are_ordered(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        points=8,
    )

    targets = frontier["target_return"].to_numpy()

    assert np.all(np.diff(targets) >= -1e-10)


def test_efficient_frontier_weights_are_feasible(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> None:
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        constraints,
        points=8,
    )

    weight_columns = [
        f"weight_{asset}"
        for asset in expected_returns.index
    ]

    for _, row in frontier.iterrows():
        weights = row[weight_columns]

        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights.min() >= -1e-6
        assert weights.max() <= 0.50 + 1e-6


def test_efficient_frontier_returns_match_target_returns(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        points=8,
    )

    np.testing.assert_allclose(
        frontier["expected_annual_return"].to_numpy(),
        frontier["target_return"].to_numpy(),
        atol=1e-6,
    )


def test_efficient_frontier_requires_at_least_two_points(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least two frontier points",
    ):
        efficient_frontier(
            expected_returns,
            covariance,
            points=1,
        )


def test_efficient_frontier_rejects_non_fully_invested_constraints(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    rules = PortfolioConstraints(
        fully_invested=False,
    )

    with pytest.raises(
        ValueError,
        match="requires fully invested portfolios",
    ):
        efficient_frontier(
            expected_returns,
            covariance,
            rules,
        )


# ---------------------------------------------------------------------
# Portfolio performance calculations
# ---------------------------------------------------------------------


def test_portfolio_performance_calculates_expected_return(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = pd.Series(
        [0.25, 0.25, 0.25, 0.25],
        index=expected_returns.index,
    )

    result = portfolio_performance(
        weights,
        expected_returns,
        covariance,
        risk_free_rate=0.02,
    )

    expected_return = float(
        weights.to_numpy() @ expected_returns.to_numpy()
    )

    assert result["expected_annual_return"] == pytest.approx(
        expected_return
    )


def test_portfolio_performance_calculates_volatility(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = pd.Series(
        [0.25, 0.25, 0.25, 0.25],
        index=expected_returns.index,
    )

    result = portfolio_performance(
        weights,
        expected_returns,
        covariance,
    )

    expected_volatility = np.sqrt(
        weights.to_numpy()
        @ covariance.to_numpy()
        @ weights.to_numpy()
    )

    assert result["annual_volatility"] == pytest.approx(
        expected_volatility
    )


def test_portfolio_performance_calculates_sharpe_ratio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = pd.Series(
        [0.25, 0.25, 0.25, 0.25],
        index=expected_returns.index,
    )

    risk_free_rate = 0.02

    result = portfolio_performance(
        weights,
        expected_returns,
        covariance,
        risk_free_rate=risk_free_rate,
    )

    expected_sharpe = (
        result["expected_annual_return"] - risk_free_rate
    ) / result["annual_volatility"]

    assert result["sharpe_ratio"] == pytest.approx(
        expected_sharpe
    )


def test_portfolio_performance_reorders_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = pd.Series(
        [0.25, 0.25, 0.25, 0.25],
        index=["D", "C", "B", "A"],
    )

    result = portfolio_performance(
        weights,
        expected_returns,
        covariance,
    )

    assert np.isfinite(
        result["expected_annual_return"]
    )


def test_portfolio_performance_rejects_missing_asset_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = pd.Series(
        [0.30, 0.30, 0.40],
        index=["A", "B", "C"],
    )

    with pytest.raises(
        ValueError,
        match="misaligned",
    ):
        portfolio_performance(
            weights,
            expected_returns,
            covariance,
        )


def test_portfolio_performance_rejects_non_finite_risk_free_rate(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    weights = pd.Series(
        [0.25, 0.25, 0.25, 0.25],
        index=expected_returns.index,
    )

    with pytest.raises(
        ValueError,
        match="risk_free_rate must be finite",
    ):
        portfolio_performance(
            weights,
            expected_returns,
            covariance,
            risk_free_rate=np.nan,
        )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_rejects_expected_returns_that_are_not_a_series(
    covariance: pd.DataFrame,
) -> None:
    with pytest.raises(
        TypeError,
        match="expected_returns must be a pandas Series",
    ):
        minimum_volatility_weights(
            [0.08, 0.12, 0.16, 0.20],
            covariance,
        )


def test_rejects_covariance_that_is_not_a_dataframe(
    expected_returns: pd.Series,
) -> None:
    with pytest.raises(
        TypeError,
        match="covariance must be a pandas DataFrame",
    ):
        minimum_volatility_weights(
            expected_returns,
            np.eye(4),
        )


def test_rejects_empty_expected_returns(
    covariance: pd.DataFrame,
) -> None:
    empty_returns = pd.Series(dtype=float)

    with pytest.raises(
        ValueError,
        match="Expected returns cannot be empty",
    ):
        minimum_volatility_weights(
            empty_returns,
            covariance,
        )


def test_rejects_duplicate_asset_labels(
    covariance: pd.DataFrame,
) -> None:
    duplicate_returns = pd.Series(
        [0.08, 0.12, 0.16, 0.20],
        index=["A", "A", "C", "D"],
    )

    with pytest.raises(
        ValueError,
        match="Asset labels must be unique",
    ):
        minimum_volatility_weights(
            duplicate_returns,
            covariance,
        )


def test_rejects_mismatched_asset_labels(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    mismatched = expected_returns.rename(
        index={"D": "UNKNOWN"}
    )

    with pytest.raises(
        ValueError,
        match="identical assets",
    ):
        minimum_volatility_weights(
            mismatched,
            covariance,
        )


def test_rejects_non_finite_expected_returns(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    invalid_returns = expected_returns.copy()
    invalid_returns.iloc[0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite values",
    ):
        minimum_volatility_weights(
            invalid_returns,
            covariance,
        )


def test_rejects_non_finite_covariance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    invalid_covariance = covariance.copy()
    invalid_covariance.iloc[0, 0] = np.inf

    with pytest.raises(
        ValueError,
        match="non-finite values",
    ):
        minimum_volatility_weights(
            expected_returns,
            invalid_covariance,
        )


def test_rejects_asymmetric_covariance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    invalid_covariance = covariance.copy()
    invalid_covariance.iloc[0, 1] = 0.10

    with pytest.raises(
        ValueError,
        match="must be symmetric",
    ):
        minimum_volatility_weights(
            expected_returns,
            invalid_covariance,
        )


def test_rejects_non_positive_definite_covariance(
    expected_returns: pd.Series,
) -> None:
    invalid_covariance = pd.DataFrame(
        np.ones((4, 4)),
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(
        ValueError,
        match="positive definite",
    ):
        minimum_volatility_weights(
            expected_returns,
            invalid_covariance,
        )


@pytest.mark.parametrize(
    "risk_free_rate",
    [np.nan, np.inf, -np.inf],
)
def test_maximum_sharpe_rejects_non_finite_risk_free_rate(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="risk_free_rate must be finite",
    ):
        maximum_sharpe_weights(
            expected_returns,
            covariance,
            risk_free_rate=risk_free_rate,
        )
