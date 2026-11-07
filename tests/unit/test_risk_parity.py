
"""Unit tests for equal risk contribution portfolio optimization."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantfin.optimization.risk_parity import (
    _validate_returns,
    calculate_risk_contributions,
    equal_risk_contribution_weights,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Create reproducible positive-definite return data."""

    rng = np.random.default_rng(42)

    observations = 500
    assets = ["AAPL", "MSFT", "JPM", "KO"]

    common_factor = rng.normal(0.0003, 0.008, observations)

    returns = {}

    for index, asset in enumerate(assets):
        idiosyncratic = rng.normal(
            0.0002,
            0.006 + index * 0.001,
            observations,
        )

        returns[asset] = (
            common_factor * (0.5 + index * 0.1)
            + idiosyncratic
        )

    return pd.DataFrame(returns)


@pytest.fixture
def sample_covariance() -> pd.DataFrame:
    """Create a symmetric positive-definite covariance matrix."""

    values = np.array(
        [
            [0.040, 0.010, 0.005],
            [0.010, 0.090, 0.015],
            [0.005, 0.015, 0.160],
        ]
    )

    assets = ["AAPL", "MSFT", "JPM"]

    return pd.DataFrame(
        values,
        index=assets,
        columns=assets,
    )


@pytest.fixture
def sample_weights() -> pd.Series:
    """Create valid long-only portfolio weights."""

    return pd.Series(
        [0.40, 0.35, 0.25],
        index=["AAPL", "MSFT", "JPM"],
    )


# ---------------------------------------------------------------------
# Historical return validation
# ---------------------------------------------------------------------


def test_validate_returns_accepts_valid_data(sample_returns):
    result = _validate_returns(sample_returns)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_returns.shape
    assert result.index.equals(sample_returns.index)
    assert result.columns.equals(sample_returns.columns)


def test_validate_returns_converts_numeric_strings():
    returns = pd.DataFrame(
        {
            "AAPL": ["0.01", "0.02", "0.03"],
            "MSFT": ["0.02", "0.01", "0.04"],
        }
    )

    result = _validate_returns(returns)

    assert result.dtypes.eq(float).all()
    assert result.shape == (3, 2)


def test_validate_returns_drops_incomplete_rows():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, np.nan, 0.03, 0.04],
            "MSFT": [0.02, 0.03, 0.01, 0.02],
        }
    )

    result = _validate_returns(returns)

    assert len(result) == 3
    assert result.isna().sum().sum() == 0


def test_validate_returns_drops_infinite_rows():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, np.inf, 0.03, 0.04],
            "MSFT": [0.02, 0.03, 0.01, 0.02],
        }
    )

    result = _validate_returns(returns)

    assert len(result) == 3
    assert np.isfinite(result.to_numpy()).all()


def test_validate_returns_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        _validate_returns(np.array([[0.01, 0.02], [0.03, 0.04]]))


def test_validate_returns_rejects_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_returns(pd.DataFrame())


def test_validate_returns_rejects_duplicate_assets():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04], [0.02, 0.01]],
        columns=["AAPL", "AAPL"],
    )

    with pytest.raises(ValueError, match="unique"):
        _validate_returns(returns)


def test_validate_returns_requires_two_complete_observations():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, np.nan],
            "MSFT": [0.02, np.nan],
        }
    )

    with pytest.raises(ValueError, match="two complete observations"):
        _validate_returns(returns)


def test_validate_returns_requires_two_assets():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
        }
    )

    with pytest.raises(ValueError, match="at least two assets"):
        _validate_returns(returns)


def test_validate_returns_rejects_returns_at_or_below_minus_one():
    returns = pd.DataFrame(
        {
            "AAPL": [-1.0, 0.02, 0.03],
            "MSFT": [0.02, 0.03, 0.01],
        }
    )

    with pytest.raises(ValueError, match="less than or equal to -100%"):
        _validate_returns(returns)


def test_validate_returns_rejects_zero_volatility():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.01, 0.01, 0.01],
            "MSFT": [0.02, 0.03, 0.01, 0.04],
        }
    )

    with pytest.raises(ValueError, match="positive volatility"):
        _validate_returns(returns)


def test_validate_returns_coerces_invalid_strings():
    returns = pd.DataFrame(
        {
            "AAPL": ["0.01", "invalid", "0.03", "0.04"],
            "MSFT": ["0.02", "0.03", "0.01", "0.02"],
        }
    )

    result = _validate_returns(returns)

    assert len(result) == 3
    assert np.isfinite(result.to_numpy()).all()


# ---------------------------------------------------------------------
# Risk contribution calculations
# ---------------------------------------------------------------------


def test_calculate_risk_contributions_returns_dataframe(
    sample_weights,
    sample_covariance,
):
    result = calculate_risk_contributions(
        sample_weights,
        sample_covariance,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(sample_weights.index)

    expected_columns = {
        "weight",
        "marginal_risk",
        "risk_contribution",
        "risk_contribution_pct",
    }

    assert set(result.columns) == expected_columns


def test_risk_contributions_are_finite(
    sample_weights,
    sample_covariance,
):
    result = calculate_risk_contributions(
        sample_weights,
        sample_covariance,
    )

    assert np.isfinite(result.to_numpy()).all()


def test_risk_contribution_percentages_sum_to_one(
    sample_weights,
    sample_covariance,
):
    result = calculate_risk_contributions(
        sample_weights,
        sample_covariance,
    )

    assert np.isclose(
        result["risk_contribution_pct"].sum(),
        1.0,
        atol=1e-10,
    )


def test_risk_contributions_reproduce_portfolio_volatility(
    sample_weights,
    sample_covariance,
):
    result = calculate_risk_contributions(
        sample_weights,
        sample_covariance,
    )

    weights = sample_weights.to_numpy()
    covariance = sample_covariance.to_numpy()

    expected_volatility = np.sqrt(
        weights @ covariance @ weights
    )

    actual_volatility = result["risk_contribution"].sum()

    assert np.isclose(
        actual_volatility,
        expected_volatility,
        atol=1e-10,
    )


def test_risk_contribution_is_weight_times_marginal_risk(
    sample_weights,
    sample_covariance,
):
    result = calculate_risk_contributions(
        sample_weights,
        sample_covariance,
    )

    expected = (
        result["weight"] * result["marginal_risk"]
    )

    pd.testing.assert_series_equal(
        result["risk_contribution"],
        expected,
        check_names=False,
    )


def test_risk_contributions_reorder_covariance_labels(
    sample_weights,
    sample_covariance,
):
    reordered = sample_covariance.loc[
        ["JPM", "AAPL", "MSFT"],
        ["JPM", "AAPL", "MSFT"],
    ]

    original_result = calculate_risk_contributions(
        sample_weights,
        sample_covariance,
    )

    reordered_result = calculate_risk_contributions(
        sample_weights,
        reordered,
    )

    pd.testing.assert_frame_equal(
        original_result,
        reordered_result,
    )


def test_risk_contributions_reject_non_series_weights(
    sample_covariance,
):
    with pytest.raises(TypeError, match="pandas Series"):
        calculate_risk_contributions(
            [0.4, 0.35, 0.25],
            sample_covariance,
        )


def test_risk_contributions_reject_non_dataframe_covariance(
    sample_weights,
):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        calculate_risk_contributions(
            sample_weights,
            np.eye(3),
        )


def test_risk_contributions_reject_empty_weights(
    sample_covariance,
):
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_risk_contributions(
            pd.Series(dtype=float),
            sample_covariance,
        )


def test_risk_contributions_reject_empty_covariance(
    sample_weights,
):
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_risk_contributions(
            sample_weights,
            pd.DataFrame(),
        )


def test_risk_contributions_reject_duplicate_weight_labels(
    sample_covariance,
):
    weights = pd.Series(
        [0.4, 0.35, 0.25],
        index=["AAPL", "AAPL", "JPM"],
    )

    with pytest.raises(ValueError, match="unique"):
        calculate_risk_contributions(
            weights,
            sample_covariance,
        )


def test_risk_contributions_reject_duplicate_covariance_labels(
    sample_weights,
    sample_covariance,
):
    covariance = sample_covariance.copy()
    covariance.index = ["AAPL", "AAPL", "JPM"]

    with pytest.raises(ValueError, match="unique"):
        calculate_risk_contributions(
            sample_weights,
            covariance,
        )


def test_risk_contributions_reject_mismatched_assets(
    sample_weights,
    sample_covariance,
):
    weights = sample_weights.rename(
        index={"JPM": "KO"}
    )

    with pytest.raises(ValueError, match="must match"):
        calculate_risk_contributions(
            weights,
            sample_covariance,
        )


def test_risk_contributions_reject_nonfinite_weights(
    sample_covariance,
):
    weights = pd.Series(
        [0.4, np.nan, 0.25],
        index=["AAPL", "MSFT", "JPM"],
    )

    with pytest.raises(ValueError, match="finite"):
        calculate_risk_contributions(
            weights,
            sample_covariance,
        )


def test_risk_contributions_reject_nonfinite_covariance(
    sample_weights,
    sample_covariance,
):
    covariance = sample_covariance.copy()
    covariance.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        calculate_risk_contributions(
            sample_weights,
            covariance,
        )


def test_risk_contributions_reject_short_positions(
    sample_covariance,
):
    weights = pd.Series(
        [0.50, -0.10, 0.60],
        index=["AAPL", "MSFT", "JPM"],
    )

    with pytest.raises(ValueError, match="long-only"):
        calculate_risk_contributions(
            weights,
            sample_covariance,
        )


def test_risk_contributions_reject_asymmetric_covariance(
    sample_weights,
    sample_covariance,
):
    covariance = sample_covariance.copy()
    covariance.iloc[0, 1] = 0.50

    with pytest.raises(ValueError, match="symmetric"):
        calculate_risk_contributions(
            sample_weights,
            covariance,
        )


def test_risk_contributions_reject_zero_portfolio_variance(
    sample_weights,
):
    covariance = pd.DataFrame(
        np.zeros((3, 3)),
        index=sample_weights.index,
        columns=sample_weights.index,
    )

    with pytest.raises(ValueError, match="variance must be positive"):
        calculate_risk_contributions(
            sample_weights,
            covariance,
        )


# ---------------------------------------------------------------------
# Risk parity optimization
# ---------------------------------------------------------------------


def test_risk_parity_returns_series(sample_returns):
    weights = equal_risk_contribution_weights(sample_returns)

    assert isinstance(weights, pd.Series)
    assert weights.name == "risk_parity_weight"


def test_risk_parity_weights_sum_to_one(sample_returns):
    weights = equal_risk_contribution_weights(sample_returns)

    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_risk_parity_weights_are_nonnegative(sample_returns):
    weights = equal_risk_contribution_weights(sample_returns)

    assert (weights >= 0).all()


def test_risk_parity_respects_maximum_weight(sample_returns):
    maximum = 0.40

    weights = equal_risk_contribution_weights(
        sample_returns,
        max_weight=maximum,
    )

    assert (weights <= maximum + 1e-7).all()


def test_risk_parity_preserves_asset_labels(sample_returns):
    weights = equal_risk_contribution_weights(sample_returns)

    assert weights.index.equals(sample_returns.columns)


def test_risk_parity_weights_are_finite(sample_returns):
    weights = equal_risk_contribution_weights(sample_returns)

    assert np.isfinite(weights.to_numpy()).all()


def test_risk_parity_has_equal_risk_contributions(sample_returns):
    weights = equal_risk_contribution_weights(sample_returns)

    covariance = sample_returns.cov()

    contributions = calculate_risk_contributions(
        weights,
        covariance,
    )

    contribution_percentages = contributions[
        "risk_contribution_pct"
    ]

    assert contribution_percentages.max() - contribution_percentages.min() < 1e-3


def test_risk_parity_respects_tighter_weight_limit(sample_returns):
    maximum = 0.30

    weights = equal_risk_contribution_weights(
        sample_returns,
        max_weight=maximum,
    )

    assert weights.max() <= maximum + 1e-7
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_risk_parity_accepts_maximum_weight_one(sample_returns):
    weights = equal_risk_contribution_weights(
        sample_returns,
        max_weight=1.0,
    )

    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_risk_parity_rejects_zero_maximum_weight(sample_returns):
    with pytest.raises(ValueError, match="max_weight must be"):
        equal_risk_contribution_weights(
            sample_returns,
            max_weight=0.0,
        )


def test_risk_parity_rejects_negative_maximum_weight(sample_returns):
    with pytest.raises(ValueError, match="max_weight must be"):
        equal_risk_contribution_weights(
            sample_returns,
            max_weight=-0.1,
        )


def test_risk_parity_rejects_maximum_weight_above_one(sample_returns):
    with pytest.raises(ValueError, match="max_weight must be"):
        equal_risk_contribution_weights(
            sample_returns,
            max_weight=1.1,
        )


def test_risk_parity_rejects_infeasible_weight_limit(sample_returns):
    with pytest.raises(ValueError, match="too restrictive"):
        equal_risk_contribution_weights(
            sample_returns,
            max_weight=0.20,
        )


def test_risk_parity_rejects_zero_tolerance(sample_returns):
    with pytest.raises(ValueError, match="tolerance must be positive"):
        equal_risk_contribution_weights(
            sample_returns,
            tolerance=0,
        )


def test_risk_parity_rejects_negative_tolerance(sample_returns):
    with pytest.raises(ValueError, match="tolerance must be positive"):
        equal_risk_contribution_weights(
            sample_returns,
            tolerance=-1e-5,
        )


def test_risk_parity_rejects_zero_iterations(sample_returns):
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        equal_risk_contribution_weights(
            sample_returns,
            max_iterations=0,
        )


def test_risk_parity_rejects_negative_iterations(sample_returns):
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        equal_risk_contribution_weights(
            sample_returns,
            max_iterations=-10,
        )


def test_risk_parity_rejects_non_positive_definite_covariance(
    sample_returns,
    monkeypatch,
):
    """Reject a covariance matrix with a zero eigenvalue."""

    monkeypatch.setattr(
        "quantfin.optimization.risk_parity.np.linalg.eigvalsh",
        lambda covariance: np.array([0.0, 0.01, 0.02, 0.03]),
    )

    with pytest.raises(ValueError, match="positive definite"):
        equal_risk_contribution_weights(sample_returns)



def test_risk_parity_rejects_nonfinite_covariance():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, np.nan, 0.03],
            "MSFT": [0.02, 0.01, 0.03, 0.04],
        }
    )

    # Complete observations still leave a valid covariance matrix.
    weights = equal_risk_contribution_weights(returns)

    assert np.isfinite(weights.to_numpy()).all()


def test_risk_parity_raises_when_optimizer_fails(sample_returns):
    failed_result = type(
        "OptimizationResult",
        (),
        {
            "success": False,
            "fun": 1.0,
            "x": np.full(sample_returns.shape[1], 1 / sample_returns.shape[1]),
        },
    )()

    with patch(
        "quantfin.optimization.risk_parity.minimize",
        return_value=failed_result,
    ):
        with pytest.raises(RuntimeError, match="failed to converge"):
            equal_risk_contribution_weights(sample_returns)


def test_risk_parity_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        equal_risk_contribution_weights(
            np.array([[0.01, 0.02], [0.02, 0.03]])
        )


def test_risk_parity_is_reproducible(sample_returns):
    first = equal_risk_contribution_weights(sample_returns)
    second = equal_risk_contribution_weights(sample_returns)

    pd.testing.assert_series_equal(first, second)
