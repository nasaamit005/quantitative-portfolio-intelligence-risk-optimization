
"""Unit tests for expected-return and covariance estimators."""

import numpy as np
import pandas as pd
import pytest

from quantfin.estimators.returns import (
    _validate_returns,
    estimate_covariance,
    estimate_expected_returns,
    estimate_market_inputs,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Create deterministic daily returns for three assets."""
    return pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "MSFT": [0.02, -0.01, 0.02, 0.03, -0.02, 0.01],
            "JPM": [0.01, 0.01, -0.01, 0.02, 0.01, -0.02],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="B"),
    )


def test_validate_returns_preserves_asset_columns(sample_returns):
    result = _validate_returns(sample_returns)

    assert list(result.columns) == ["AAPL", "MSFT", "JPM"]
    assert result.shape == sample_returns.shape
    assert result.index.equals(sample_returns.index)


def test_validate_returns_converts_values_to_float(sample_returns):
    result = _validate_returns(sample_returns)

    assert all(
        np.issubdtype(dtype, np.floating)
        for dtype in result.dtypes
    )


def test_validate_returns_removes_incomplete_observations(sample_returns):
    data = sample_returns.copy()
    data.loc[data.index[0], "AAPL"] = np.nan

    result = _validate_returns(data)

    assert len(result) == len(data) - 1
    assert result.isna().sum().sum() == 0


def test_validate_returns_removes_infinite_observations(sample_returns):
    data = sample_returns.copy()
    data.loc[data.index[0], "MSFT"] = np.inf

    result = _validate_returns(data)

    assert len(result) == len(data) - 1
    assert np.isfinite(result.to_numpy()).all()


def test_validate_returns_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        _validate_returns(np.array([[0.01, 0.02]]))


def test_validate_returns_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        _validate_returns(pd.DataFrame())


def test_validate_returns_rejects_duplicate_columns():
    data = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04]],
        columns=["AAPL", "AAPL"],
    )

    with pytest.raises(ValueError, match="unique"):
        _validate_returns(data)


def test_validate_returns_rejects_insufficient_observations():
    data = pd.DataFrame({"AAPL": [0.01]})

    with pytest.raises(ValueError, match="At least two"):
        _validate_returns(data)


def test_validate_returns_rejects_all_invalid_observations():
    data = pd.DataFrame(
        {
            "AAPL": [np.nan, np.inf],
            "MSFT": [np.nan, -np.inf],
        }
    )

    with pytest.raises(ValueError, match="No complete finite"):
        _validate_returns(data)


@pytest.mark.parametrize("invalid_return", [-1.0, -1.05])
def test_validate_returns_rejects_returns_at_or_below_negative_one(
    sample_returns,
    invalid_return,
):
    data = sample_returns.copy()
    data.iloc[0, 0] = invalid_return

    with pytest.raises(ValueError, match="less than or equal"):
        _validate_returns(data)


def test_estimate_expected_returns_annualizes_correctly(sample_returns):
    result = estimate_expected_returns(
        sample_returns,
        periods_per_year=252,
    )

    expected = sample_returns.mean() * 252
    expected.name = "expected_annual_return"

    pd.testing.assert_series_equal(result, expected)


def test_estimate_expected_returns_uses_custom_annualization(
    sample_returns,
):
    result = estimate_expected_returns(
        sample_returns,
        periods_per_year=12,
    )

    expected = sample_returns.mean() * 12
    expected.name = "expected_annual_return"

    pd.testing.assert_series_equal(result, expected)


def test_estimate_expected_returns_has_correct_name(sample_returns):
    result = estimate_expected_returns(sample_returns)

    assert result.name == "expected_annual_return"


def test_estimate_expected_returns_rejects_invalid_periods(
    sample_returns,
):
    for periods in [0, -1]:
        with pytest.raises(ValueError, match="positive"):
            estimate_expected_returns(
                sample_returns,
                periods_per_year=periods,
            )


def test_estimate_expected_returns_rejects_unsupported_method(
    sample_returns,
):
    with pytest.raises(ValueError, match="Unsupported"):
        estimate_expected_returns(
            sample_returns,
            method="unsupported",
        )


def test_estimate_sample_covariance_matches_pandas(sample_returns):
    result = estimate_covariance(
        sample_returns,
        method="sample",
        periods_per_year=252,
    )

    expected = sample_returns.cov() * 252
    expected.index.name = "asset"
    expected.columns.name = "asset"

    pd.testing.assert_frame_equal(result, expected)


def test_estimate_sample_covariance_is_symmetric(sample_returns):
    result = estimate_covariance(
        sample_returns,
        method="sample",
    )

    assert np.allclose(
        result.to_numpy(),
        result.to_numpy().T,
        atol=1e-12,
    )


def test_estimate_sample_covariance_has_correct_labels(sample_returns):
    result = estimate_covariance(
        sample_returns,
        method="sample",
    )

    assert list(result.index) == list(sample_returns.columns)
    assert list(result.columns) == list(sample_returns.columns)
    assert result.index.name == "asset"
    assert result.columns.name == "asset"


def test_estimate_sample_covariance_uses_custom_annualization(
    sample_returns,
):
    result = estimate_covariance(
        sample_returns,
        method="sample",
        periods_per_year=12,
    )

    expected = sample_returns.cov() * 12
    expected.index.name = "asset"
    expected.columns.name = "asset"

    pd.testing.assert_frame_equal(result, expected)


def test_estimate_ledoit_wolf_covariance_is_symmetric(sample_returns):
    result = estimate_covariance(
        sample_returns,
        method="ledoit_wolf",
    )

    assert np.allclose(
        result.to_numpy(),
        result.to_numpy().T,
        atol=1e-12,
    )


def test_estimate_ledoit_wolf_covariance_is_positive_definite(
    sample_returns,
):
    result = estimate_covariance(
        sample_returns,
        method="ledoit_wolf",
    )

    eigenvalues = np.linalg.eigvalsh(result.to_numpy())

    assert np.all(eigenvalues > 0)


def test_estimate_ledoit_wolf_covariance_has_correct_labels(
    sample_returns,
):
    result = estimate_covariance(
        sample_returns,
        method="ledoit_wolf",
    )

    assert list(result.index) == list(sample_returns.columns)
    assert list(result.columns) == list(sample_returns.columns)
    assert result.index.name == "asset"
    assert result.columns.name == "asset"


def test_estimate_covariance_rejects_invalid_periods(sample_returns):
    for periods in [0, -1]:
        with pytest.raises(ValueError, match="positive"):
            estimate_covariance(
                sample_returns,
                periods_per_year=periods,
            )


def test_estimate_covariance_rejects_unsupported_method(sample_returns):
    with pytest.raises(ValueError, match="Unsupported"):
        estimate_covariance(
            sample_returns,
            method="unsupported",
        )


def test_estimate_market_inputs_returns_aligned_outputs(sample_returns):
    expected_returns, covariance = estimate_market_inputs(
        sample_returns,
        return_method="historical_mean",
        covariance_method="ledoit_wolf",
    )

    assert list(expected_returns.index) == list(sample_returns.columns)
    assert list(covariance.index) == list(sample_returns.columns)
    assert list(covariance.columns) == list(sample_returns.columns)


def test_estimate_market_inputs_returns_finite_values(sample_returns):
    expected_returns, covariance = estimate_market_inputs(
        sample_returns,
    )

    assert np.isfinite(expected_returns.to_numpy()).all()
    assert np.isfinite(covariance.to_numpy()).all()


def test_estimate_market_inputs_covariance_is_symmetric(sample_returns):
    _, covariance = estimate_market_inputs(sample_returns)

    assert np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
        atol=1e-10,
    )


def test_estimate_market_inputs_covariance_is_positive_definite(
    sample_returns,
):
    _, covariance = estimate_market_inputs(sample_returns)

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())

    assert np.all(eigenvalues > 0)


def test_estimate_market_inputs_supports_sample_covariance(sample_returns):
    expected_returns, covariance = estimate_market_inputs(
        sample_returns,
        covariance_method="sample",
    )

    assert expected_returns.shape == (3,)
    assert covariance.shape == (3, 3)
    assert np.isfinite(covariance.to_numpy()).all()


def test_estimate_market_inputs_rejects_invalid_covariance_method(
    sample_returns,
):
    with pytest.raises(ValueError, match="Unsupported"):
        estimate_market_inputs(
            sample_returns,
            covariance_method="unsupported",
        )
