
"""Unit tests for correlation, covariance and diversification analytics."""

import numpy as np
import pandas as pd
import pytest

from quantfin.analytics.correlation import (
    calculate_correlation_matrices,
    calculate_correlation_stability,
    calculate_covariance_matrices,
    calculate_diversification_metrics,
    calculate_matrix_diagnostics,
    calculate_rolling_correlations,
)


def make_returns(observations: int = 100) -> pd.DataFrame:
    """Create reproducible multi-asset daily returns."""
    rng = np.random.default_rng(42)

    market = rng.normal(0.0005, 0.01, observations)
    noise_a = rng.normal(0, 0.005, observations)
    noise_b = rng.normal(0, 0.007, observations)

    return pd.DataFrame(
        {
            "SPY": market,
            "AAPL": 1.2 * market + noise_a,
            "MSFT": 0.9 * market + noise_b,
        },
        index=pd.date_range("2025-01-01", periods=observations),
    )


def test_correlation_matrices_have_correct_shape_and_labels():
    """Pearson and Spearman matrices must retain asset labels."""
    returns = make_returns()

    pearson, spearman = calculate_correlation_matrices(returns)

    assert pearson.shape == (3, 3)
    assert spearman.shape == (3, 3)
    assert pearson.index.equals(returns.columns)
    assert pearson.columns.equals(returns.columns)
    assert spearman.index.equals(returns.columns)
    assert spearman.columns.equals(returns.columns)


def test_correlation_matrices_are_symmetric():
    """Correlation matrices must be symmetric."""
    pearson, spearman = calculate_correlation_matrices(make_returns())

    np.testing.assert_allclose(
        pearson.to_numpy(),
        pearson.to_numpy().T,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        spearman.to_numpy(),
        spearman.to_numpy().T,
        atol=1e-12,
    )


def test_correlation_diagonal_equals_one():
    """Each asset must have unit self-correlation."""
    pearson, spearman = calculate_correlation_matrices(make_returns())

    np.testing.assert_allclose(np.diag(pearson), 1.0)
    np.testing.assert_allclose(np.diag(spearman), 1.0)


def test_covariance_matrices_are_annualized_correctly():
    """Annualized sample covariance must equal daily covariance times 252."""
    returns = make_returns()

    sample, annualized, shrinkage = calculate_covariance_matrices(
        returns
    )

    np.testing.assert_allclose(
        annualized.to_numpy(),
        sample.to_numpy() * 252,
        rtol=1e-12,
        atol=1e-12,
    )

    assert sample.shape == (3, 3)
    assert shrinkage.shape == (3, 3)


def test_ledoit_wolf_covariance_is_symmetric():
    """Ledoit-Wolf covariance must be symmetric."""
    _, _, shrinkage = calculate_covariance_matrices(make_returns())

    np.testing.assert_allclose(
        shrinkage.to_numpy(),
        shrinkage.to_numpy().T,
        atol=1e-12,
    )


def test_ledoit_wolf_covariance_is_positive_definite():
    """Ledoit-Wolf covariance should be positive definite for this dataset."""
    _, _, shrinkage = calculate_covariance_matrices(make_returns())

    eigenvalues = np.linalg.eigvalsh(shrinkage.to_numpy())

    assert np.isfinite(eigenvalues).all()
    assert eigenvalues.min() > 0


def test_diversification_metrics_have_expected_fields():
    """Check diversification output structure and basic properties."""
    metrics = calculate_diversification_metrics(make_returns())

    expected_fields = {
        "instrument_count",
        "observations",
        "equal_weight_portfolio_volatility",
        "weighted_average_asset_volatility",
        "diversification_ratio",
        "average_pairwise_correlation",
    }

    assert set(metrics) == expected_fields
    assert metrics["instrument_count"] == 3
    assert metrics["observations"] == 100
    assert metrics["equal_weight_portfolio_volatility"] >= 0
    assert metrics["weighted_average_asset_volatility"] >= 0
    assert metrics["diversification_ratio"] > 0
    assert -1 <= metrics["average_pairwise_correlation"] <= 1


def test_diversification_ratio_uses_weighted_average_volatility():
    """Verify the diversification ratio formula."""
    returns = make_returns()
    metrics = calculate_diversification_metrics(returns)

    expected = (
        metrics["weighted_average_asset_volatility"]
        / metrics["equal_weight_portfolio_volatility"]
    )

    assert metrics["diversification_ratio"] == pytest.approx(expected)


def test_matrix_diagnostics_identify_positive_definite_matrix():
    """A symmetric positive-definite matrix should pass diagnostics."""
    matrix = pd.DataFrame(
        [[2.0, 0.5], [0.5, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = calculate_matrix_diagnostics(matrix, "test_matrix")

    assert result["matrix"] == "test_matrix"
    assert result["dimension"] == 2
    assert result["symmetric"] is True
    assert result["positive_definite"] is True
    assert result["minimum_eigenvalue"] > 0
    assert result["finite_values"] is True
    assert result["condition_number"] >= 1


def test_matrix_diagnostics_identify_non_positive_definite_matrix():
    """A symmetric indefinite matrix should fail the PD diagnostic."""
    matrix = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = calculate_matrix_diagnostics(matrix, "indefinite")

    assert result["symmetric"] is True
    assert result["positive_definite"] is False
    assert result["minimum_eigenvalue"] < 0


def test_rolling_correlations_have_expected_warmup_period():
    """Rolling correlations should be missing until 63 observations exist."""
    returns = make_returns(80)

    rolling = calculate_rolling_correlations(returns)

    assert "AAPL_vs_SPY" in rolling.columns
    assert "MSFT_vs_SPY" in rolling.columns
    assert rolling.shape == (80, 2)

    assert rolling.iloc[:62].isna().all().all()
    assert rolling.iloc[62:].notna().all().all()


def test_rolling_correlations_skip_spy_column():
    """SPY should not be correlated against itself in the output."""
    rolling = calculate_rolling_correlations(make_returns())

    assert "SPY_vs_SPY" not in rolling.columns


def test_rolling_correlations_without_spy_return_empty_columns():
    """Without SPY, the function should return an empty-column DataFrame."""
    returns = make_returns().drop(columns=["SPY"])

    rolling = calculate_rolling_correlations(returns)

    assert rolling.empty
    assert rolling.index.equals(returns.index)


def test_correlation_stability_summary():
    """Check summary statistics for rolling correlations."""
    returns = make_returns(100)
    rolling = calculate_rolling_correlations(returns)

    stability = calculate_correlation_stability(rolling)

    assert set(stability.index) == {"AAPL_vs_SPY", "MSFT_vs_SPY"}
    assert (stability["observations"] == 38).all()

    assert stability["minimum_rolling_correlation"].between(-1, 1).all()
    assert stability["maximum_rolling_correlation"].between(-1, 1).all()
    assert stability["latest_rolling_correlation"].between(-1, 1).all()


def test_correlation_stability_handles_empty_input():
    """An empty rolling-correlation DataFrame should return an empty result."""
    rolling = pd.DataFrame()

    stability = calculate_correlation_stability(rolling)

    assert stability.empty
