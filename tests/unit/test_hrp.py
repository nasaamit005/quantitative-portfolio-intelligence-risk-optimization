
"""Unit tests for Hierarchical Risk Parity optimization."""

import numpy as np
import pandas as pd
import pytest

from quantfin.optimization.hrp import (
    _cluster_variance,
    _validate_returns,
    hierarchical_risk_parity_weights,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    """Create reproducible correlated simple-return observations."""

    rng = np.random.default_rng(42)
    observations = 500

    market_factor = rng.normal(0.0004, 0.012, observations)
    technology_factor = rng.normal(0.0003, 0.009, observations)
    defensive_factor = rng.normal(0.0002, 0.006, observations)

    data = {
        "AAPL": (
            0.75 * market_factor
            + 0.65 * technology_factor
            + rng.normal(0, 0.004, observations)
        ),
        "MSFT": (
            0.70 * market_factor
            + 0.70 * technology_factor
            + rng.normal(0, 0.004, observations)
        ),
        "JPM": (
            0.85 * market_factor
            + 0.30 * defensive_factor
            + rng.normal(0, 0.006, observations)
        ),
        "JNJ": (
            0.35 * market_factor
            + 0.85 * defensive_factor
            + rng.normal(0, 0.003, observations)
        ),
        "XOM": (
            0.65 * market_factor
            - 0.20 * defensive_factor
            + rng.normal(0, 0.008, observations)
        ),
    }

    return pd.DataFrame(data)


@pytest.fixture
def covariance() -> pd.DataFrame:
    """Provide a positive-definite covariance matrix."""

    values = np.array(
        [
            [0.040, 0.010, 0.005],
            [0.010, 0.062, 0.008],
            [0.005, 0.008, 0.090],
        ],
        dtype=float,
    )

    assets = ["A", "B", "C"]

    return pd.DataFrame(
        values,
        index=assets,
        columns=assets,
    )


# ---------------------------------------------------------------------
# HRP allocation behavior
# ---------------------------------------------------------------------


def test_hrp_returns_pandas_series(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert isinstance(weights, pd.Series)


def test_hrp_returns_one_weight_per_asset(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert len(weights) == returns.shape[1]
    assert set(weights.index) == set(returns.columns)


def test_hrp_weights_sum_to_one(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert weights.sum() == pytest.approx(1.0, abs=1e-10)


def test_hrp_weights_are_non_negative(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert (weights >= 0).all()


def test_hrp_weights_are_finite(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert np.isfinite(weights.to_numpy()).all()


def test_hrp_weights_are_not_all_equal(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert weights.nunique() > 1


def test_hrp_preserves_original_asset_order(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert list(weights.index) == list(returns.columns)


def test_hrp_assigns_correct_series_name(
    returns: pd.DataFrame,
) -> None:
    weights = hierarchical_risk_parity_weights(returns)

    assert weights.name == "hrp_weight"


def test_hrp_weights_are_reproducible(
    returns: pd.DataFrame,
) -> None:
    first = hierarchical_risk_parity_weights(returns)
    second = hierarchical_risk_parity_weights(returns)

    np.testing.assert_allclose(
        first.to_numpy(),
        second.to_numpy(),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "linkage_method",
    ["single", "complete", "average", "weighted"],
)
def test_hrp_supports_all_linkage_methods(
    returns: pd.DataFrame,
    linkage_method: str,
) -> None:
    weights = hierarchical_risk_parity_weights(
        returns,
        linkage_method=linkage_method,
    )

    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert (weights >= 0).all()
    assert np.isfinite(weights.to_numpy()).all()


def test_hrp_accepts_two_assets() -> None:
    rng = np.random.default_rng(10)

    data = pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.02, 100),
            "B": rng.normal(0.0005, 0.03, 100),
        }
    )

    weights = hierarchical_risk_parity_weights(data)

    assert len(weights) == 2
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert (weights >= 0).all()


def test_hrp_accepts_more_than_five_assets() -> None:
    rng = np.random.default_rng(20)

    data = pd.DataFrame(
        rng.normal(0.0005, 0.015, size=(300, 12)),
        columns=[f"Asset_{i}" for i in range(12)],
    )

    weights = hierarchical_risk_parity_weights(data)

    assert len(weights) == 12
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------
# Return-data validation
# ---------------------------------------------------------------------


def test_validate_returns_accepts_valid_data(
    returns: pd.DataFrame,
) -> None:
    result = _validate_returns(returns)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == returns.shape
    assert np.isfinite(result.to_numpy()).all()


def test_validate_returns_converts_numeric_strings() -> None:
    data = pd.DataFrame(
        {
            "A": ["0.01", "0.02", "0.03"],
            "B": ["0.02", "0.01", "0.04"],
        }
    )

    result = _validate_returns(data)

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in result.dtypes
    )


def test_validate_returns_drops_missing_observations() -> None:
    data = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.nan, 0.03],
            "B": [0.02, 0.01, 0.03, 0.04],
        }
    )

    result = _validate_returns(data)

    assert len(result) == 3
    assert not result.isna().any().any()


def test_validate_returns_drops_infinite_observations() -> None:
    data = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.inf, 0.03],
            "B": [0.02, 0.01, 0.03, 0.04],
        }
    )

    result = _validate_returns(data)

    assert len(result) == 3
    assert np.isfinite(result.to_numpy()).all()


def test_hrp_handles_missing_observations(
    returns: pd.DataFrame,
) -> None:
    data = returns.copy()
    data.iloc[10, 0] = np.nan

    weights = hierarchical_risk_parity_weights(data)

    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.isfinite(weights.to_numpy()).all()


def test_validate_returns_rejects_non_dataframe() -> None:
    with pytest.raises(
        TypeError,
        match="returns must be a pandas DataFrame",
    ):
        _validate_returns(np.array([[0.01, 0.02], [0.02, 0.03]]))


def test_validate_returns_rejects_empty_data() -> None:
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Returns data cannot be empty",
    ):
        _validate_returns(data)


def test_validate_returns_rejects_duplicate_asset_labels() -> None:
    data = pd.DataFrame(
        [[0.01, 0.02], [0.02, 0.03], [0.03, 0.01]],
        columns=["A", "A"],
    )

    with pytest.raises(
        ValueError,
        match="Asset labels must be unique",
    ):
        _validate_returns(data)


def test_validate_returns_requires_two_complete_observations() -> None:
    data = pd.DataFrame(
        {
            "A": [0.01, np.nan, np.nan],
            "B": [0.02, 0.03, np.nan],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two complete observations",
    ):
        _validate_returns(data)


def test_validate_returns_requires_at_least_two_assets() -> None:
    data = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two assets",
    ):
        _validate_returns(data)


def test_validate_returns_rejects_returns_at_minus_one() -> None:
    data = pd.DataFrame(
        {
            "A": [-1.0, 0.02, 0.03],
            "B": [0.02, 0.03, 0.04],
        }
    )

    with pytest.raises(
        ValueError,
        match="less than or equal to -100%",
    ):
        _validate_returns(data)


def test_validate_returns_rejects_returns_below_minus_one() -> None:
    data = pd.DataFrame(
        {
            "A": [-1.05, 0.02, 0.03],
            "B": [0.02, 0.03, 0.04],
        }
    )

    with pytest.raises(
        ValueError,
        match="less than or equal to -100%",
    ):
        _validate_returns(data)


def test_validate_returns_rejects_constant_asset() -> None:
    data = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.01],
            "B": [0.02, 0.03, 0.01, 0.04],
        }
    )

    with pytest.raises(
        ValueError,
        match="positive return volatility",
    ):
        _validate_returns(data)


def test_validate_returns_rejects_non_numeric_data() -> None:
    data = pd.DataFrame(
        {
            "A": ["bad", "bad", "bad"],
            "B": [0.01, 0.02, 0.03],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two complete observations",
    ):
        _validate_returns(data)


# ---------------------------------------------------------------------
# Linkage-method validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "linkage_method",
    [
        "ward",
        "centroid",
        "median",
        "invalid",
        "",
    ],
)
def test_hrp_rejects_unsupported_linkage_methods(
    returns: pd.DataFrame,
    linkage_method: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported linkage method",
    ):
        hierarchical_risk_parity_weights(
            returns,
            linkage_method=linkage_method,
        )


# ---------------------------------------------------------------------
# Cluster variance calculations
# ---------------------------------------------------------------------


def test_cluster_variance_is_positive(
    covariance: pd.DataFrame,
) -> None:
    variance = _cluster_variance(
        covariance,
        ["A", "B"],
    )

    assert variance > 0
    assert np.isfinite(variance)


def test_cluster_variance_is_finite(
    covariance: pd.DataFrame,
) -> None:
    variance = _cluster_variance(
        covariance,
        ["A", "B", "C"],
    )

    assert np.isfinite(variance)


def test_cluster_variance_uses_inverse_variance_weights(
    covariance: pd.DataFrame,
) -> None:
    assets = ["A", "B"]

    submatrix = covariance.loc[assets, assets].to_numpy()

    inverse_variance = 1.0 / np.diag(submatrix)
    weights = inverse_variance / inverse_variance.sum()

    expected = float(weights @ submatrix @ weights)

    actual = _cluster_variance(
        covariance,
        assets,
    )

    assert actual == pytest.approx(expected)


def test_cluster_variance_rejects_non_positive_asset_variance() -> None:
    invalid_covariance = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Asset variances must be positive",
    ):
        _cluster_variance(
            invalid_covariance,
            ["A", "B"],
        )


def test_cluster_variance_rejects_non_positive_cluster_variance() -> None:
    invalid_covariance = pd.DataFrame(
        [
            [1.0, -2.0],
            [-2.0, 1.0],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(
        ValueError,
        match="Cluster variance must be finite and positive",
    ):
        _cluster_variance(
            invalid_covariance,
            ["A", "B"],
        )
