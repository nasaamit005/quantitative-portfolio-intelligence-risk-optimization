
"""Unit tests for Black-Litterman return estimation."""

import numpy as np
import pandas as pd
import pytest

from quantfin.optimization.black_litterman import (
    _validate_covariance,
    _validate_market_weights,
    black_litterman_posterior,
    build_view_covariance,
    market_implied_equilibrium_returns,
)


@pytest.fixture
def sample_covariance() -> pd.DataFrame:
    """Create an annualized positive-definite covariance matrix."""

    values = np.array(
        [
            [0.040, 0.010, 0.005],
            [0.010, 0.090, 0.015],
            [0.005, 0.015, 0.160],
        ]
    )

    assets = ["AAPL", "MSFT", "KO"]

    return pd.DataFrame(
        values,
        index=assets,
        columns=assets,
    )


@pytest.fixture
def sample_market_weights() -> pd.Series:
    """Create market weights for the sample assets."""

    return pd.Series(
        [0.50, 0.30, 0.20],
        index=["AAPL", "MSFT", "KO"],
    )


@pytest.fixture
def sample_view_matrix() -> pd.DataFrame:
    """Create two investor views across three assets."""

    return pd.DataFrame(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        index=["view_1", "view_2"],
        columns=["AAPL", "MSFT", "KO"],
    )


@pytest.fixture
def sample_view_returns() -> pd.Series:
    """Create expected excess returns for the investor views."""

    return pd.Series(
        [0.04, 0.02],
        index=["view_1", "view_2"],
    )


# ---------------------------------------------------------------------
# Covariance validation
# ---------------------------------------------------------------------


def test_validate_covariance_accepts_valid_matrix(sample_covariance):
    result = _validate_covariance(sample_covariance)

    pd.testing.assert_frame_equal(result, sample_covariance)


def test_validate_covariance_returns_dataframe(sample_covariance):
    result = _validate_covariance(sample_covariance)

    assert isinstance(result, pd.DataFrame)


def test_validate_covariance_reorders_columns_to_index_order():
    covariance = pd.DataFrame(
        [
            [0.09, 0.01, 0.015],
            [0.01, 0.04, 0.005],
            [0.015, 0.005, 0.16],
        ],
        index=["MSFT", "AAPL", "KO"],
        columns=["MSFT", "AAPL", "KO"],
    )

    result = _validate_covariance(covariance)

    assert result.index.equals(covariance.index)
    assert result.columns.equals(covariance.index)
    assert np.allclose(result.to_numpy(), result.to_numpy().T)


def test_validate_covariance_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        _validate_covariance(np.eye(3))


def test_validate_covariance_rejects_empty_matrix():
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_covariance(pd.DataFrame())


def test_validate_covariance_rejects_duplicate_index_labels():
    covariance = pd.DataFrame(
        np.eye(3),
        index=["AAPL", "AAPL", "KO"],
        columns=["AAPL", "MSFT", "KO"],
    )

    with pytest.raises(ValueError, match="index labels must be unique"):
        _validate_covariance(covariance)


def test_validate_covariance_rejects_duplicate_column_labels():
    covariance = pd.DataFrame(
        np.eye(3),
        index=["AAPL", "MSFT", "KO"],
        columns=["AAPL", "AAPL", "KO"],
    )

    with pytest.raises(ValueError, match="column labels must be unique"):
        _validate_covariance(covariance)


def test_validate_covariance_rejects_mismatched_labels():
    covariance = pd.DataFrame(
        np.eye(3),
        index=["AAPL", "MSFT", "KO"],
        columns=["AAPL", "MSFT", "JPM"],
    )

    with pytest.raises(ValueError, match="must match"):
        _validate_covariance(covariance)


def test_validate_covariance_rejects_nonfinite_values(sample_covariance):
    covariance = sample_covariance.copy()
    covariance.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        _validate_covariance(covariance)


def test_validate_covariance_rejects_asymmetric_matrix(sample_covariance):
    covariance = sample_covariance.copy()
    covariance.iloc[0, 1] = 0.50

    with pytest.raises(ValueError, match="symmetric"):
        _validate_covariance(covariance)


def test_validate_covariance_rejects_non_positive_definite_matrix():
    covariance = pd.DataFrame(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    with pytest.raises(ValueError, match="positive definite"):
        _validate_covariance(covariance)


def test_validate_covariance_converts_numeric_values_to_float():
    covariance = pd.DataFrame(
        [
            [4, 1],
            [1, 9],
        ],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    result = _validate_covariance(covariance)

    assert all(pd.api.types.is_float_dtype(dtype) for dtype in result.dtypes)


# ---------------------------------------------------------------------
# Market weight validation
# ---------------------------------------------------------------------


def test_validate_market_weights_accepts_valid_weights(
    sample_market_weights,
    sample_covariance,
):
    result = _validate_market_weights(
        sample_market_weights,
        sample_covariance.index,
    )

    assert isinstance(result, pd.Series)
    assert np.isclose(result.sum(), 1.0)


def test_validate_market_weights_normalizes_weights():
    weights = pd.Series(
        [5.0, 3.0, 2.0],
        index=["AAPL", "MSFT", "KO"],
    )

    result = _validate_market_weights(
        weights,
        pd.Index(["AAPL", "MSFT", "KO"]),
    )

    expected = pd.Series(
        [0.5, 0.3, 0.2],
        index=["AAPL", "MSFT", "KO"],
    )

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )


def test_validate_market_weights_reorders_assets():
    weights = pd.Series(
        [0.2, 0.5, 0.3],
        index=["KO", "AAPL", "MSFT"],
    )

    result = _validate_market_weights(
        weights,
        pd.Index(["AAPL", "MSFT", "KO"]),
    )

    assert result.index.tolist() == ["AAPL", "MSFT", "KO"]
    assert np.allclose(result.to_numpy(), [0.5, 0.3, 0.2])


def test_validate_market_weights_rejects_non_series():
    with pytest.raises(TypeError, match="pandas Series"):
        _validate_market_weights(
            [0.5, 0.3, 0.2],
            pd.Index(["AAPL", "MSFT", "KO"]),
        )


def test_validate_market_weights_rejects_duplicate_labels():
    weights = pd.Series(
        [0.5, 0.3, 0.2],
        index=["AAPL", "AAPL", "KO"],
    )

    with pytest.raises(ValueError, match="unique"):
        _validate_market_weights(
            weights,
            pd.Index(["AAPL", "MSFT", "KO"]),
        )


def test_validate_market_weights_rejects_missing_or_extra_assets():
    weights = pd.Series(
        [0.5, 0.3, 0.2],
        index=["AAPL", "MSFT", "JPM"],
    )

    with pytest.raises(ValueError, match="exactly"):
        _validate_market_weights(
            weights,
            pd.Index(["AAPL", "MSFT", "KO"]),
        )


def test_validate_market_weights_rejects_nonfinite_values():
    weights = pd.Series(
        [0.5, np.nan, 0.5],
        index=["AAPL", "MSFT", "KO"],
    )

    with pytest.raises(ValueError, match="finite"):
        _validate_market_weights(
            weights,
            pd.Index(["AAPL", "MSFT", "KO"]),
        )


def test_validate_market_weights_rejects_negative_values():
    weights = pd.Series(
        [0.6, -0.1, 0.5],
        index=["AAPL", "MSFT", "KO"],
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        _validate_market_weights(
            weights,
            pd.Index(["AAPL", "MSFT", "KO"]),
        )


def test_validate_market_weights_rejects_zero_total():
    weights = pd.Series(
        [0.0, 0.0, 0.0],
        index=["AAPL", "MSFT", "KO"],
    )

    with pytest.raises(ValueError, match="positive sum"):
        _validate_market_weights(
            weights,
            pd.Index(["AAPL", "MSFT", "KO"]),
        )


# ---------------------------------------------------------------------
# Market-implied equilibrium returns
# ---------------------------------------------------------------------


def test_equilibrium_returns_match_black_litterman_formula(
    sample_covariance,
    sample_market_weights,
):
    risk_aversion = 2.5

    result = market_implied_equilibrium_returns(
        sample_covariance,
        sample_market_weights,
        risk_aversion=risk_aversion,
    )

    normalized_weights = (
        sample_market_weights / sample_market_weights.sum()
    )

    expected = (
        risk_aversion
        * sample_covariance.to_numpy()
        @ normalized_weights.to_numpy()
    )

    assert np.allclose(result.to_numpy(), expected)


def test_equilibrium_returns_are_series(
    sample_covariance,
    sample_market_weights,
):
    result = market_implied_equilibrium_returns(
        sample_covariance,
        sample_market_weights,
    )

    assert isinstance(result, pd.Series)
    assert result.name == "equilibrium_excess_return"


def test_equilibrium_returns_preserve_asset_labels(
    sample_covariance,
    sample_market_weights,
):
    result = market_implied_equilibrium_returns(
        sample_covariance,
        sample_market_weights,
    )

    assert result.index.equals(sample_covariance.index)


def test_equilibrium_returns_are_finite(
    sample_covariance,
    sample_market_weights,
):
    result = market_implied_equilibrium_returns(
        sample_covariance,
        sample_market_weights,
    )

    assert np.isfinite(result.to_numpy()).all()


def test_equilibrium_returns_reject_zero_risk_aversion(
    sample_covariance,
    sample_market_weights,
):
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        market_implied_equilibrium_returns(
            sample_covariance,
            sample_market_weights,
            risk_aversion=0,
        )


def test_equilibrium_returns_reject_negative_risk_aversion(
    sample_covariance,
    sample_market_weights,
):
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        market_implied_equilibrium_returns(
            sample_covariance,
            sample_market_weights,
            risk_aversion=-2.5,
        )


def test_equilibrium_returns_reject_nonfinite_risk_free_rate(
    sample_covariance,
    sample_market_weights,
):
    with pytest.raises(ValueError, match="risk_free_rate must be finite"):
        market_implied_equilibrium_returns(
            sample_covariance,
            sample_market_weights,
            risk_free_rate=np.inf,
        )


def test_equilibrium_returns_are_excess_returns(
    sample_covariance,
    sample_market_weights,
):
    result = market_implied_equilibrium_returns(
        sample_covariance,
        sample_market_weights,
        risk_free_rate=0.04,
    )

    # The implementation returns Pi = delta * Sigma * w.
    # It does not add the risk-free rate to the result.
    expected = (
        2.5
        * sample_covariance.to_numpy()
        @ (
            sample_market_weights
            / sample_market_weights.sum()
        ).to_numpy()
    )

    assert np.allclose(result.to_numpy(), expected)


# ---------------------------------------------------------------------
# View uncertainty covariance
# ---------------------------------------------------------------------


def test_build_view_covariance_returns_dataframe(
    sample_covariance,
    sample_view_matrix,
):
    result = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


def test_build_view_covariance_has_view_labels(
    sample_covariance,
    sample_view_matrix,
):
    result = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
    )

    assert result.index.equals(sample_view_matrix.index)
    assert result.columns.equals(sample_view_matrix.index)


def test_build_view_covariance_is_diagonal(
    sample_covariance,
    sample_view_matrix,
):
    result = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
    )

    assert np.allclose(
        result.to_numpy(),
        np.diag(np.diag(result.to_numpy())),
    )


def test_build_view_covariance_matches_formula(
    sample_covariance,
    sample_view_matrix,
):
    tau = 0.05
    confidence = 0.60

    result = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
        tau=tau,
        confidence=confidence,
    )

    p = sample_view_matrix.to_numpy()
    sigma = sample_covariance.to_numpy()

    expected_diagonal = (
        tau
        * np.diag(p @ sigma @ p.T)
        * (1.0 - confidence)
        / confidence
    )

    assert np.allclose(
        np.diag(result.to_numpy()),
        expected_diagonal,
    )


def test_build_view_covariance_accepts_individual_confidences(
    sample_covariance,
    sample_view_matrix,
):
    result = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
        confidence=[0.60, 0.80],
    )

    assert result.shape == (2, 2)
    assert np.all(np.diag(result.to_numpy()) > 0)


def test_build_view_covariance_reduces_uncertainty_at_higher_confidence(
    sample_covariance,
    sample_view_matrix,
):
    low_confidence = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
        confidence=0.40,
    )

    high_confidence = build_view_covariance(
        sample_covariance,
        sample_view_matrix,
        confidence=0.80,
    )

    assert (
        np.diag(high_confidence.to_numpy())
        < np.diag(low_confidence.to_numpy())
    ).all()


def test_build_view_covariance_rejects_non_dataframe_view_matrix(
    sample_covariance,
):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        build_view_covariance(
            sample_covariance,
            np.array([[1.0, -1.0, 0.0]]),
        )


def test_build_view_covariance_rejects_empty_views(sample_covariance):
    views = pd.DataFrame(
        columns=sample_covariance.columns,
        dtype=float,
    )

    with pytest.raises(ValueError, match="At least one"):
        build_view_covariance(sample_covariance, views)


def test_build_view_covariance_rejects_duplicate_asset_labels(
    sample_covariance,
):
    views = pd.DataFrame(
        [[1.0, -1.0, 0.0]],
        index=["view_1"],
        columns=["AAPL", "AAPL", "KO"],
    )

    with pytest.raises(ValueError, match="unique"):
        build_view_covariance(sample_covariance, views)


def test_build_view_covariance_rejects_mismatched_assets(
    sample_covariance,
):
    views = pd.DataFrame(
        [[1.0, -1.0, 0.0]],
        index=["view_1"],
        columns=["AAPL", "MSFT", "JPM"],
    )

    with pytest.raises(ValueError, match="must match"):
        build_view_covariance(sample_covariance, views)


def test_build_view_covariance_rejects_invalid_tau(
    sample_covariance,
    sample_view_matrix,
):
    for tau in [0.0, -0.05, np.inf, np.nan]:
        with pytest.raises(ValueError, match="tau must be finite"):
            build_view_covariance(
                sample_covariance,
                sample_view_matrix,
                tau=tau,
            )


def test_build_view_covariance_rejects_nonfinite_view_values(
    sample_covariance,
    sample_view_matrix,
):
    views = sample_view_matrix.copy()
    views.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_view_covariance(sample_covariance, views)


def test_build_view_covariance_rejects_all_zero_views(
    sample_covariance,
):
    views = pd.DataFrame(
        [[0.0, 0.0, 0.0]],
        index=["view_1"],
        columns=sample_covariance.columns,
    )

    with pytest.raises(ValueError, match="zero rows"):
        build_view_covariance(sample_covariance, views)


def test_build_view_covariance_rejects_wrong_confidence_count(
    sample_covariance,
    sample_view_matrix,
):
    with pytest.raises(ValueError, match="one confidence value"):
        build_view_covariance(
            sample_covariance,
            sample_view_matrix,
            confidence=[0.50],
        )


@pytest.mark.parametrize(
    "confidence",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        np.nan,
        np.inf,
    ],
)
def test_build_view_covariance_rejects_invalid_confidence(
    sample_covariance,
    sample_view_matrix,
    confidence,
):
    with pytest.raises(ValueError, match="strictly between"):
        build_view_covariance(
            sample_covariance,
            sample_view_matrix,
            confidence=confidence,
        )


    covariance = sample_covariance.copy()
    covariance.loc["KO", "KO"] = 0.0
    covariance.loc["AAPL", "AAPL"] = 0.04
    covariance.loc["MSFT", "MSFT"] = 0.09
    covariance.loc["AAPL", "KO"] = 0.0
    covariance.loc["KO", "AAPL"] = 0.0
    covariance.loc["MSFT", "KO"] = 0.0
    covariance.loc["KO", "MSFT"] = 0.0

    # The covariance validator rejects a non-positive-definite
    # matrix first, so use a zero view row variance through a
    # zero view vector, which is rejected by the all-zero check.
    # This test instead verifies that the view matrix is rejected.
    with pytest.raises(ValueError, match="zero rows"):
        build_view_covariance(
            sample_covariance,
            pd.DataFrame(
                [[0.0, 0.0, 0.0]],
                index=["view_1"],
                columns=sample_covariance.columns,
            ),
        )


# ---------------------------------------------------------------------
# Black-Litterman posterior estimation
# ---------------------------------------------------------------------


def test_black_litterman_returns_four_outputs(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    result = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    assert isinstance(result, tuple)
    assert len(result) == 4

    posterior_returns, posterior_covariance, equilibrium, omega = result

    assert isinstance(posterior_returns, pd.Series)
    assert isinstance(posterior_covariance, pd.DataFrame)
    assert isinstance(equilibrium, pd.Series)
    assert isinstance(omega, pd.DataFrame)


def test_black_litterman_preserves_asset_labels(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    posterior_returns, posterior_covariance, _, _ = (
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            sample_view_returns,
        )
    )

    assert posterior_returns.index.equals(sample_covariance.index)
    assert posterior_covariance.index.equals(sample_covariance.index)
    assert posterior_covariance.columns.equals(sample_covariance.columns)


def test_black_litterman_returns_expected_names(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    posterior_returns, posterior_covariance, equilibrium, omega = (
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            sample_view_returns,
        )
    )

    assert posterior_returns.name == "posterior_excess_return"
    assert equilibrium.name == "equilibrium_excess_return"
    assert posterior_covariance.index.equals(sample_covariance.index)
    assert omega.index.equals(sample_view_matrix.index)


def test_black_litterman_outputs_are_finite(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    posterior_returns, posterior_covariance, equilibrium, omega = (
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            sample_view_returns,
        )
    )

    assert np.isfinite(posterior_returns.to_numpy()).all()
    assert np.isfinite(posterior_covariance.to_numpy()).all()
    assert np.isfinite(equilibrium.to_numpy()).all()
    assert np.isfinite(omega.to_numpy()).all()


def test_black_litterman_posterior_covariance_is_symmetric(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    _, posterior_covariance, _, _ = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    assert np.allclose(
        posterior_covariance.to_numpy(),
        posterior_covariance.to_numpy().T,
        atol=1e-10,
    )


def test_black_litterman_posterior_covariance_is_positive_definite(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    _, posterior_covariance, _, _ = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    eigenvalues = np.linalg.eigvalsh(
        posterior_covariance.to_numpy()
    )

    assert eigenvalues.min() > 0


def test_black_litterman_posterior_covariance_exceeds_prior_covariance(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    _, posterior_covariance, _, _ = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    difference = (
        posterior_covariance.to_numpy()
        - sample_covariance.to_numpy()
    )

    assert np.linalg.eigvalsh(difference).min() > -1e-10


def test_black_litterman_aligns_view_rows_with_view_returns(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    original = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    reordered_matrix = sample_view_matrix.loc[
        ["view_2", "view_1"]
    ]

    reordered_returns = sample_view_returns.loc[
        ["view_1", "view_2"]
    ]

    reordered = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        reordered_matrix,
        reordered_returns,
    )

    pd.testing.assert_series_equal(original[0], reordered[0])
    pd.testing.assert_frame_equal(original[1], reordered[1])


def test_black_litterman_aligns_market_weights(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    original = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    reordered_weights = sample_market_weights.loc[
        ["KO", "AAPL", "MSFT"]
    ]

    reordered = black_litterman_posterior(
        sample_covariance,
        reordered_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    pd.testing.assert_series_equal(original[0], reordered[0])
    pd.testing.assert_frame_equal(original[1], reordered[1])


def test_black_litterman_rejects_non_dataframe_view_matrix(
    sample_covariance,
    sample_market_weights,
    sample_view_returns,
):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            np.array([[1.0, -1.0, 0.0]]),
            sample_view_returns,
        )


def test_black_litterman_rejects_non_series_view_returns(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
):
    with pytest.raises(TypeError, match="pandas Series"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            [0.04, 0.02],
        )


def test_black_litterman_rejects_empty_views(
    sample_covariance,
    sample_market_weights,
):
    views = pd.DataFrame(
        columns=sample_covariance.columns,
        dtype=float,
    )

    view_returns = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="At least one"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            views,
            view_returns,
        )


def test_black_litterman_rejects_duplicate_view_identifiers(
    sample_covariance,
    sample_market_weights,
    sample_view_returns,
):
    views = pd.DataFrame(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        index=["view_1", "view_1"],
        columns=sample_covariance.columns,
    )

    with pytest.raises(ValueError, match="View identifiers must be unique"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            views,
            sample_view_returns,
        )


def test_black_litterman_rejects_duplicate_view_return_identifiers(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
):
    view_returns = pd.Series(
        [0.04, 0.02],
        index=["view_1", "view_1"],
    )

    with pytest.raises(
        ValueError,
        match="View return identifiers must be unique",
    ):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            view_returns,
        )


def test_black_litterman_rejects_mismatched_view_identifiers(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
):
    view_returns = pd.Series(
        [0.04, 0.02],
        index=["view_1", "view_3"],
    )

    with pytest.raises(ValueError, match="rows must match"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            view_returns,
        )


def test_black_litterman_rejects_nonfinite_view_returns(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
):
    view_returns = pd.Series(
        [np.nan, 0.02],
        index=["view_1", "view_2"],
    )

    with pytest.raises(ValueError, match="finite"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            view_returns,
        )


def test_black_litterman_rejects_nonfinite_view_matrix(
    sample_covariance,
    sample_market_weights,
    sample_view_returns,
    sample_view_matrix,
):
    views = sample_view_matrix.copy()
    views.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            views,
            sample_view_returns,
        )


def test_black_litterman_rejects_all_zero_view_matrix(
    sample_covariance,
    sample_market_weights,
    sample_view_returns,
):
    views = pd.DataFrame(
        np.zeros((2, 3)),
        index=["view_1", "view_2"],
        columns=sample_covariance.columns,
    )

    with pytest.raises(ValueError, match="zero rows"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            views,
            sample_view_returns,
        )


def test_black_litterman_rejects_invalid_risk_aversion(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            sample_view_returns,
            risk_aversion=0,
        )


def test_black_litterman_rejects_invalid_tau(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    with pytest.raises(ValueError, match="tau must be finite"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            sample_view_returns,
            tau=0,
        )


def test_black_litterman_rejects_invalid_confidence(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    with pytest.raises(ValueError, match="strictly between"):
        black_litterman_posterior(
            sample_covariance,
            sample_market_weights,
            sample_view_matrix,
            sample_view_returns,
            confidence=1.0,
        )


def test_black_litterman_posterior_is_reproducible(
    sample_covariance,
    sample_market_weights,
    sample_view_matrix,
    sample_view_returns,
):
    first = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    second = black_litterman_posterior(
        sample_covariance,
        sample_market_weights,
        sample_view_matrix,
        sample_view_returns,
    )

    for first_item, second_item in zip(first, second, strict=True):
        if isinstance(first_item, pd.Series):
            pd.testing.assert_series_equal(first_item, second_item)
        else:
            pd.testing.assert_frame_equal(first_item, second_item)
