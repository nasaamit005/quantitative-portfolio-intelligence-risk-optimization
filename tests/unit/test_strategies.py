"""Unit tests for portfolio investment strategy construction."""

import numpy as np
import pandas as pd
import pytest

from quantfin.strategies.portfolio import (
    _validate_returns,
    _validate_weights,
    black_litterman_strategy,
    buy_and_hold_weights,
    equal_weight_weights,
    hrp_strategy,
    market_cap_weighted_weights,
    maximum_sharpe_strategy,
    minimum_volatility_strategy,
    risk_parity_strategy,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Create deterministic, positive-definite synthetic asset returns."""
    rng = np.random.default_rng(42)

    assets = ["AAPL", "MSFT", "GOOGL"]
    means = np.array([0.0005, 0.0007, 0.0009])
    factor = rng.normal(0.0, 0.006, size=180)
    noise = rng.normal(0.0, 0.008, size=(180, 3))

    values = means + factor[:, None] + noise

    return pd.DataFrame(
        values,
        index=pd.bdate_range("2024-01-01", periods=180),
        columns=assets,
    )


@pytest.fixture
def sample_market_caps() -> pd.Series:
    """Market capitalizations in a consistent illustrative unit."""
    return pd.Series(
        {"AAPL": 3000.0, "MSFT": 2500.0, "GOOGL": 2000.0},
        name="market_cap",
    )


@pytest.fixture
def sample_bl_views() -> tuple[pd.DataFrame, pd.Series]:
    """Create one relative-return view: AAPL outperforming KO is not used here."""
    view_matrix = pd.DataFrame(
        [[1.0, -1.0, 0.0]],
        index=["AAPL_vs_MSFT"],
        columns=["AAPL", "MSFT", "GOOGL"],
    )
    view_returns = pd.Series(
        {"AAPL_vs_MSFT": 0.02},
        name="view_return",
    )

    return view_matrix, view_returns


def assert_valid_weights(
    weights: pd.Series,
    expected_assets: list[str],
    max_weight: float | None = None,
) -> None:
    """Check common long-only, fully invested portfolio requirements."""
    assert isinstance(weights, pd.Series)
    assert weights.index.is_unique
    assert list(weights.index) == expected_assets
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights >= -1e-8).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    if max_weight is not None:
        assert (weights <= max_weight + 1e-6).all()


# ---------------------------------------------------------------------
# _validate_returns
# ---------------------------------------------------------------------


def test_validate_returns_returns_clean_dataframe(sample_returns):
    result = _validate_returns(sample_returns)

    pd.testing.assert_frame_equal(result, sample_returns.astype(float))


def test_validate_returns_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        _validate_returns(np.array([[0.01, 0.02], [0.03, 0.04]]))


@pytest.mark.parametrize(
    "returns",
    [
        pd.DataFrame(),
        pd.DataFrame({"AAPL": [0.01, 0.02]}),
    ],
)
def test_validate_returns_rejects_empty_or_single_asset_data(returns):
    with pytest.raises(ValueError, match="At least two assets"):
        _validate_returns(returns)


def test_validate_returns_rejects_duplicate_asset_columns():
    returns = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04]],
        columns=["AAPL", "AAPL"],
    )

    with pytest.raises(ValueError, match="unique"):
        _validate_returns(returns)


def test_validate_returns_rejects_non_chronological_index():
    returns = pd.DataFrame(
        {"AAPL": [0.01, 0.02, 0.03], "MSFT": [0.02, 0.01, 0.04]},
        index=[3, 1, 2],
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        _validate_returns(returns)


def test_validate_returns_drops_incomplete_rows():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, np.nan, 0.03],
            "MSFT": [0.02, 0.04, 0.05],
        }
    )

    result = _validate_returns(returns)

    assert len(result) == 2
    assert result.isna().sum().sum() == 0


def test_validate_returns_rejects_infinite_values():
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, np.inf, 0.03],
            "MSFT": [0.02, 0.04, 0.05],
        }
    )

    with pytest.raises(ValueError, match="non-finite"):
        _validate_returns(returns)


def test_validate_returns_rejects_returns_at_or_below_minus_one():
    returns = pd.DataFrame(
        {
            "AAPL": [-1.0, 0.02, 0.03],
            "MSFT": [0.02, 0.04, 0.05],
        }
    )

    with pytest.raises(ValueError, match="less than or equal"):
        _validate_returns(returns)


def test_validate_returns_rejects_data_without_complete_rows():
    returns = pd.DataFrame(
        {
            "AAPL": [np.nan, np.nan],
            "MSFT": [0.02, 0.04],
        }
    )

    with pytest.raises(ValueError, match="No complete"):
        _validate_returns(returns)


# ---------------------------------------------------------------------
# _validate_weights and buy-and-hold
# ---------------------------------------------------------------------


def test_validate_weights_accepts_valid_portfolio():
    weights = pd.Series({"AAPL": 0.4, "MSFT": 0.6})

    result = _validate_weights(weights)

    pd.testing.assert_series_equal(result, weights)


def test_validate_weights_normalizes_small_sum_difference():
    weights = pd.Series({"AAPL": 0.4, "MSFT": 0.5999995})

    result = _validate_weights(weights)

    assert result.sum() == pytest.approx(1.0)


def test_validate_weights_rejects_non_series():
    with pytest.raises(TypeError, match="pandas Series"):
        _validate_weights({"AAPL": 0.5, "MSFT": 0.5})


def test_validate_weights_rejects_empty_weights():
    with pytest.raises(ValueError, match="unique asset labels"):
        _validate_weights(pd.Series(dtype=float))


def test_validate_weights_rejects_duplicate_labels():
    weights = pd.Series([0.5, 0.5], index=["AAPL", "AAPL"])

    with pytest.raises(ValueError, match="unique"):
        _validate_weights(weights)


def test_validate_weights_rejects_nonfinite_values():
    weights = pd.Series({"AAPL": np.nan, "MSFT": 1.0})

    with pytest.raises(ValueError, match="finite"):
        _validate_weights(weights)


def test_validate_weights_rejects_short_positions():
    weights = pd.Series({"AAPL": -0.1, "MSFT": 1.1})

    with pytest.raises(ValueError, match="Short positions"):
        _validate_weights(weights)


def test_validate_weights_rejects_weights_not_summing_to_one():
    weights = pd.Series({"AAPL": 0.3, "MSFT": 0.3})

    with pytest.raises(ValueError, match="sum to one"):
        _validate_weights(weights)


def test_buy_and_hold_weights_preserves_valid_allocations():
    initial = pd.Series({"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25})

    result = buy_and_hold_weights(initial)

    assert_valid_weights(result, ["AAPL", "MSFT", "GOOGL"])
    assert result.name == "Buy and Hold"
    pd.testing.assert_series_equal(
        result,
        initial.rename("Buy and Hold"),
    )


def test_buy_and_hold_weights_does_not_modify_input():
    initial = pd.Series({"AAPL": 0.4, "MSFT": 0.6})
    original = initial.copy()

    buy_and_hold_weights(initial)

    pd.testing.assert_series_equal(initial, original)


def test_buy_and_hold_weights_rejects_invalid_allocation():
    initial = pd.Series({"AAPL": 0.4, "MSFT": 0.4})

    with pytest.raises(ValueError, match="sum to one"):
        buy_and_hold_weights(initial)


# ---------------------------------------------------------------------
# Equal-weight strategy
# ---------------------------------------------------------------------


def test_equal_weight_strategy_allocates_equally(sample_returns):
    result = equal_weight_weights(sample_returns)

    assert_valid_weights(result, list(sample_returns.columns))
    assert result.name == "Equal Weight"
    assert np.allclose(result.to_numpy(), 1.0 / 3.0)


def test_equal_weight_strategy_uses_cleaned_observations(sample_returns):
    returns = sample_returns.copy()
    returns.iloc[0, 0] = np.nan

    result = equal_weight_weights(returns)

    assert_valid_weights(result, list(sample_returns.columns))


# ---------------------------------------------------------------------
# Market-cap-weighted strategy
# ---------------------------------------------------------------------


def test_market_cap_weights_calculates_proportional_allocations(
    sample_market_caps,
):
    result = market_cap_weighted_weights(sample_market_caps)

    expected = sample_market_caps / sample_market_caps.sum()

    pd.testing.assert_series_equal(
        result,
        expected.rename("Market Cap Weighted"),
    )


def test_market_cap_weights_aligns_to_requested_asset_order(
    sample_market_caps,
):
    assets = pd.Index(["GOOGL", "AAPL", "MSFT"])

    result = market_cap_weighted_weights(sample_market_caps, assets)

    assert list(result.index) == list(assets)
    assert result["GOOGL"] == pytest.approx(2000.0 / 7500.0)


def test_market_cap_weights_rejects_non_series():
    with pytest.raises(TypeError, match="pandas Series"):
        market_cap_weighted_weights({"AAPL": 100.0, "MSFT": 200.0})


def test_market_cap_weights_rejects_empty_input():
    with pytest.raises(ValueError, match="unique asset labels"):
        market_cap_weighted_weights(pd.Series(dtype=float))


def test_market_cap_weights_rejects_duplicate_labels():
    caps = pd.Series([100.0, 200.0], index=["AAPL", "AAPL"])

    with pytest.raises(ValueError, match="unique"):
        market_cap_weighted_weights(caps)


@pytest.mark.parametrize(
    "caps",
    [
        pd.Series({"AAPL": 100.0, "MSFT": 0.0}),
        pd.Series({"AAPL": 100.0, "MSFT": -10.0}),
        pd.Series({"AAPL": 100.0, "MSFT": np.inf}),
    ],
)
def test_market_cap_weights_rejects_invalid_capitalizations(caps):
    with pytest.raises(ValueError, match="finite and positive"):
        market_cap_weighted_weights(caps)


def test_market_cap_weights_rejects_missing_requested_assets(
    sample_market_caps,
):
    assets = pd.Index(["AAPL", "MSFT", "NVDA"])

    with pytest.raises(ValueError, match="Missing market capitalizations"):
        market_cap_weighted_weights(sample_market_caps, assets)


# ---------------------------------------------------------------------
# Optimizer-backed strategies
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        minimum_volatility_strategy,
        maximum_sharpe_strategy,
        hrp_strategy,
        risk_parity_strategy,
    ],
)
def test_optimization_strategies_return_valid_weights(
    strategy,
    sample_returns,
):
    result = strategy(sample_returns)

    assert_valid_weights(
        result,
        list(sample_returns.columns),
        max_weight=0.50 if strategy is not hrp_strategy else None,
    )


def test_minimum_volatility_strategy_respects_custom_max_weight(
    sample_returns,
):
    result = minimum_volatility_strategy(
        sample_returns,
        max_weight=0.80,
    )

    assert_valid_weights(
        result,
        list(sample_returns.columns),
        max_weight=0.80,
    )


def test_maximum_sharpe_strategy_respects_custom_parameters(sample_returns):
    result = maximum_sharpe_strategy(
        sample_returns,
        risk_free_rate=0.02,
        max_weight=0.80,
    )

    assert_valid_weights(
        result,
        list(sample_returns.columns),
        max_weight=0.80,
    )


def test_risk_parity_strategy_respects_custom_max_weight(sample_returns):
    result = risk_parity_strategy(
        sample_returns,
        max_weight=0.80,
    )

    assert_valid_weights(
        result,
        list(sample_returns.columns),
        max_weight=0.80,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        minimum_volatility_strategy,
        maximum_sharpe_strategy,
        hrp_strategy,
        risk_parity_strategy,
    ],
)
def test_optimization_strategies_reject_invalid_returns(strategy):
    invalid_returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
            "MSFT": [0.02, 0.03, 0.04],
        },
        index=[3, 1, 2],
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        strategy(invalid_returns)


# ---------------------------------------------------------------------
# Black-Litterman strategy
# ---------------------------------------------------------------------


def test_black_litterman_strategy_returns_valid_weights(
    sample_returns,
    sample_bl_views,
):
    view_matrix, view_returns = sample_bl_views

    result = black_litterman_strategy(
        returns=sample_returns,
        market_weights=pd.Series(
            {"AAPL": 1 / 3, "MSFT": 1 / 3, "GOOGL": 1 / 3}
        ),
        view_matrix=view_matrix,
        view_returns=view_returns,
        risk_free_rate=0.02,
        max_weight=0.80,
    )

    assert_valid_weights(
        result,
        list(sample_returns.columns),
        max_weight=0.80,
    )


def test_black_litterman_strategy_aligns_market_weights(
    sample_returns,
    sample_bl_views,
):
    view_matrix, view_returns = sample_bl_views

    market_weights = pd.Series(
        {"GOOGL": 0.3, "AAPL": 0.4, "MSFT": 0.3}
    )

    result = black_litterman_strategy(
        returns=sample_returns,
        market_weights=market_weights,
        view_matrix=view_matrix,
        view_returns=view_returns,
        max_weight=0.80,
    )

    assert_valid_weights(
        result,
        list(sample_returns.columns),
        max_weight=0.80,
    )


def test_black_litterman_strategy_rejects_invalid_returns(
    sample_bl_views,
):
    view_matrix, view_returns = sample_bl_views

    invalid_returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
            "MSFT": [0.02, 0.03, 0.04],
        },
        index=[3, 1, 2],
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        black_litterman_strategy(
            returns=invalid_returns,
            market_weights=pd.Series({"AAPL": 0.5, "MSFT": 0.5}),
            view_matrix=view_matrix.reindex(columns=invalid_returns.columns),
            view_returns=view_returns,
        )