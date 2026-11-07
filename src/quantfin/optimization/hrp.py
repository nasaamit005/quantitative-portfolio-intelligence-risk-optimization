
"""Hierarchical Risk Parity portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate historical simple returns for HRP optimization."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("Returns data cannot be empty.")

    if returns.columns.has_duplicates:
        raise ValueError("Asset labels must be unique.")

    data = returns.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if data.shape[0] < 2:
        raise ValueError("At least two complete observations are required.")

    if data.shape[1] < 2:
        raise ValueError("HRP requires at least two assets.")

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Returns contain non-finite observations.")

    if (data <= -1.0).any().any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    if (data.std(ddof=1) <= 0).any():
        raise ValueError("Every asset must have positive return volatility.")

    return data.astype(float)


def _cluster_variance(
    covariance: pd.DataFrame,
    assets: list[str],
) -> float:
    """Calculate inverse-variance-weighted cluster variance."""

    cov = covariance.loc[assets, assets].to_numpy(dtype=float)

    diagonal = np.diag(cov)

    if (diagonal <= 0).any():
        raise ValueError("Asset variances must be positive.")

    inverse_variance = 1.0 / diagonal
    weights = inverse_variance / inverse_variance.sum()

    variance = float(weights @ cov @ weights)

    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Cluster variance must be finite and positive.")

    return variance


def hierarchical_risk_parity_weights(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """
    Calculate hierarchical risk parity portfolio weights.

    Steps:
    1. Estimate covariance and correlation.
    2. Convert correlation into a distance matrix.
    3. Perform hierarchical clustering.
    4. Quasi-diagonalize the asset ordering.
    5. Allocate weights through recursive bisection.
    """

    data = _validate_returns(returns)

    allowed_methods = {
        "single",
        "complete",
        "average",
        "weighted",
    }

    if linkage_method not in allowed_methods:
        raise ValueError(
            f"Unsupported linkage method: {linkage_method}"
        )

    covariance = data.cov()
    correlation = data.corr()

    correlation_values = correlation.to_numpy(dtype=float)

    if not np.isfinite(correlation_values).all():
        raise ValueError("Correlation matrix contains non-finite values.")

    # Convert correlation into hierarchical clustering distances.
    distance = np.sqrt(
        np.clip((1.0 - correlation_values) / 2.0, 0.0, 1.0)
    )

    np.fill_diagonal(distance, 0.0)

    condensed_distance = squareform(distance, checks=True)

    # Hierarchical clustering.
    linkage_matrix = linkage(
        condensed_distance,
        method=linkage_method,
        optimal_ordering=True,
    )

    # Obtain quasi-diagonalized asset order.
    ordered_indices = leaves_list(linkage_matrix)

    ordered_assets = [
        data.columns[index] for index in ordered_indices
    ]

    # Recursive bisection.
    weights = pd.Series(1.0, index=ordered_assets, dtype=float)

    clusters = [ordered_assets]

    while clusters:
        next_clusters = []

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            midpoint = len(cluster) // 2

            left_cluster = cluster[:midpoint]
            right_cluster = cluster[midpoint:]

            left_variance = _cluster_variance(
                covariance,
                left_cluster,
            )

            right_variance = _cluster_variance(
                covariance,
                right_cluster,
            )

            allocation_left = (
                right_variance
                / (left_variance + right_variance)
            )

            allocation_right = 1.0 - allocation_left

            weights.loc[left_cluster] *= allocation_left
            weights.loc[right_cluster] *= allocation_right

            next_clusters.extend(
                [left_cluster, right_cluster]
            )

        clusters = next_clusters

    # Restore original asset order.
    weights = weights.reindex(data.columns)
    weights.name = "hrp_weight"

    # Final numerical normalization.
    weights = weights / weights.sum()

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("HRP produced non-finite portfolio weights.")

    if (weights < -1e-10).any():
        raise ValueError("HRP produced negative portfolio weights.")

    return weights
