"""Diagnose maximum-Sharpe SLSQP convergence on two failed windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

DATA_PATH = "data/processed/us_equities_simple_returns.csv"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04

EXPERIMENTS = [
    {
        "name": "126-day lookback, 50% maximum weight",
        "start": "2018-05-03",
        "end": "2018-10-30",
        "expected_observations": 126,
        "max_weight": 0.50,
    },
    {
        "name": "252-day lookback, 30% maximum weight",
        "start": "2021-10-29",
        "end": "2022-10-28",
        "expected_observations": 252,
        "max_weight": 0.30,
    },
]


def negative_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Return negative annualized Sharpe ratio."""

    excess_return = float(weights @ mu - RISK_FREE_RATE)
    variance = float(weights @ sigma @ weights)

    if variance <= 1e-24:
        return 1e10

    return -excess_return / np.sqrt(variance)


def negative_sharpe_gradient(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Calculate the analytical gradient of negative Sharpe."""

    excess_return = float(weights @ mu - RISK_FREE_RATE)
    variance = float(weights @ sigma @ weights)

    if variance <= 1e-24:
        return np.zeros_like(weights)

    volatility = np.sqrt(variance)

    return (
        -mu / volatility
        + excess_return * (sigma @ weights) / volatility**3
    )


def weight_sum_constraint(
    weights: np.ndarray,
    *_args: np.ndarray,
) -> float:
    """Enforce a fully invested portfolio."""

    return float(weights.sum() - 1.0)


def weight_sum_jacobian(
    weights: np.ndarray,
    *_args: np.ndarray,
) -> np.ndarray:
    """Return the gradient of the weight-sum constraint."""

    return np.ones_like(weights)


def main() -> None:
    returns = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    ).dropna()

    returns = returns.sort_index()

    for experiment in EXPERIMENTS:
        print("\n" + "=" * 78)
        print(experiment["name"])
        print("=" * 78)

        window = returns.loc[
            experiment["start"]:experiment["end"]
        ]

        expected_observations = experiment["expected_observations"]

        if len(window) != expected_observations:
            print("Unexpected observation count:", len(window))
            print("Expected:", expected_observations)
            continue

        mu = window.mean().to_numpy(dtype=float) * TRADING_DAYS
        sigma = window.cov().to_numpy(dtype=float) * TRADING_DAYS

        n_assets = len(mu)
        max_weight = experiment["max_weight"]
        bounds = [(0.0, max_weight)] * n_assets

        equal_weights = np.full(n_assets, 1.0 / n_assets)

        maximum_return = linprog(
            c=-mu,
            A_eq=np.ones((1, n_assets)),
            b_eq=[1.0],
            bounds=bounds,
            method="highs",
        )

        minimum_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
        maximum_eigenvalue = float(np.linalg.eigvalsh(sigma).max())

        print("Observations:", len(window))
        print("Assets:", n_assets)
        print("Maximum expected asset return:", mu.max())
        print("Minimum expected asset return:", mu.min())
        print("Equal-weight expected return:", equal_weights @ mu)
        print("Minimum covariance eigenvalue:", minimum_eigenvalue)
        print("Maximum covariance eigenvalue:", maximum_eigenvalue)

        if not maximum_return.success:
            print("Maximum-return linear program failed:")
            print(maximum_return.message)
            continue

        feasible_maximum_return = float(-maximum_return.fun)

        print(
            "Maximum feasible portfolio return:",
            feasible_maximum_return,
        )
        print(
            "Equal-weight excess return:",
            float(equal_weights @ mu - RISK_FREE_RATE),
        )
        print(
            "Maximum feasible excess return:",
            feasible_maximum_return - RISK_FREE_RATE,
        )

        constraints = [
            {
                "type": "eq",
                "fun": weight_sum_constraint,
                "jac": weight_sum_jacobian,
            }
        ]

        starts = {
            "equal_weight": equal_weights,
            "maximum_return": maximum_return.x,
        }

        gradient_methods = [
            ("numerical_gradient", None),
            ("analytical_gradient", negative_sharpe_gradient),
        ]

        for start_name, initial in starts.items():
            for gradient_name, jacobian in gradient_methods:
                result = minimize(
                    fun=negative_sharpe,
                    x0=initial,
                    args=(mu, sigma),
                    jac=jacobian,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "maxiter": 3000,
                        "ftol": 1e-12,
                        "disp": False,
                    },
                )

                weight_sum = float(result.x.sum())
                minimum_weight = float(result.x.min())
                maximum_weight = float(result.x.max())

                gradient = negative_sharpe_gradient(
                    result.x,
                    mu,
                    sigma,
                )

                print("\n" + "-" * 60)
                print("Start:", start_name)
                print("Gradient:", gradient_name)
                print("Success:", result.success)
                print("Message:", result.message)
                print("Iterations:", result.nit)
                print("Objective:", result.fun)
                print("Sharpe:", -result.fun)
                print("Weight sum:", weight_sum)
                print("Minimum weight:", minimum_weight)
                print("Maximum weight:", maximum_weight)
                print(
                    "Maximum bound violation:",
                    max(
                        0.0,
                        -minimum_weight,
                        maximum_weight - max_weight,
                    ),
                )
                print(
                    "Gradient norm:",
                    float(np.linalg.norm(gradient)),
                )


if __name__ == "__main__":
    main()