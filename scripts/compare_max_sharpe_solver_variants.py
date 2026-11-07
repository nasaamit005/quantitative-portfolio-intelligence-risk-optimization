
"""Compare SLSQP gradients and starting points on failed production windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from quantfin.optimization import PortfolioConstraints

ROOT = Path(__file__).resolve().parents[1]

RETURNS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "us_equities_simple_returns.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "robustness_analysis"
    / "max_sharpe_solver_variants.csv"
)

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
SOLVER_MAX_ITERATIONS = 3000
SOLVER_FTOL = 1e-12

WINDOWS = [
    {
        "name": "126d_first_failure",
        "start": "2018-05-03",
        "end": "2018-10-30",
        "max_weight": 0.50,
    },
    {
        "name": "126d_second_failure",
        "start": "2018-06-04",
        "end": "2018-11-29",
        "max_weight": 0.50,
    },
    {
        "name": "252d_failure",
        "start": "2021-10-29",
        "end": "2022-10-28",
        "max_weight": 0.30,
    },
]


def load_returns() -> pd.DataFrame:
    """Load and validate the processed equity returns."""

    frame = pd.read_csv(RETURNS_PATH)

    date_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower() == "date"
        ),
        None,
    )

    if date_column is None:
        raise ValueError("Dataset must contain a Date column.")

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    if frame[date_column].isna().any():
        raise ValueError("Invalid dates in the dataset.")

    frame = (
        frame.rename(columns={date_column: "date"})
        .set_index("date")
        .sort_index()
    )

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all").dropna()

    if frame.empty:
        raise ValueError("The returns dataset is empty.")

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite return observations found.")

    return frame


def negative_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Calculate the negative annualized Sharpe ratio."""

    variance = float(weights @ covariance @ weights)

    if not np.isfinite(variance) or variance <= 0:
        return 1e10

    excess_return = float(weights @ mu - RISK_FREE_RATE)

    return -excess_return / np.sqrt(variance)


def negative_sharpe_gradient(
    weights: np.ndarray,
    mu: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Analytical gradient of the negative Sharpe ratio."""

    variance = float(weights @ covariance @ weights)

    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Portfolio variance must be positive.")

    volatility = np.sqrt(variance)
    excess_return = float(weights @ mu - RISK_FREE_RATE)

    gradient = (
        -mu / volatility
        + excess_return
        * (covariance @ weights)
        / (variance ** 1.5)
    )

    if not np.isfinite(gradient).all():
        raise ValueError("Analytical gradient contains non-finite values.")

    return gradient


def maximum_return_start(
    mu: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Find a feasible maximum-expected-return starting portfolio."""

    result = linprog(
        c=-mu,
        A_eq=np.ones((1, len(mu))),
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            f"Maximum-return initialization failed: {result.message}"
        )

    return np.asarray(result.x, dtype=float)


def run_solver(
    mu: np.ndarray,
    covariance: np.ndarray,
    bounds: list[tuple[float, float]],
    initial_weights: np.ndarray,
    use_analytical_gradient: bool,
) -> dict[str, Any]:
    """Run SLSQP with the specified gradient and initial weights."""

    equality_constraint = {
        "type": "eq",
        "fun": lambda weights: float(weights.sum() - 1.0),
        "jac": lambda weights: np.ones_like(weights),
    }

    if use_analytical_gradient:
        def jacobian(
            weights: np.ndarray,
            *args: np.ndarray,
        ) -> np.ndarray:
            """Accept the same extra arguments passed by SciPy."""

            gradient_mu, gradient_covariance = args

            return negative_sharpe_gradient(
                weights,
                gradient_mu,
                gradient_covariance,
            )
    else:
        jacobian = None

    try:
        result = minimize(
            fun=negative_sharpe,
            x0=initial_weights.copy(),
            args=(mu, covariance),
            method="SLSQP",
            jac=jacobian,
            bounds=bounds,
            constraints=[equality_constraint],
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FTOL,
            },
        )

        weights = np.asarray(result.x, dtype=float)

        if weights.shape != initial_weights.shape:
            raise ValueError("Optimizer returned an unexpected weight shape.")

        if not np.isfinite(weights).all():
            raise ValueError("Optimizer returned non-finite weights.")

        weight_sum = float(weights.sum())
        minimum_weight = float(weights.min())
        maximum_weight = float(weights.max())

        lower_bound = max(bound[0] for bound in bounds)
        upper_bound = min(bound[1] for bound in bounds)

        bound_violation = max(
            0.0,
            lower_bound - minimum_weight,
            maximum_weight - upper_bound,
        )

        feasible = (
            abs(weight_sum - 1.0) <= 1e-6
            and bound_violation <= 1e-6
        )

        variance = float(weights @ covariance @ weights)

        sharpe = (
            float(weights @ mu - RISK_FREE_RATE)
            / np.sqrt(variance)
            if variance > 0
            else np.nan
        )

        return {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "sharpe_ratio": float(sharpe),
            "weight_sum": weight_sum,
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "bound_violation": float(bound_violation),
            "feasible": bool(feasible),
            "weights": weights,
        }

    except Exception as exc:
        return {
            "success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "iterations": 0,
            "objective": np.nan,
            "sharpe_ratio": np.nan,
            "weight_sum": np.nan,
            "minimum_weight": np.nan,
            "maximum_weight": np.nan,
            "bound_violation": np.nan,
            "feasible": False,
            "weights": np.full(len(mu), np.nan),
        }


def main() -> None:
    """Compare four solver variants on the three failed windows."""

    returns = load_returns()
    records: list[dict[str, Any]] = []

    print("=" * 100)
    print("MAXIMUM-SHARPE SOLVER VARIANT COMPARISON")
    print("=" * 100)

    for window in WINDOWS:
        history = returns.loc[
            window["start"]:window["end"]
        ]

        lookback = len(history)
        max_weight = float(window["max_weight"])

        if lookback < 2:
            raise ValueError(
                f"Invalid or missing window: {window['name']}"
            )

        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=max_weight,
            allow_short_selling=False,
            fully_invested=True,
        )

        bounds = constraints.bounds(len(history.columns))

        mu_series = history.mean() * TRADING_DAYS
        covariance_frame = history.cov() * TRADING_DAYS

        mu = mu_series.to_numpy(dtype=float)
        covariance = covariance_frame.to_numpy(dtype=float)

        if not np.isfinite(mu).all():
            raise ValueError(
                f"Expected returns contain non-finite values: {window['name']}"
            )

        if not np.isfinite(covariance).all():
            raise ValueError(
                f"Covariance contains non-finite values: {window['name']}"
            )

        equal_start = np.full(
            len(mu),
            1.0 / len(mu),
        )

        return_start = maximum_return_start(
            mu,
            bounds,
        )

        eigenvalues = np.linalg.eigvalsh(covariance)

        if eigenvalues.min() <= 0:
            raise ValueError(
                f"Covariance is not positive definite: {window['name']}"
            )

        print("\n" + "-" * 100)
        print(f"Window: {window['name']}")
        print(
            f"Dates: {history.index.min().date()} to "
            f"{history.index.max().date()}"
        )
        print(f"Observations: {lookback}")
        print(f"Maximum weight: {max_weight:.2f}")
        print(
            f"Minimum covariance eigenvalue: "
            f"{eigenvalues.min():.10f}"
        )
        print(
            f"Covariance condition number: "
            f"{eigenvalues.max() / eigenvalues.min():.6f}"
        )

        initializations = {
            "equal_weight": equal_start,
            "maximum_return": return_start,
        }

        for start_name, initial_weights in initializations.items():
            for analytical in (False, True):
                gradient_name = (
                    "analytical"
                    if analytical
                    else "numerical"
                )

                result = run_solver(
                    mu=mu,
                    covariance=covariance,
                    bounds=bounds,
                    initial_weights=initial_weights,
                    use_analytical_gradient=analytical,
                )

                record = {
                    "window": window["name"],
                    "observations": lookback,
                    "max_weight": max_weight,
                    "starting_point": start_name,
                    "gradient": gradient_name,
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "weights"
                    },
                }

                records.append(record)

                print(
                    f"\nStart={start_name:<15} "
                    f"Gradient={gradient_name:<10}"
                )
                print(f"  Success:       {result['success']}")
                print(f"  Message:       {result['message']}")
                print(f"  Iterations:    {result['iterations']}")
                print(
                    f"  Sharpe:        "
                    f"{result['sharpe_ratio']:.12f}"
                )
                print(
                    f"  Weight sum:    "
                    f"{result['weight_sum']:.12f}"
                )
                print(
                    f"  Minimum:       "
                    f"{result['minimum_weight']:.8f}"
                )
                print(
                    f"  Maximum:       "
                    f"{result['maximum_weight']:.8f}"
                )
                print(
                    f"  Bound violation: "
                    f"{result['bound_violation']}"
                )
                print(f"  Feasible:      {result['feasible']}")
                print(
                    "  Weights:       "
                    + np.array2string(
                        result["weights"],
                        precision=6,
                        suppress_small=True,
                    )
                )

    output = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        output[
            [
                "window",
                "starting_point",
                "gradient",
                "success",
                "iterations",
                "sharpe_ratio",
                "feasible",
            ]
        ].to_string(index=False)
    )

    print(f"\nTotal solver runs: {len(output)}")
    print(f"Successful runs: {int(output['success'].sum())}")
    print(f"Failed runs: {int((~output['success']).sum())}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

