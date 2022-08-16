"""Implementation of Levenberg-Marquardt algorithm for nonlinear least squares curve fitting.

Main difference between Newton-Gaussian algorithm and LM is introduction of lambda parameter in the main formulation.
We call it - LM parameter.

References:
    (1) https://people.duke.edu/~hpgavin/ce281/lm.pdf
    (2) https://www.ams.org/journals/qam/1944-02-02/S0033-569X-1944-10666-0/S0033-569X-1944-10666-0.pdf
"""

import argparse
from collections import namedtuple
import logging
import time
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit
from sklearn.model_selection import ParameterGrid

CallableArg = Union[float, np.ndarray]
Result = namedtuple("Result", "error hyperparams params")
np_dtype = np.float32


def configure_logging(logging_level) -> logging.Logger:
    """Configures logging module."""
    logger = logging.getLogger("LM-log")
    logger.setLevel(logging_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d: [%(levelname)s] %(message)s")
    formatter.datefmt = "%y-%m-%d %H:%M:%S"
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def func_easy(x: CallableArg, a: float, b: float) -> CallableArg:
    """Function exp(ax + b).

    Called easy because it's expected to be fit easier.

    Args:
        x (CallableArg): input.
        a (float): function parameter.
        b (float): function parameter.

    Returns:
        out (CallableArg): output.
    """
    out = np.exp(a * x + b)

    return out


def func_hard(x: CallableArg, a: float, b: float):
    """Function exp(ax + b).

    Called easy because it's expected to be fit harder.

    Args:
        x (CallableArg): input.
        a (float): function parameter.
        b (float): function parameter.

    Returns:
        out (CallableArg): output.
    """
    out = a * np.cos(b * x) + b * np.sin(a * x)

    return out


def calc_jacobian(func, x: np.ndarray, params: np.ndarray, delta: float = 1e-8):
    """Calculates Jacobian matrix numerically in the given point.

    Args:
        func (Callable): calculates Jacobian for this function.
        x (np.ndarray): data points.
        params (np.ndarray): parameters to be updated.
        delta (float): difference value (i.e. dx).

    Returns:
        J (np.ndarray): Jacobian matrix.
    """
    m = x.shape[0]
    n = params.shape[0]

    J = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            delta_j = np.zeros_like(params)
            delta_j[j] = delta
            delta_params = params + delta_j

            # calculates [f(x + dx) - f(x)] / dx
            diff = func(x[i], *delta_params) - func(x[i], *params)
            J[i, j] = diff / delta

    return J


def calc_error(func, x: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
    """Calculates residuals for the given function and parameters.

    Args:
        func (Callable): function that fits the curve.
        x (np.ndarray): data points.
        y (np.ndarray): target values.
        params (np.ndarray): parameters to be updated.

    Returns:
        error (float): sum of squared residuals.
    """
    r = y - func(x, *params)
    error = np.sum(np.power(r, 2))

    return error


def run_single_iteration(func, x: np.ndarray, y: np.ndarray, params: np.ndarray, lmbd: float) -> Tuple[np.ndarray, float]:
    """Runs single-update iteration and calculates error.

    Args:
        x (np.ndarray): data points.
        y (np.ndarray): target values.
        params (np.ndarray): parameters to be updated.
        lmbd (float): LM parameter.

    Returns:
        params_upd, curr_error (tuple): updated parameters and error.
    """
    m = params.shape[0]

    # (J.T @ J + lmbd * I) @ params = J.T @ r
    J = calc_jacobian(func, x, params)
    A = J.T @ J + lmbd * np.eye(m)
    r = y - func(x, *params)
    b = J.T @ r

    params_delta = linalg.solve(A, b)
    params_upd = params + params_delta
    curr_error = calc_error(func, x, y, params_upd)

    return params_upd, curr_error


def levenberg_marquardt(func, x: np.ndarray, y: np.ndarray, init_params: np.ndarray, init_lmbd: float,
                        logger: logging.Logger, num_iters: int = 100, lmbd_inc: float = 11, lmbd_dec: float = 9,
                        lmbd_cap: float = 1e7, error_tol: float = 1e-5) -> np.ndarray:
    params = np.copy(init_params)
    lmbd = init_lmbd

    num_iters_len = len(str(num_iters)) - 1
    for curr_iter in range(num_iters):
        params, curr_error = run_single_iteration(func, x, y, params, lmbd)
        logger.debug(f"Current iteration {curr_iter:>{num_iters_len}}: error={curr_error:.5f} lmbd={lmbd:.5f}")

        if curr_error < error_tol:
            logger.warning("Error is lower than the threshold. Early stopping engaged.")
            return params

        # updating lambda parameter as suggested in the (1)
        _, error_same_lmbd = run_single_iteration(func, x, y, params, lmbd)
        smaller_lmbd = lmbd / lmbd_dec
        _, error_smaller_lmbd = run_single_iteration(func, x, y, params, smaller_lmbd)
        if error_same_lmbd > curr_error and error_smaller_lmbd > curr_error:
            larger_lmbd = lmbd * lmbd_inc
            _, error_larger_lmbd = run_single_iteration(func, x, y, params, larger_lmbd)
            while error_larger_lmbd > curr_error and larger_lmbd < lmbd_cap:
                larger_lmbd = larger_lmbd * lmbd_inc
                _, error_larger_lmbd = run_single_iteration(func, x, y, params, larger_lmbd)

            lmbd = min(larger_lmbd, lmbd_cap)
        elif error_same_lmbd < curr_error and error_smaller_lmbd > curr_error:
            lmbd = lmbd
        elif error_same_lmbd > curr_error and error_smaller_lmbd < curr_error:
            lmbd = smaller_lmbd
        elif error_same_lmbd < error_smaller_lmbd:
            lmbd = lmbd
        else:
            lmbd = smaller_lmbd

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="Number of data samples (parameter n in formulae).", type=int, required=True)
    parser.add_argument("--logging_level", help="Logging level (e.g. debug or info).", type=str, default="info")
    args, _ = parser.parse_known_args()

    logger = configure_logging(getattr(logging, args.logging_level.upper()))

    target = lambda x: 100 * np.cos(102 * x) + 102 * np.sin(100 * x)
    x = np.linspace(0, 100, args.num_samples)

    # target = lambda x: np.exp(0.3 * x + 0.5)
    # noise = np.random.randn(args.num_samples) * 0.1
    # x = np.linspace(0, 10, args.num_samples)

    y = target(x)
    plt.plot(x, y, color="blue")

    init_params = np.array([100, 100], dtype=np.float32)

    params_scipy = curve_fit(func_hard, x, y, init_params)[0]
    logger.info(f"scipy: {params_scipy}")
    plt.plot(x, func_hard(x, *params_scipy), color="green")

    hyperparam_grid = ParameterGrid({"init_lmbd": np.linspace(1e-4, 1e-1, 5)})

    results = []
    for hyperparams in hyperparam_grid:
        logger.info(f"hyperparams: {hyperparams}")
        params = levenberg_marquardt(func_hard, x, y, init_params, logger=logger, **hyperparams)
        curr_error = calc_error(func_hard, x, y, params)
        results.append(Result(curr_error, hyperparams, params))

    results = sorted(results, key=lambda res: res.error)
    errors = list(map(lambda res: res.error, results))
    logger.info(f"optimal hyperparams: {results[0].hyperparams}")

    params_lm = results[0].params
    logger.info(f"levenberg_marquardt: {params_lm}")
    plt.plot(x, func_hard(x, *params_lm), color="red")

    plt.show()
