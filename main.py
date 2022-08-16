import logging
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit
from sklearn.model_selection import ParameterGrid

CallableArg = Union[float, np.ndarray]


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d: [%(levelname)s] %(message)s",
        datefmt = "%y-%m-%d %H:%M:%S"
    )
    logging.Formatter.converter = time.gmtime


def func_easy(x: CallableArg, a: float, b: float):
    return np.exp(a * x + b)


def func_hard(x: CallableArg, a: float, b: float):
    return a * np.cos(b * x) + b * np.sin(a * x)


def calc_jacobian(func, x: np.ndarray, params: np.ndarray, delta: float = 1e-8):
    m = x.shape[0]
    n = params.shape[0]

    J = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            delta_j = np.zeros_like(params)
            delta_j[j] = delta
            delta_params = params + delta_j

            diff = func(x[i], *delta_params) - func(x[i], *params)
            J[i, j] = diff / delta

    return J


def calc_error(func, x: np.ndarray, y: np.ndarray, params: np.ndarray):
    r = y - func(x, *params)
    error = np.sum(np.power(r, 2))

    return error


def run_single_iteration(func, x: np.ndarray, y: np.ndarray, params: np.ndarray, lmbd: float):
    m = params.shape[0]

    J = calc_jacobian(func, x, params)
    A = J.T @ J + lmbd * np.eye(m)
    r = y - func(x, *params)
    b = J.T @ r

    params_delta = linalg.solve(A, b)
    params_upd = params + params_delta
    curr_error = calc_error(func, x, y, params_upd)

    return params_upd, curr_error


def levenberg_marquardt(func, x: np.ndarray, y: np.ndarray, init_params: np.ndarray, init_lmbd: float,
                        num_iters: int = 100, velocity: float = 1.1, error_tol: float = 1e-5, lmbd_tol: float = 5):
    params = np.copy(init_params)
    lmbd = init_lmbd

    num_iters_len = len(str(num_iters)) - 1
    for curr_iter in range(num_iters):
        params, curr_error = run_single_iteration(func, x, y, params, lmbd)
        logging.debug(f"Current iteration {curr_iter:>{num_iters_len}}: error={curr_error:.5f} lmbd={lmbd:.5f}")

        if curr_error < error_tol:
            logging.warning("Error is lower than the threshold. Early stopping engaged.")
            return params

        _, error_same_lmbd = run_single_iteration(func, x, y, params, lmbd)
        smaller_lmbd = lmbd / velocity
        _, error_smaller_lmbd = run_single_iteration(func, x, y, params, smaller_lmbd)
        if error_same_lmbd > curr_error and error_smaller_lmbd > curr_error:
            larger_lmbd = lmbd * velocity
            _, error_larger_lmbd = run_single_iteration(func, x, y, params, larger_lmbd)
            while error_larger_lmbd > curr_error and larger_lmbd < lmbd_tol:
                larger_lmbd = larger_lmbd * velocity
                _, error_larger_lmbd = run_single_iteration(func, x, y, params, larger_lmbd)

            lmbd = min(larger_lmbd, lmbd_tol)
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
    configure_logging()
    n = 100

    target = lambda x: 100 * np.cos(102 * x) + 102 * np.sin(100 * x)
    x = np.linspace(0, 100, n)

    # target = lambda x: 5 * x + 3
    # x = np.linspace(-10, 10, n)

    # target = lambda x: 5 * x**2 + 7
    # noise = np.random.randn(100) * 0.1
    # x = np.linspace(-5, 5, n)

    # target = lambda x: np.exp(0.3 * x + 0.5)
    # noise = np.random.randn(n) * 0.1
    # x = np.linspace(0, 10, n)

    y = target(x)
    plt.plot(x, y, color="blue")

    init_params = np.array([100, 100], dtype=np.float32)

    params_scipy = curve_fit(func_hard, x, y, init_params)[0]
    logging.info(f"scipy: {params_scipy}")
    plt.plot(x, func_hard(x, *params_scipy), color="green")

    hyperparam_grid = ParameterGrid({"init_lmbd": np.linspace(1e-3, 1e-1, 10),
                                     "velocity": np.linspace(1.1, 2, 5)})

    results = []
    for hyperparams in hyperparam_grid:
        logging.info(f"hyperparams: {hyperparams}")
        params = levenberg_marquardt(func_hard, x, y, init_params, **hyperparams)
        curr_error = calc_error(func_hard, x, y, params)
        results.append((curr_error, hyperparams, params))

    results = sorted(results, key=lambda x: x[0])
    errors = list(map(lambda r: r[0], results))
    logging.info(f"optimal hyperparams: {results[0][1]}")

    params_lm = results[0][2]
    logging.info(f"levenberg_marquardt: {params_lm}")
    plt.plot(x, func_hard(x, *params_lm), color="red")

    plt.show()
