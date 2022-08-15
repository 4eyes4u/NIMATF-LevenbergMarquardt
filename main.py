import logging
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

CallableArg = Union[float, np.ndarray]


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d: [%(levelname)s] %(message)s",
        datefmt = "%y-%m-%d %H:%M:%S"
    )
    logging.Formatter.converter = time.gmtime


def func(x: CallableArg, beta: np.ndarray):
    return np.exp(beta[0] * x + beta[1])


# def func(x: CallableArg, beta: np.ndarray):
#     return beta[0] * x + beta[1]


# def func(x: CallableArg, beta: np.ndarray):
#     return beta[0] * np.cos(beta[1] * x) + beta[1] * np.sin(beta[0] * x)


# def func(x: CallableArg, beta: np.ndarray):
#     return beta[0] * x ** 2 + beta[1]


def calc_jacobian(x: np.ndarray, beta: np.ndarray, delta: float = 1e-8):
    m = x.shape[0]
    n = beta.shape[0]

    J = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            delta_j = np.zeros_like(beta)
            delta_j[j] = delta

            diff = func(x[i], beta + delta_j) - func(x[i], beta)
            J[i, j] = diff / delta

    return J


def calc_error(x: np.ndarray, y: np.ndarray, beta: np.ndarray):
    r = y - func(x, beta)
    error = np.sum(np.power(r, 2))

    return error


# def solve_linear_system(A: np.ndarray, b: np.ndarray, x: np.ndarray):
#     L = linalg.cholesky(A, lower=True)


def run_single_iteration(x: np.ndarray, y: np.ndarray, beta: np.ndarray, lmbd: float):
    m = init_beta.shape[0]

    J = calc_jacobian(x, beta)
    A = J.T @ J + lmbd * np.eye(m)
    r = y - func(x, beta)
    b = J.T @ r

    beta_delta = linalg.solve(A, b)
    beta_upd = beta + beta_delta
    curr_error = calc_error(x, y, beta_upd)

    return beta_upd, curr_error


def levenberg_marquardt(x: np.ndarray, y: np.ndarray, init_beta: np.ndarray, init_lmbd: np.ndarray,
                        num_iters: int = 100, velocity: float = 1.1, error_tol: float = 1e-5, lmbd_tol: float = 5):
    beta = np.copy(init_beta)
    lmbd = init_lmbd

    num_iters_len = len(str(num_iters)) - 1
    for curr_iter in range(num_iters):
        beta, curr_error = run_single_iteration(x, y, beta, lmbd)

        logging.info(f"Current iteration {curr_iter:>{num_iters_len}}: error={curr_error:.5f} lmbd={lmbd:.5f}")

        if curr_error < error_tol:
            logging.warning("Error is lower than the threshold. Early stopping engaged.")
            return beta

        _, error_same_lmbd = run_single_iteration(x, y, beta, lmbd)
        smaller_lmbd = lmbd / velocity
        _, error_smaller_lmbd = run_single_iteration(x, y, beta, smaller_lmbd)
        if error_same_lmbd > curr_error and error_smaller_lmbd > curr_error:
            larger_lmbd = lmbd * velocity
            _, error_larger_lmbd = run_single_iteration(x, y, beta, larger_lmbd)
            while error_larger_lmbd > curr_error and larger_lmbd < lmbd_tol:
                larger_lmbd = larger_lmbd * velocity
                _, error_larger_lmbd = run_single_iteration(x, y, beta, larger_lmbd)

            lmbd = min(larger_lmbd, lmbd_tol)
        elif error_same_lmbd < curr_error and error_smaller_lmbd > curr_error:
            lmbd = lmbd
        elif error_same_lmbd > curr_error and error_smaller_lmbd < curr_error:
            lmbd = smaller_lmbd
        elif error_same_lmbd < error_smaller_lmbd:
            lmbd = lmbd
        else:
            lmbd = smaller_lmbd

    return beta


if __name__ == "__main__":
    configure_logging()
    n = 100

    # target = lambda x: 100 * np.cos(102 * x) + 102 * np.sin(100 * x)
    # x = np.linspace(0, 100, n)

    # target = lambda x: 5 * x + 3
    # x = np.linspace(-10, 10, n)

    # target = lambda x: 5 * x**2 + 7
    # noise = np.random.randn(100) * 0.1
    # x = np.linspace(-5, 5, n)

    target = lambda x: np.exp(0.3 * x + 0.5)
    noise = np.random.randn(n) * 0.1
    x = np.linspace(0, 10, n)

    y = target(x)

    init_beta = np.array([1, 1], dtype=np.float32)
    init_lmbd = 0.001
    beta = levenberg_marquardt(x + noise, y, init_beta, init_lmbd)
    logging.info(f"Solution: {beta}")

    plt.plot(x + noise, y, color="blue")
    # plt.plot(x + noise, y, color="green")
    plt.plot(x + noise, func(x, beta), color="red")
    plt.show()
